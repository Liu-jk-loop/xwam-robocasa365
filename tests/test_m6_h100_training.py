from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from project_tools.h100_training import (
    build_m6_formal_chunk_report,
    build_m6_gate_report,
    build_m6_preflight_report,
    quarantine_m6_incomplete_checkpoints,
    resolve_epoch_schedule,
    resolve_m6_formal_chunk,
    validate_m6_formal_training_contract,
)


METRICS = (
    "train/video_loss: 1.0, train/action_loss: 0.5, "
    "train/proprio_loss: 0.25, "
    "train/action_proprio_supervision_ratio: 1.0, "
    "train/depth_loss: 0.0, train/loss: 1.75"
)


def _digest(payload: dict) -> str:
    canonical = dict(payload)
    canonical.pop("manifest_digest", None)
    encoded = json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_gate_run(
    root: Path,
    *,
    name: str,
    global_step: int,
    resume_checkpoint: str | None,
    commit: str,
) -> dict[str, str]:
    run_root = root / name
    checkpoint = run_root / "hot" / f"step={global_step}.ckpt"
    state_root = checkpoint / "checkpoint"
    state_root.mkdir(parents=True)
    (state_root / "mp_rank_00_model_states.pt").write_bytes(b"model")
    for rank in range(4):
        (state_root / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt").write_bytes(
            b"optimizer"
        )
    metadata = {
        "run_id": name,
        "git": {"commit": commit, "dirty": False, "status": []},
        "formal_contract": {
            "accelerator": "GH200",
            "zero_stage": 1,
            "checks": {"contract": True},
        },
        "formal_runtime": {
            "accelerator": "GH200",
            "checks": {"runtime": True},
        },
        "training": {
            "num_train_epochs": 5,
            "global_batch_size": 128,
            "num_training_steps": 16390,
            "samples_dropped_per_epoch": 122,
            "manifest_digest": "digest",
        },
        "dataset": {
            "sampler_provenance": {
                "type": "epoch_aligned_distributed",
                "dropped_per_epoch": 122,
            }
        },
        "tracking": {
            "wandb": {
                "enabled": True,
                "project": "xwam-robocasa365",
                "entity": "target-team",
                "run_id": "persistent-run-id",
                "mode": "online",
                "resume": "allow",
                "auth": "api_key_env",
            }
        },
        "checkpoint": {
            "storage": {
                "rolling": {
                    "directory": str((run_root / "hot").resolve()),
                    "interval_steps": 500,
                    "save_top_k": 5,
                    "monitor": "step",
                    "mode": "max",
                },
                "durable": {
                    "directory": str((run_root / "durable").resolve()),
                    "interval_steps": 3000,
                    "save_top_k": -1,
                },
                "final_directory": str((run_root / "durable").resolve()),
            }
        },
    }
    result = {
        "result": "pass",
        "global_step": global_step,
        "trainer_max_steps": global_step,
        "resume_checkpoint": resume_checkpoint,
    }
    metadata_path = run_root / "metadata.json"
    result_path = run_root / "result.json"
    events_path = run_root / "events.jsonl"
    optimizer_dir = run_root / "optimizer"
    log_path = run_root / "console.log"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    result_path.write_text(json.dumps(result), encoding="utf-8")
    events_path.write_text(
        json.dumps(
            {
                "event": "checkpoint_save_complete",
                "global_step": global_step,
                "filepath": str(checkpoint),
                "checkpoint_tier": "rolling",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    optimizer_dir.mkdir()
    for rank in range(4):
        (optimizer_dir / f"optimizer_state_rank_{rank:03d}.json").write_text(
            json.dumps(
                {
                    "global_rank": rank,
                    "result": "pass",
                    "fp32_only": True,
                    "floating_state_tensors": 2,
                }
            ),
            encoding="utf-8",
        )
    log_path.write_text(
        f"[METRICS] Step: {global_step - 1} - {METRICS}\n",
        encoding="utf-8",
    )
    return {
        "metadata_path": str(metadata_path),
        "result_path": str(result_path),
        "events_path": str(events_path),
        "optimizer_dir": str(optimizer_dir),
        "log_path": str(log_path),
        "checkpoint": str(checkpoint.resolve()),
    }


class M6H100TrainingTest(unittest.TestCase):
    def test_formal_chunk_planner_uses_only_the_newest_complete_checkpoint(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoints"
            complete = root / "epoch=0-step=1000.ckpt" / "checkpoint"
            incomplete = root / "epoch=0-step=2000.ckpt" / "checkpoint"
            complete.mkdir(parents=True)
            incomplete.mkdir(parents=True)
            (complete / "mp_rank_00_model_states.pt").write_bytes(b"model")
            for rank in range(4):
                (
                    complete
                    / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
                ).write_bytes(b"optimizer")
            (incomplete / "mp_rank_00_model_states.pt").write_bytes(b"model")

            plan = resolve_m6_formal_chunk(
                root, total_steps=16390, chunk_steps=1000
            )
            self.assertEqual(plan["completed_step"], 1000)
            self.assertEqual(plan["target_step"], 2000)
            self.assertEqual(
                plan["resume_checkpoint"], str(complete.parent.resolve())
            )
            self.assertFalse(plan["final_chunk"])
            self.assertEqual(len(plan["incomplete_checkpoints"]), 1)
            quarantine = Path(tmp) / "quarantine"
            quarantined = quarantine_m6_incomplete_checkpoints(plan, quarantine)
            self.assertFalse(incomplete.parent.exists())
            self.assertTrue(
                (quarantine / "epoch=0-step=2000.ckpt").is_dir()
            )
            self.assertEqual(len(quarantined["quarantined_checkpoints"]), 1)

            final = root / "final-step=16390.ckpt" / "checkpoint"
            final.mkdir(parents=True)
            (final / "mp_rank_00_model_states.pt").write_bytes(b"model")
            for rank in range(4):
                (
                    final
                    / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
                ).write_bytes(b"optimizer")
            finished = resolve_m6_formal_chunk(
                root, total_steps=16390, chunk_steps=1000
            )
            self.assertTrue(finished["already_complete"])
            self.assertEqual(finished["target_step"], 16390)

    def test_restartable_formal_chunk_has_the_full_run_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = _write_gate_run(
                Path(tmp),
                name="formal-chunk",
                global_step=1000,
                resume_checkpoint=None,
                commit="a" * 40,
            )
            artifacts.pop("checkpoint")
            report = build_m6_formal_chunk_report(
                **artifacts,
                expected_step=1000,
                expected_resume_checkpoint=None,
                expected_wandb_run_id="persistent-run-id",
                expected_rolling_checkpoint_root=str(
                    Path(artifacts["metadata_path"]).parent / "hot"
                ),
                expected_durable_checkpoint_root=str(
                    Path(artifacts["metadata_path"]).parent / "durable"
                ),
            )
            self.assertTrue(report["ok"], report)
            self.assertTrue(report["checks"]["within_formal_schedule"])
            self.assertTrue(report["checks"]["exact_resume_source"])
            mismatch = build_m6_formal_chunk_report(
                **artifacts,
                expected_step=1000,
                expected_resume_checkpoint=None,
                expected_wandb_run_id="different-run-id",
                expected_rolling_checkpoint_root=str(
                    Path(artifacts["metadata_path"]).parent / "hot"
                ),
                expected_durable_checkpoint_root=str(
                    Path(artifacts["metadata_path"]).parent / "durable"
                ),
            )
            self.assertFalse(mismatch["ok"])
            self.assertFalse(mismatch["checks"]["expected_wandb_run"])

    def test_formal_chunk_planner_selects_latest_across_hot_and_durable_roots(
        self,
    ) -> None:
        def write_complete(root: Path, step: int) -> Path:
            state = root / f"epoch=1-step={step}.ckpt" / "checkpoint"
            state.mkdir(parents=True)
            (state / "mp_rank_00_model_states.pt").write_bytes(b"model")
            for rank in range(4):
                (
                    state
                    / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
                ).write_bytes(b"optimizer")
            return state.parent.resolve()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hot = root / "iops" / "checkpoints"
            durable = root / "store" / "checkpoints"
            write_complete(durable, 3000)
            hot_3500 = write_complete(hot, 3500)
            incomplete = hot / "epoch=1-step=4000.ckpt" / "checkpoint"
            incomplete.mkdir(parents=True)
            (incomplete / "mp_rank_00_model_states.pt").write_bytes(b"model")

            plan = resolve_m6_formal_chunk(
                [hot, durable], total_steps=16390, chunk_steps=1000
            )
            self.assertEqual(plan["completed_step"], 3500)
            self.assertEqual(plan["target_step"], 4000)
            self.assertEqual(plan["resume_checkpoint"], str(hot_3500))
            self.assertEqual(
                plan["checkpoint_roots"],
                [str(hot.resolve()), str(durable.resolve())],
            )

            hot_quarantine = root / "iops" / "incomplete" / "job"
            durable_quarantine = root / "store" / "incomplete" / "job"
            quarantined = quarantine_m6_incomplete_checkpoints(
                plan, [hot_quarantine, durable_quarantine]
            )
            self.assertEqual(len(quarantined["quarantined_checkpoints"]), 1)
            self.assertTrue(
                (hot_quarantine / "epoch=1-step=4000.ckpt").is_dir()
            )

            durable_3500 = write_complete(durable, 3500)
            same_step = resolve_m6_formal_chunk(
                [hot, durable], total_steps=16390, chunk_steps=1000
            )
            self.assertEqual(same_step["resume_checkpoint"], str(durable_3500))

    def test_gh200_formal_gate_requires_exact_full_checkpoint_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            commit = "f" * 40
            initial = _write_gate_run(
                root,
                name="initial",
                global_step=2,
                resume_checkpoint=None,
                commit=commit,
            )
            resumed = _write_gate_run(
                root,
                name="resumed",
                global_step=4,
                resume_checkpoint=initial["checkpoint"],
                commit=commit,
            )
            initial.pop("checkpoint")
            resumed.pop("checkpoint")
            report = build_m6_gate_report(initial=initial, resumed=resumed)
            self.assertTrue(report["ok"], report)
            self.assertEqual(report["stage"], "M6-formal-accelerator-gate")
            self.assertTrue(report["checks"]["resume_uses_initial_checkpoint"])
            self.assertTrue(report["initial"]["checks"]["checkpoint_layout"])

    def test_five_epoch_schedule_drops_only_global_batch_tail(self) -> None:
        schedule = resolve_epoch_schedule(
            total_samples=1000,
            global_batch_size=128,
            num_train_epochs=5,
        )
        self.assertEqual(schedule["steps_per_epoch"], 7)
        self.assertEqual(schedule["num_training_steps"], 35)
        self.assertEqual(schedule["samples_per_epoch_used"], 896)
        self.assertEqual(schedule["samples_dropped_per_epoch"], 104)
        self.assertEqual(
            resolve_epoch_schedule(
                total_samples=1000,
                global_batch_size=128,
                num_train_epochs=5,
                trainer_max_steps=4,
            )["trainer_max_steps"],
            4,
        )

    def test_preflight_binds_stats_to_the_exact_manifest(self) -> None:
        tasks = [
            {"task_name": f"Task{index:02d}", "valid_clips": 100} for index in range(18)
        ]
        manifest = {
            "schema_version": 1,
            "scope": "atomic_only",
            "split": "pretrain",
            "sampling": "natural_proportional",
            "total_valid_clips": 1800,
            "tasks": tasks,
            "ok": True,
            "result": "pass",
        }
        manifest["manifest_digest"] = _digest(manifest)
        block = {
            "q01": [0.0] * 16,
            "q99": [1.0] * 16,
            "min": [0.0] * 16,
            "max": [1.0] * 16,
        }
        stats = {
            "scope": "atomic_only",
            "split": "pretrain",
            "task_count": 18,
            "manifest_digest": manifest["manifest_digest"],
            "total_frames": 3600,
            "observation.state": block,
            "action": {key: values[:12] for key, values in block.items()},
            "ok": True,
            "result": "pass",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            stats_path = root / "stats.json"
            manifest_path.write_text(json.dumps(manifest))
            stats_path.write_text(json.dumps(stats))
            report = build_m6_preflight_report(manifest_path, stats_path)
            self.assertTrue(report["ok"])
            self.assertEqual(report["schedule"]["steps_per_epoch"], 14)
            self.assertEqual(report["schedule"]["num_training_steps"], 70)

            stats["manifest_digest"] = "wrong"
            stats_path.write_text(json.dumps(stats))
            self.assertFalse(build_m6_preflight_report(manifest_path, stats_path)["ok"])

    def test_h100_configs_freeze_formal_and_fallback_gbs128(self) -> None:
        root = Path(__file__).resolve().parents[1]
        preferred = (root / "configs/hardware/h100x4_80gb_gbs128.yaml").read_text()
        fallback = (root / "configs/hardware/h100x4_80gb_gbs128_safe.yaml").read_text()
        formal = (
            root / "configs/experiment/robocasa365_m6_h100_rgb_formal.yaml"
        ).read_text()
        data = (root / "configs/data/robocasa365_m6_atomic_seen18.yaml").read_text()
        for config in (preferred, fallback):
            self.assertIn("devices: 4", config)
            self.assertIn("global_batch_size: 128", config)
            self.assertIn("precision: bf16-mixed", config)
            self.assertIn("formal_zero_stage: 1", config)
            self.assertIn("deepspeed_stage: 1", config)
            self.assertIn("deepspeed_offload_optimizer: false", config)
            self.assertIn("deepspeed_fp32_optimizer_states: true", config)
            self.assertIn("deepspeed_exclude_frozen_parameters: false", config)
        self.assertIn("batch_size_per_gpu: 4", preferred)
        self.assertIn("accumulate_grad_batches: 8", preferred)
        self.assertIn("batch_size_per_gpu: 2", fallback)
        self.assertIn("accumulate_grad_batches: 16", fallback)
        self.assertIn("num_train_epochs: 5", formal)
        self.assertIn("trainer_max_steps: null", formal)
        self.assertIn("save_final_checkpoint: true", formal)
        self.assertIn("expected_task_count: 18", data)
        self.assertIn("expected_sampling: natural_proportional", data)

    def test_gh200_configs_preserve_the_m6_formal_contract(self) -> None:
        root = Path(__file__).resolve().parents[1]
        preferred = (root / "configs/hardware/gh200x4_96gb_gbs128.yaml").read_text()
        balanced = (
            root / "configs/hardware/gh200x4_96gb_gbs128_balanced.yaml"
        ).read_text()
        fallback = (
            root / "configs/hardware/gh200x4_96gb_gbs128_safe.yaml"
        ).read_text()
        gate = (
            root / "configs/experiment/robocasa365_m6_gh200_gate.yaml"
        ).read_text()
        formal = (
            root / "configs/experiment/robocasa365_m6_gh200_rgb_formal.yaml"
        ).read_text()
        for config in (preferred, balanced, fallback):
            self.assertIn("devices: 4", config)
            self.assertIn("global_batch_size: 128", config)
            self.assertIn("formal_zero_stage: 1", config)
            self.assertIn("deepspeed_stage: 1", config)
            self.assertIn("deepspeed_offload_optimizer: false", config)
            self.assertIn("deepspeed_fp32_optimizer_states: true", config)
            self.assertIn("deepspeed_exclude_frozen_parameters: false", config)
            self.assertIn("m6_formal_guard: true", config)
            self.assertIn("formal_accelerator: GH200", config)
            self.assertIn("formal_minimum_memory_gib: 90", config)
        self.assertIn("batch_size_per_gpu: 16", preferred)
        self.assertIn("accumulate_grad_batches: 2", preferred)
        self.assertIn("batch_size_per_gpu: 8", balanced)
        self.assertIn("accumulate_grad_batches: 4", balanced)
        self.assertIn("batch_size_per_gpu: 4", fallback)
        self.assertIn("accumulate_grad_batches: 8", fallback)
        self.assertIn("trainer_max_steps: 2", gate)
        self.assertIn("save_interval: 2", gate)
        self.assertIn("trainer_max_steps: null", formal)
        self.assertIn("save_interval: 500", formal)
        self.assertIn("save_top_k: 5", formal)
        self.assertIn("durable_save_interval: 3000", formal)
        self.assertIn("durable_save_top_k: -1", formal)
        self.assertIn("save_final_checkpoint: true", formal)
        self.assertIn("enable_wandb: true", formal)
        self.assertIn("wandb_mode: online", formal)
        self.assertIn("wandb_require_api_key: true", formal)

        contract = validate_m6_formal_training_contract(
            {
                "formal_accelerator": "GH200",
                "formal_minimum_memory_gib": 90,
                "batch_size_per_gpu": 16,
                "accumulate_grad_batches": 2,
                "global_batch_size": 128,
                "precision": "bf16-mixed",
                "formal_zero_stage": 1,
                "deepspeed_stage": 1,
                "deepspeed_offload_optimizer": False,
                "deepspeed_fp32_optimizer_states": True,
                "deepspeed_overlap_comm": True,
                "deepspeed_exclude_frozen_parameters": False,
                "use_depth": False,
                "depth_loss_weight": 0.0,
                "num_train_epochs": 5,
                "dataset": {"expected_sampling": "natural_proportional"},
                "train_subset_size": None,
                "train_shuffle": True,
            },
            world_size=4,
        )
        self.assertEqual(contract["accelerator"], "GH200")
        self.assertEqual(contract["zero_stage"], 1)
        self.assertTrue(all(contract["checks"].values()))

        with self.assertRaisesRegex(ValueError, "formal_zero1"):
            validate_m6_formal_training_contract(
                {
                    "formal_accelerator": "GH200",
                    "formal_minimum_memory_gib": 90,
                    "batch_size_per_gpu": 16,
                    "accumulate_grad_batches": 2,
                    "global_batch_size": 128,
                    "precision": "bf16-mixed",
                    "formal_zero_stage": 1,
                    "deepspeed_stage": 2,
                    "deepspeed_offload_optimizer": False,
                    "deepspeed_fp32_optimizer_states": True,
                    "deepspeed_overlap_comm": True,
                    "deepspeed_exclude_frozen_parameters": False,
                    "use_depth": False,
                    "depth_loss_weight": 0.0,
                    "num_train_epochs": 5,
                    "dataset": {"expected_sampling": "natural_proportional"},
                    "train_subset_size": None,
                    "train_shuffle": True,
                },
                world_size=4,
            )

    def test_training_entry_uses_epoch_aligned_sampler_and_formal_guards(self) -> None:
        root = Path(__file__).resolve().parents[1]
        entrypoint = (root / "scripts/train_sft.py").read_text()
        for token in (
            "EpochAlignedDistributedSampler",
            "use_distributed_sampler=train_sampler is None",
            "validate_m6_formal_training_contract",
            "_validate_formal_runtime",
            "OptimizerStateDtypeAudit",
            "final_checkpoint_save_complete",
            "WandbLogger",
            'resume="allow"',
            "wandb_require_api_key",
            "resolve_checkpoint_monitor",
            '"monitor": rolling_monitor.get("monitor")',
            "必须显式提供非空 WANDB_ENTITY",
            '"auth": "api_key_env"',
        ):
            self.assertIn(token, entrypoint)


if __name__ == "__main__":
    unittest.main()
