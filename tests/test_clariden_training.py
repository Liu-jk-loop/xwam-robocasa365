from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from project_tools.clariden_training import (
    build_clariden_4gpu_resume_report,
    resolve_clariden_initial_checkpoint,
)


METRICS = (
    "train/video_loss: 1.0, train/action_loss: 0.5, "
    "train/proprio_loss: 0.25, "
    "train/action_proprio_supervision_ratio: 1.0, "
    "train/task_index: 0.0, train/depth_loss: 0.0, train/loss: 1.75"
)


def _write_run(
    root: Path,
    *,
    name: str,
    global_step: int,
    metric_steps: list[int],
    resume_checkpoint: str | None,
    commit: str,
    optimizer_shard_prefix: str = "",
    use_depth: bool = False,
) -> dict[str, str]:
    run_root = root / name
    checkpoint = run_root / f"step={global_step}.ckpt"
    state_root = checkpoint / "checkpoint"
    state_root.mkdir(parents=True)
    (state_root / "mp_rank_00_model_states.pt").write_bytes(b"model")
    for rank in range(4):
        shard_name = (
            f"{optimizer_shard_prefix}zero_pp_rank_{rank}_"
            "mp_rank_00_optim_states.pt"
        )
        (state_root / shard_name).write_bytes(b"optimizer")

    metadata = {
        "run_id": name,
        "git": {"commit": commit, "dirty": False, "status": []},
        "dataset": {
            "path": "/data/CloseFridge/lerobot",
            "task_name": "CloseFridge",
            "subset_indices": list(range(8)),
            "shuffle": False,
            "use_depth": use_depth,
        },
        "environment": {"gpu_names": ["NVIDIA GH200 96GB"] * 4},
        "training": {"num_training_steps": 4, "trainer_max_steps": global_step},
        "topology": {
            "world_size": 4,
            "num_nodes": 1,
            "visible_devices": 4,
        },
        "deepspeed": {
            "stage": 2,
            "offload_optimizer": True,
            "exclude_frozen_parameters": True,
        },
        "optimizer": {
            "backend": "deepspeed_cpu_adam",
            "fp32_optimizer_states": True,
        },
    }
    result = {
        "result": "pass",
        "error": None,
        "global_step": global_step,
        "trainer_max_steps": global_step,
        "resume_checkpoint": resume_checkpoint,
        "resume_module_load": (
            {
                "mode": "excluded_frozen_parameters",
                "missing_frozen_count": 438,
                "unexpected_count": 0,
            }
            if resume_checkpoint
            else None
        ),
        "last_checkpoint": str(checkpoint),
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
                    "world_size": 4,
                    "result": "pass",
                    "fp32_only": True,
                    "floating_state_tensors": 2,
                }
            ),
            encoding="utf-8",
        )
    metrics = METRICS.replace(
        "train/depth_loss: 0.0",
        "train/depth_loss: 0.75" if use_depth else "train/depth_loss: 0.0",
    )
    log_path.write_text(
        "".join(f"[METRICS] Step: {step} - {metrics}\n" for step in metric_steps),
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


class ClaridenTrainingAuditTest(unittest.TestCase):
    def test_initial_checkpoint_resolver_writes_validated_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "step=2.ckpt"
            checkpoint.mkdir()
            result = root / "initial_result.json"
            record = root / "checkpoint.txt"
            result.write_text(
                json.dumps(
                    {
                        "result": "pass",
                        "global_step": 2,
                        "error": None,
                        "last_checkpoint": str(checkpoint),
                    }
                ),
                encoding="utf-8",
            )
            resolved = resolve_clariden_initial_checkpoint(
                result_path=result,
                record_path=record,
            )
            self.assertEqual(resolved, checkpoint.resolve())
            self.assertEqual(record.read_text(encoding="utf-8").strip(), str(resolved))

    def test_initial_checkpoint_resolver_rejects_failed_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = root / "initial_result.json"
            result.write_text(
                json.dumps(
                    {
                        "result": "fail",
                        "global_step": 2,
                        "error": "boom",
                        "last_checkpoint": None,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "initial result不是pass"):
                resolve_clariden_initial_checkpoint(
                    result_path=result,
                    record_path=root / "checkpoint.txt",
                )

    def test_four_gpu_step_two_to_four_resume_evidence_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            commit = "a" * 40
            initial = _write_run(
                root,
                name="initial",
                global_step=2,
                metric_steps=[0, 1],
                resume_checkpoint=None,
                commit=commit,
            )
            resumed = _write_run(
                root,
                name="resumed",
                global_step=4,
                metric_steps=[2, 3],
                resume_checkpoint=initial["checkpoint"],
                commit=commit,
            )
            initial.pop("checkpoint")
            resumed.pop("checkpoint")
            report = build_clariden_4gpu_resume_report(
                initial=initial,
                resumed=resumed,
            )
            self.assertTrue(report["ok"], report)
            self.assertTrue(all(report["checks"].values()))
            self.assertEqual(
                report["resumed"]["resume_module_load"]["missing_frozen_count"],
                438,
            )

    def test_bf16_prefixed_deepspeed_optimizer_shards_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            commit = "e" * 40
            initial = _write_run(
                root,
                name="initial",
                global_step=2,
                metric_steps=[0, 1],
                resume_checkpoint=None,
                commit=commit,
                optimizer_shard_prefix="bf16_",
            )
            resumed = _write_run(
                root,
                name="resumed",
                global_step=4,
                metric_steps=[2, 3],
                resume_checkpoint=initial["checkpoint"],
                commit=commit,
                optimizer_shard_prefix="bf16_",
            )
            initial.pop("checkpoint")
            resumed.pop("checkpoint")
            report = build_clariden_4gpu_resume_report(
                initial=initial,
                resumed=resumed,
            )
            self.assertTrue(report["ok"], report)
            self.assertEqual(report["initial"]["checkpoint"]["optimizer_ranks"], [0, 1, 2, 3])
            self.assertTrue(
                report["resumed"]["checks"]["checkpoint_layout"], report
            )

    def test_rgbd_step_two_to_four_requires_positive_depth_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            commit = "f" * 40
            initial = _write_run(
                root,
                name="initial",
                global_step=2,
                metric_steps=[0, 1],
                resume_checkpoint=None,
                commit=commit,
                use_depth=True,
            )
            resumed = _write_run(
                root,
                name="resumed",
                global_step=4,
                metric_steps=[2, 3],
                resume_checkpoint=initial["checkpoint"],
                commit=commit,
                use_depth=True,
            )
            initial.pop("checkpoint")
            resumed.pop("checkpoint")
            report = build_clariden_4gpu_resume_report(
                initial=initial,
                resumed=resumed,
                expect_depth=True,
            )
            self.assertTrue(report["ok"], report)
            self.assertTrue(
                report["initial"]["metrics"]["checks"]["depth_loss_contract"]
            )

    def test_resume_must_use_the_initial_completed_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            commit = "b" * 40
            initial = _write_run(
                root,
                name="initial",
                global_step=2,
                metric_steps=[0, 1],
                resume_checkpoint=None,
                commit=commit,
            )
            wrong = root / "wrong.ckpt"
            wrong.mkdir()
            resumed = _write_run(
                root,
                name="resumed",
                global_step=4,
                metric_steps=[2, 3],
                resume_checkpoint=str(wrong),
                commit=commit,
            )
            initial.pop("checkpoint")
            resumed.pop("checkpoint")
            report = build_clariden_4gpu_resume_report(
                initial=initial,
                resumed=resumed,
            )
            self.assertFalse(report["ok"])
            self.assertFalse(report["checks"]["resume_uses_initial_checkpoint"])

    def test_orchestration_only_commit_delta_can_reuse_initial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial_commit = "c" * 40
            resumed_commit = "d" * 40
            initial = _write_run(
                root,
                name="initial",
                global_step=2,
                metric_steps=[0, 1],
                resume_checkpoint=None,
                commit=initial_commit,
            )
            resumed = _write_run(
                root,
                name="resumed",
                global_step=4,
                metric_steps=[2, 3],
                resume_checkpoint=initial["checkpoint"],
                commit=resumed_commit,
            )
            initial.pop("checkpoint")
            resumed.pop("checkpoint")
            report = build_clariden_4gpu_resume_report(
                initial=initial,
                resumed=resumed,
                commit_compatibility={
                    "ok": True,
                    "mode": "orchestration_only_commit_delta",
                    "initial_commit": initial_commit,
                    "resumed_commit": resumed_commit,
                    "changed_paths": [
                        "deployment/clariden/smoke_train_resume_xwam.sbatch"
                    ],
                    "disallowed_paths": [],
                },
            )
            self.assertTrue(report["ok"], report)
            self.assertTrue(report["checks"]["compatible_training_source"])


if __name__ == "__main__":
    unittest.main()
