from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from project_tools.training_run import (
    append_jsonl_fsync,
    collect_memory_snapshot,
    collect_git_state,
    resolve_resume_checkpoint,
    resolve_checkpoint_monitor,
    resolve_save_last,
    resolve_subset_indices,
    resolve_training_schedule,
    validate_excluded_frozen_resume_keys,
    write_json_atomic,
)


class TrainingRunTest(unittest.TestCase):
    def test_m3_layered_configs_keep_debug_and_formal_boundaries_explicit(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        model = (
            repo_root / "configs/model/wan22_5b_robocasa365_atomic.yaml"
        ).read_text()
        hardware = (repo_root / "configs/hardware/a800_80gb_debug.yaml").read_text()
        low_memory_hardware = (
            repo_root / "configs/hardware/a800_80gb_120g_debug.yaml"
        ).read_text()
        experiment = (
            repo_root / "configs/experiment/robocasa365_close_fridge_m3_overfit.yaml"
        ).read_text()
        self.assertIn("action_dim: 12", model)
        self.assertIn("use_depth: false", model)
        self.assertNotIn("deepspeed_offload_optimizer", model)
        self.assertIn("deepspeed_offload_optimizer: true", hardware)
        self.assertIn("# A800 80GB 单卡调试层", hardware)
        self.assertIn("deepspeed_fp32_optimizer_states: true", hardware)
        self.assertIn("deepspeed_fp32_optimizer_states: false", low_memory_hardware)
        self.assertIn("deepspeed_exclude_frozen_parameters: true", low_memory_hardware)
        self.assertIn("禁止用于正式训练", low_memory_hardware)
        self.assertIn("train_subset_size: 1", experiment)
        self.assertIn("trainer_max_steps: 8", experiment)
        self.assertIn("num_training_steps: 10", experiment)
        self.assertIn("persist_generator_state: true", experiment)
        self.assertIn("save_last: link", experiment)

    def test_training_entry_wires_layers_subset_resume_and_run_artifacts(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        entrypoint = (repo_root / "scripts/train_sft.py").read_text()
        runner = (repo_root / "runners/xwam_runner.py").read_text()
        for token in (
            '"hardware_config"',
            '"experiment_config"',
            "Subset(base_train_dataset, subset_indices)",
            "ckpt_path=resume_checkpoint",
            '"deferred_to_trainer"',
            '"run_id": run_id',
            '"global_step": int(trainer.global_step)',
            '"process_max_rss_raw"',
            '"gpu_names"',
            "allow_distributed_generator_state",
            "ResourceAwareModelCheckpoint",
            "checkpoint_save_start",
            "synchronize_after_save",
            '"memory_at_result"',
            "enable_excluded_frozen_resume_loading",
            '"resume_module_load"',
        ):
            self.assertIn(token, entrypoint)
        self.assertIn("xwam_generator_state", runner)
        self.assertIn("xwam_generator_states_by_rank", runner)
        self.assertIn("Restored training generator state", runner)
        self.assertIn("resume checkpoint 缺少 xwam_generator_state", runner)
        self.assertIn("Excluded-frozen resume load accepted", runner)
        self.assertIn("self._apply_restored_generator_state()", runner)
        self.assertIn("train/action_proprio_supervision_ratio", runner)
        self.assertIn("train/task_index", runner)

    def test_m3_three_task_short_config_preserves_original_training_semantics(
        self,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        data_config = (
            repo_root / "configs/data/robocasa365_m3_three_task.yaml"
        ).read_text()
        experiment = (
            repo_root / "configs/experiment/robocasa365_m3_three_task_short.yaml"
        ).read_text()
        factory = (repo_root / "data/dataset_factory.py").read_text()
        self.assertIn("multitask_manifest:", data_config)
        self.assertIn("expected_task_count: 3", data_config)
        self.assertIn("BalancedRoundRobinDataset", factory)
        self.assertIn("clean_action_ratio: 0.5", experiment)
        self.assertIn("trainer_max_steps: 12", experiment)
        self.assertIn("train_shuffle: false", experiment)
        self.assertIn("enable_checkpointing: false", experiment)

    def test_schedule_separates_scheduler_horizon_from_invocation_limit(self) -> None:
        self.assertEqual(
            resolve_training_schedule(num_training_steps=10, trainer_max_steps=8),
            {"num_training_steps": 10, "trainer_max_steps": 8},
        )
        self.assertEqual(
            resolve_training_schedule(num_training_steps=10),
            {"num_training_steps": 10, "trainer_max_steps": 10},
        )
        with self.assertRaisesRegex(ValueError, "不能超过"):
            resolve_training_schedule(num_training_steps=8, trainer_max_steps=10)

    def test_subset_is_fixed_and_range_checked(self) -> None:
        self.assertEqual(
            resolve_subset_indices(dataset_length=100, subset_size=1, subset_start=7),
            (7,),
        )
        self.assertIsNone(resolve_subset_indices(dataset_length=100, subset_size=None))
        with self.assertRaisesRegex(ValueError, "超出"):
            resolve_subset_indices(dataset_length=3, subset_size=2, subset_start=2)

    def test_checkpoint_retention_ranks_multiple_periodic_saves_by_step(self) -> None:
        self.assertEqual(
            resolve_checkpoint_monitor(5),
            {"monitor": "step", "mode": "max"},
        )
        self.assertEqual(
            resolve_checkpoint_monitor(2),
            {"monitor": "step", "mode": "max"},
        )
        for save_top_k in (-1, 0, 1):
            self.assertEqual(resolve_checkpoint_monitor(save_top_k), {})
        with self.assertRaisesRegex(ValueError, "大于等于 -1"):
            resolve_checkpoint_monitor(-2)

    def test_resume_checkpoint_requires_existing_file_or_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "last.ckpt"
            checkpoint.mkdir()
            self.assertEqual(
                resolve_resume_checkpoint(checkpoint), str(checkpoint.resolve())
            )
            self.assertIsNone(resolve_resume_checkpoint(None))
            with self.assertRaises(FileNotFoundError):
                resolve_resume_checkpoint(Path(tmp) / "missing.ckpt")

    def test_excluded_frozen_resume_accepts_only_frozen_missing_parameters(
        self,
    ) -> None:
        report = validate_excluded_frozen_resume_keys(
            missing_keys={"text_encoder.weight", "vae.bias"},
            unexpected_keys=set(),
            frozen_parameter_names={
                "text_encoder.weight",
                "vae.bias",
                "unused_frozen.weight",
            },
        )
        self.assertEqual(report["missing_frozen_count"], 2)
        self.assertEqual(report["unexpected_count"], 0)

        with self.assertRaisesRegex(RuntimeError, "非冻结缺失参数"):
            validate_excluded_frozen_resume_keys(
                missing_keys={"model.trainable_weight"},
                unexpected_keys=set(),
                frozen_parameter_names={"text_encoder.weight"},
            )
        with self.assertRaisesRegex(RuntimeError, "额外参数"):
            validate_excluded_frozen_resume_keys(
                missing_keys={"text_encoder.weight"},
                unexpected_keys={"unknown.weight"},
                frozen_parameter_names={"text_encoder.weight"},
            )

    def test_git_provenance_and_atomic_json_are_machine_readable(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        git_state = collect_git_state(repo_root)
        self.assertRegex(str(git_state.get("commit")), r"^[0-9a-f]{40}$")
        with tempfile.TemporaryDirectory() as tmp:
            output = write_json_atomic(
                Path(tmp) / "run.json", {"git": git_state, "result": "pending"}
            )
            self.assertEqual(json.loads(output.read_text())["result"], "pending")

    def test_save_last_supports_local_symlink_without_boolean_coercion(self) -> None:
        self.assertEqual(resolve_save_last("link"), "link")
        self.assertTrue(resolve_save_last(True))
        self.assertFalse(resolve_save_last(False))
        with self.assertRaises(ValueError):
            resolve_save_last("copy")

    def test_memory_snapshot_and_durable_jsonl_are_machine_readable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cgroup = root / "cgroup"
            cgroup.mkdir()
            (cgroup / "memory.current").write_text("1024\n")
            (cgroup / "memory.peak").write_text("2048\n")
            (cgroup / "memory.max").write_text("4096\n")
            (cgroup / "memory.events").write_text("oom 1\noom_kill 1\n")
            proc_status = root / "status"
            proc_status.write_text("VmRSS:\t10 kB\nVmHWM:\t20 kB\n")

            snapshot = collect_memory_snapshot(
                cgroup_root=cgroup,
                proc_status_path=proc_status,
            )
            self.assertEqual(snapshot["process"]["VmRSS"], 10 * 1024)
            self.assertEqual(snapshot["process"]["VmHWM"], 20 * 1024)
            self.assertEqual(snapshot["cgroup"]["memory.max"], "4096")
            self.assertEqual(snapshot["cgroup"]["memory.events"]["oom_kill"], 1)

            events = root / "events.jsonl"
            append_jsonl_fsync(events, {"event": "start", "memory": snapshot})
            append_jsonl_fsync(events, {"event": "complete"})
            rows = [json.loads(line) for line in events.read_text().splitlines()]
            self.assertEqual([row["event"] for row in rows], ["start", "complete"])


if __name__ == "__main__":
    unittest.main()
