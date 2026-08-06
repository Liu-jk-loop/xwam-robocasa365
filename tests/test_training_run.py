from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from project_tools.training_run import (
    collect_git_state,
    resolve_resume_checkpoint,
    resolve_save_last,
    resolve_subset_indices,
    resolve_training_schedule,
    write_json_atomic,
)


class TrainingRunTest(unittest.TestCase):
    def test_m3_layered_configs_keep_debug_and_formal_boundaries_explicit(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        model = (repo_root / "configs/model/wan22_5b_robocasa365_atomic.yaml").read_text()
        hardware = (repo_root / "configs/hardware/a800_80gb_debug.yaml").read_text()
        experiment = (
            repo_root
            / "configs/experiment/robocasa365_close_fridge_m3_overfit.yaml"
        ).read_text()
        self.assertIn("action_dim: 12", model)
        self.assertIn("use_depth: false", model)
        self.assertNotIn("deepspeed_offload_optimizer", model)
        self.assertIn("deepspeed_offload_optimizer: true", hardware)
        self.assertIn("# A800 80GB 单卡调试层", hardware)
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
            "persist_generator_state 当前只允许 M3 单 GPU",
        ):
            self.assertIn(token, entrypoint)
        self.assertIn("xwam_generator_state", runner)
        self.assertIn("Restored training generator state", runner)
        self.assertIn("resume checkpoint 缺少 xwam_generator_state", runner)

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
        self.assertIsNone(
            resolve_subset_indices(dataset_length=100, subset_size=None)
        )
        with self.assertRaisesRegex(ValueError, "超出"):
            resolve_subset_indices(dataset_length=3, subset_size=2, subset_start=2)

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


if __name__ == "__main__":
    unittest.main()
