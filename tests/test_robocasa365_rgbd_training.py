from __future__ import annotations

import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class RoboCasa365RGBDTrainingContractTests(unittest.TestCase):
    def test_close_fridge_full_cache_job_reuses_frozen_encoding(self) -> None:
        job = (
            REPO_ROOT
            / "deployment/clariden/build_close_fridge_rgbd_cache_xwam.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("--episodes-per-task 0", job)
        self.assertIn("--encoding-input \"$ENCODING_INPUT\"", job)
        self.assertIn("--expected-task-count 1", job)
        self.assertIn("\"episode_count\": 106", job)
        self.assertIn("\"video_count\": 318", job)

    def test_rgbd_model_data_and_short_schedule_are_explicit(self) -> None:
        model = (
            REPO_ROOT
            / "configs/model/wan22_5b_robocasa365_atomic_rgbd.yaml"
        ).read_text(encoding="utf-8")
        data = (
            REPO_ROOT
            / "configs/data/robocasa365_close_fridge_rgbd.yaml"
        ).read_text(encoding="utf-8")
        experiment = (
            REPO_ROOT
            / "configs/experiment/robocasa365_close_fridge_rgbd_resume_gate.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("use_depth: true", model)
        self.assertIn("depth_loss_weight: 1.0", model)
        self.assertIn("clean_action_ratio: 0.0", model)
        self.assertIn("depth_cache_root:", data)
        self.assertIn("depth_encoding_path:", data)
        self.assertIn("depth_cache_manifest:", data)
        self.assertIn("num_training_steps: 4", experiment)
        self.assertIn("trainer_max_steps: 2", experiment)
        self.assertIn("train_subset_size: 8", experiment)

    def test_short_job_runs_batch_then_depth_resume_audit(self) -> None:
        job = (
            REPO_ROOT
            / "deployment/clariden/smoke_close_fridge_rgbd_train_resume_xwam.sbatch"
        ).read_text(encoding="utf-8")
        batch_position = job.index("scripts/audit_robocasa365_batch.py")
        initial_position = job.index("phase=initial_step_0_to_2")
        resume_position = job.index("phase=resume_step_2_to_4")
        self.assertLess(batch_position, initial_position)
        self.assertLess(initial_position, resume_position)
        self.assertIn("--expect-depth", job)
        self.assertIn("trainer_max_steps=4", job)

    def test_runner_rejects_missing_or_misaligned_depth_batch(self) -> None:
        runner = (REPO_ROOT / "runners/xwam_runner.py").read_text(encoding="utf-8")
        self.assertIn('raise KeyError("use_depth=true但batch缺少depths")', runner)
        self.assertIn('batch["depths"].shape != batch["video"].shape', runner)

    def test_close_fridge_cache_manifest_is_atomic_only(self) -> None:
        payload = json.loads(
            (
                REPO_ROOT
                / "configs/tasks/robocasa365_rgbd_close_fridge.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(payload["scope"], "atomic_only")
        self.assertEqual(payload["tasks"], ["CloseFridge"])


if __name__ == "__main__":
    unittest.main()
