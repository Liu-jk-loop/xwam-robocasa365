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

    def test_smoke_keeps_capstor_experiment_artifacts(self) -> None:
        job = (
            REPO_ROOT
            / "deployment/clariden/smoke_close_fridge_rgbd_train_resume_xwam.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn('EXP_ROOT="$DEPLOY_CAPSCR/experiments/xwam"', job)
        self.assertIn('RUN_DIR="$EXP_DIR/runs"', job)
        self.assertNotIn("rm -", job)
        self.assertNotIn("cleanup", job.lower())

    def test_formal_rgbd_training_contract_is_explicit(self) -> None:
        data = (
            REPO_ROOT
            / "configs/data/robocasa365_close_fridge_rgbd_formal.yaml"
        ).read_text(encoding="utf-8")
        hardware = (
            REPO_ROOT
            / "configs/hardware/gh200x4_96gb_gbs128_rgbd_single_task.yaml"
        ).read_text(encoding="utf-8")
        experiment = (
            REPO_ROOT
            / "configs/experiment/robocasa365_close_fridge_rgbd_formal.yaml"
        ).read_text(encoding="utf-8")
        job = (
            REPO_ROOT
            / "deployment/clariden/train_close_fridge_rgbd_xwam.sbatch"
        ).read_text(encoding="utf-8")

        for expected in (
            "task_name: CloseFridge",
            "augment: true",
            "depth_cache_root:",
            "depth_encoding_path:",
            "depth_cache_manifest:",
        ):
            self.assertIn(expected, data)
        for expected in (
            "devices: 4",
            "batch_size_per_gpu: 8",
            "accumulate_grad_batches: 4",
            "global_batch_size: 128",
            "deepspeed_stage: 1",
            "deepspeed_offload_optimizer: false",
            "deepspeed_fp32_optimizer_states: true",
            "use_gradient_checkpointing: true",
        ):
            self.assertIn(expected, hardware)
        for expected in (
            "num_training_steps: 3000",
            "trainer_max_steps: 3000",
            "clean_action_ratio: 0.0",
            "save_interval: 500",
            "save_top_k: 2",
            "durable_save_interval: 1000",
            "save_final_checkpoint: true",
        ):
            self.assertIn(expected, experiment)
        for expected in (
            "#SBATCH --gpus-per-node=4",
            "#SBATCH --time=12:00:00",
            "close_fridge_rgbd_resume_3108551.json",
            "--expected-world-size 4",
            'HOT_CHECKPOINT_ROOT="$DEPLOY_IOPS/xwam_run/$EXP_NAME/checkpoints"',
            'DURABLE_CHECKPOINT_ROOT="$DEPLOY_STORE/checkpoints/xwam/$EXP_NAME/checkpoints"',
            '"dataset.depth_cache_root=$DEPTH_CACHE_ROOT"',
            '"durable_checkpoint_dir=$DURABLE_CHECKPOINT_ROOT"',
            '"depth_loss_positive"',
            "[PASS] CloseFridge RGB-D formal training reached step",
        ):
            self.assertIn(expected, job)
        self.assertNotIn("rm -", job)

        runbook = (REPO_ROOT / "docs/CLUSTER_RUNBOOK.md").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "sbatch deployment/clariden/train_close_fridge_rgbd_xwam.sbatch",
            runbook,
        )
        self.assertIn("4×batch8×accum4=GBS128", runbook)
        self.assertIn("4×batch4×accum8=GBS128", runbook)

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
