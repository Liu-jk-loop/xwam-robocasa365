from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from project_tools.h100_training import (
    _metrics_audit,
    validate_m6_formal_training_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class Atomic9RgbdTrainingTest(unittest.TestCase):
    def test_rgbd_formal_contract_is_gbs128_on_eight_gh200s(self) -> None:
        contract = validate_m6_formal_training_contract(
            {
                "formal_modality": "rgbd",
                "formal_accelerator": "GH200",
                "formal_minimum_memory_gib": 90,
                "formal_world_size": 8,
                "formal_num_nodes": 2,
                "formal_devices_per_node": 4,
                "batch_size_per_gpu": 4,
                "accumulate_grad_batches": 4,
                "global_batch_size": 128,
                "precision": "bf16-mixed",
                "formal_zero_stage": 1,
                "deepspeed_stage": 1,
                "deepspeed_offload_optimizer": False,
                "deepspeed_fp32_optimizer_states": True,
                "deepspeed_overlap_comm": True,
                "deepspeed_exclude_frozen_parameters": False,
                "use_depth": True,
                "depth_loss_weight": 1.0,
                "formal_fixed_training_steps": None,
                "formal_num_train_epochs": 8,
                "num_train_epochs": 8,
                "num_training_steps": None,
                "dataset": {"expected_sampling": "natural_proportional"},
                "train_subset_size": None,
                "train_shuffle": True,
                "num_workers_per_gpu": 4,
                "use_gradient_checkpointing": True,
                "cache_frozen_text_embeddings": True,
                "max_cached_text_embeddings": 128,
                "enable_segment_timing": True,
                "segment_timing_interval_steps": 20,
            },
            world_size=8,
        )
        self.assertEqual(contract["formal_modality"], "rgbd")
        self.assertEqual(contract["global_batch_size"], 128)
        self.assertEqual(contract["num_train_epochs"], 8)
        self.assertTrue(all(contract["checks"].values()))

    def test_rgbd_metrics_require_a_finite_positive_depth_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "train.log"
            log.write_text(
                "[METRICS] Step: 0 - train/video_loss: 1.0, "
                "train/action_loss: 0.5, train/proprio_loss: 0.25, "
                "train/action_proprio_supervision_ratio: 1.0, "
                "train/depth_loss: 0.125, train/loss: 1.875\n",
                encoding="utf-8",
            )
            report = _metrics_audit(log, expect_depth=True)
            self.assertTrue(all(report["checks"].values()))

            log.write_text(log.read_text().replace("0.125", "0.0"), encoding="utf-8")
            report = _metrics_audit(log, expect_depth=True)
            self.assertFalse(report["checks"]["depth_loss_positive"])

    def test_cache_and_training_entries_freeze_the_atomic9_rgbd_scope(self) -> None:
        cache = (
            REPO_ROOT / "deployment/clariden/build_atomic9_rgbd_cache_xwam.sbatch"
        ).read_text(encoding="utf-8")
        wrapper = (
            REPO_ROOT / "deployment/clariden/train_atomic9_ratio00_rgbd_xwam_8gpu.sbatch"
        ).read_text(encoding="utf-8")
        data = (
            REPO_ROOT / "configs/data/robocasa365_atomic9_fastwam_overlap_rgbd.yaml"
        ).read_text(encoding="utf-8")
        experiment = (
            REPO_ROOT / "configs/experiment/robocasa365_atomic9_ratio00_gh200_rgbd.yaml"
        ).read_text(encoding="utf-8")
        hardware = (
            REPO_ROOT / "configs/hardware/gh200x8_96gb_gbs128_rgbd.yaml"
        ).read_text(encoding="utf-8")

        for expected in (
            "robocasa365_atomic9_fastwam_overlap.json",
            "--expected-task-count 9",
            "--episodes-per-task 0",
            "--encoding-input",
            "robocasa365_atomic9_rgbd_cache_manifest.json",
        ):
            self.assertIn(expected, cache)
        for expected in (
            "#SBATCH --nodes=2",
            'source "$SCHEDULE_ENV"',
            'XWAM_M6_TOTAL_STEPS="$XWAM_SCHEDULE_TOTAL_STEPS"',
            'XWAM_M6_MILESTONE_STEP="$XWAM_SCHEDULE_MILESTONE_STEP"',
            "XWAM_M6_EXPECT_DEPTH=true",
            "XWAM_M6_DEPTH_AUDIT=",
            "wan22_5b_robocasa365_atomic_rgbd.yaml",
            "robocasa365_atomic9_fastwam_overlap_manifest.json",
            "robocasa365_atomic9_fastwam_overlap_global_stats.json",
        ):
            self.assertIn(expected, wrapper)
        self.assertNotIn("close_fridge_rgbd_ratio00_seed42_8gpu/checkpoints", wrapper)
        self.assertIn("expected_task_count: 9", data)
        self.assertIn("expected_sampling: natural_proportional", data)
        self.assertIn("clean_action_ratio: 0.0", experiment)
        self.assertIn("num_train_epochs: 8", experiment)
        self.assertIn("formal_fixed_training_steps: null", experiment)
        self.assertIn("formal_num_train_epochs: 8", experiment)
        self.assertIn("milestone_checkpoint_filename: epoch5-{step}", experiment)
        self.assertIn("batch_size_per_gpu: 4", hardware)
        self.assertIn("accumulate_grad_batches: 4", hardware)
        self.assertIn("formal_modality: rgbd", hardware)

    def test_shared_formal_entry_accepts_depth_without_changing_rgb_defaults(self) -> None:
        shared = (
            REPO_ROOT / "deployment/clariden/train_m6_formal_xwam.sbatch"
        ).read_text(encoding="utf-8")
        for expected in (
            "XWAM_M6_MODEL_CONFIG:-configs/model/wan22_5b_robocasa365_atomic.yaml",
            "XWAM_M6_EXPECT_DEPTH:-false",
            "XWAM_M6_DEPTH_AUDIT:-",
            '"dataset.depth_cache_root=$DEPTH_CACHE_ROOT"',
            '"dataset.depth_encoding_path=$DEPTH_ENCODING"',
            '"dataset.depth_cache_manifest=$DEPTH_MANIFEST"',
            '"milestone_checkpoint_dir=$DURABLE_CHECKPOINT_ROOT/milestones"',
        ):
            self.assertIn(expected, shared)

    def test_preflight_exports_exact_epoch_and_epoch5_steps(self) -> None:
        preflight_job = (
            REPO_ROOT
            / "deployment/clariden/prepare_atomic9_rgbd_8epoch_preflight_xwam.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("--epochs 8", preflight_job)
        self.assertNotIn("--fixed-training-steps", preflight_job)
        self.assertIn("--milestone-epoch 5", preflight_job)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preflight = root / "preflight.json"
            output = root / "schedule.env"
            preflight.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "result": "pass",
                        "schedule": {
                            "schedule_mode": "epochs",
                            "num_train_epochs": 8,
                            "steps_per_epoch": 1733,
                            "num_training_steps": 13864,
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    "python",
                    str(REPO_ROOT / "scripts/export_robocasa365_epoch_schedule.py"),
                    "--preflight",
                    str(preflight),
                    "--expected-epochs",
                    "8",
                    "--milestone-epoch",
                    "5",
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            values = dict(
                line.split("=", 1)
                for line in output.read_text(encoding="utf-8").splitlines()
            )
            self.assertEqual(values["XWAM_SCHEDULE_TOTAL_STEPS"], "13864")
            self.assertEqual(values["XWAM_SCHEDULE_MILESTONE_STEP"], "8665")

    def test_training_entry_has_an_independent_milestone_checkpoint_tier(self) -> None:
        entrypoint = (REPO_ROOT / "scripts/train_sft.py").read_text(encoding="utf-8")
        for expected in (
            'checkpoint_tier="milestone"',
            'filename=str(',
            "every_n_train_steps=milestone_interval",
            "save_top_k=-1",
            'callbacks.insert(1, milestone_checkpoint_callback)',
            '"milestone": (',
        ):
            self.assertIn(expected, entrypoint)


if __name__ == "__main__":
    unittest.main()
