from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from project_tools.h100_training import resolve_m6_formal_chunk
from project_tools.training_run import (
    cosine_learning_rate_at_step,
    resolve_learning_rate_schedule,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_checkpoint(root: Path, step: int, *, ranks: int = 8) -> Path:
    state = root / f"epoch=0-step={step}.ckpt" / "checkpoint"
    state.mkdir(parents=True)
    (state / "mp_rank_00_model_states.pt").write_bytes(b"model")
    for rank in range(ranks):
        (
            state / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
        ).write_bytes(b"optimizer")
    return state.parent.resolve()


class Atomic9RgbContinuationTest(unittest.TestCase):
    def test_continuation_lr_matches_the_source_cosine_at_step7500(self) -> None:
        expected = cosine_learning_rate_at_step(
            base_lr=1e-5,
            num_warmup_steps=200,
            num_training_steps=8500,
            step=7500,
        )
        self.assertAlmostEqual(expected, 3.53909638412e-7, places=18)
        contract = resolve_learning_rate_schedule(
            {
                "lr_schedule_mode": "constant_resume",
                "lr": 1e-5,
                "num_warmup_steps": 200,
                "num_training_steps": 12000,
                "lr_continuation_start_step": 7500,
                "lr_continuation_expected_resume_step": 8000,
                "lr_continuation_end_step": 12000,
                "lr_continuation_learning_rate": expected,
                "lr_continuation_relative_tolerance": 0.05,
                "lr_continuation_source_base_lr": 1e-5,
                "lr_continuation_source_warmup_steps": 200,
                "lr_continuation_source_training_steps": 8500,
            }
        )
        self.assertEqual(contract["mode"], "constant_resume")
        self.assertEqual(contract["resume_start_step"], 7500)
        self.assertEqual(contract["expected_resume_step"], 8000)
        self.assertEqual(contract["continuation_end_step"], 12000)
        self.assertAlmostEqual(contract["continuation_scale"], 0.0353909638412)

    def test_continuation_lr_rejects_a_scheduler_jump(self) -> None:
        with self.assertRaisesRegex(ValueError, "不连续"):
            resolve_learning_rate_schedule(
                {
                    "lr_schedule_mode": "constant_resume",
                    "lr": 1e-5,
                    "num_warmup_steps": 200,
                    "num_training_steps": 12000,
                    "lr_continuation_start_step": 7500,
                    "lr_continuation_end_step": 12000,
                    "lr_continuation_learning_rate": 3.18e-6,
                    "lr_continuation_source_base_lr": 1e-5,
                    "lr_continuation_source_warmup_steps": 200,
                    "lr_continuation_source_training_steps": 8500,
                }
            )

    def test_planner_requires_an_exact_complete_step7500_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            continuation_hot = root / "continuation-hot"
            continuation_durable = root / "continuation-durable"
            source_hot = root / "source-hot"
            source_durable = root / "source-durable"
            checkpoint7500 = _write_checkpoint(source_hot, 7500)

            plan = resolve_m6_formal_chunk(
                [continuation_hot, continuation_durable],
                total_steps=12000,
                chunk_steps=12000,
                expected_world_size=8,
                bootstrap_checkpoint_root=[source_hot, source_durable],
                bootstrap_step=7500,
            )
            self.assertEqual(plan["completed_step"], 7500)
            self.assertEqual(plan["target_step"], 12000)
            self.assertEqual(plan["resume_checkpoint"], str(checkpoint7500))
            self.assertEqual(plan["bootstrap_checkpoint"], str(checkpoint7500))

            checkpoint8000 = _write_checkpoint(continuation_hot, 8000)
            resumed = resolve_m6_formal_chunk(
                [continuation_hot, continuation_durable],
                total_steps=12000,
                chunk_steps=12000,
                expected_world_size=8,
                bootstrap_checkpoint_root=[source_hot, source_durable],
                bootstrap_step=7500,
            )
            self.assertEqual(resumed["completed_step"], 8000)
            self.assertEqual(resumed["resume_checkpoint"], str(checkpoint8000))

            for item in source_hot.rglob("*optim_states.pt"):
                item.unlink()
            for item in continuation_hot.rglob("*optim_states.pt"):
                item.unlink()
            with self.assertRaisesRegex(ValueError, "找不到完整step 7500"):
                resolve_m6_formal_chunk(
                    [root / "empty-hot", root / "empty-durable"],
                    total_steps=12000,
                    chunk_steps=12000,
                    expected_world_size=8,
                    bootstrap_checkpoint_root=[source_hot, source_durable],
                    bootstrap_step=7500,
                )

    def test_clariden_entries_freeze_the_continuation_contract(self) -> None:
        experiment = (
            REPO_ROOT
            / "configs/experiment/robocasa365_atomic9_ratio00_gh200_rgb_cont7500_12000.yaml"
        ).read_text(encoding="utf-8")
        preflight = (
            REPO_ROOT
            / "deployment/clariden/prepare_atomic9_ratio00_rgb_cont12000_preflight_xwam.sbatch"
        ).read_text(encoding="utf-8")
        training = (
            REPO_ROOT
            / "deployment/clariden/train_atomic9_ratio00_rgb_cont12000_xwam_8gpu.sbatch"
        ).read_text(encoding="utf-8")
        parent = (
            REPO_ROOT / "deployment/clariden/train_m6_formal_xwam.sbatch"
        ).read_text(encoding="utf-8")
        runner = (REPO_ROOT / "runners/xwam_runner.py").read_text(encoding="utf-8")

        for expected in (
            "num_training_steps: 12000",
            "formal_fixed_training_steps: 12000",
            "lr_schedule_mode: constant_resume",
            "lr_continuation_start_step: 7500",
            "lr_continuation_expected_resume_step: 7500",
            "lr_continuation_learning_rate: 3.53909638412e-7",
            "clean_action_ratio: 0.0",
        ):
            self.assertIn(expected, experiment)
        for expected in (
            "--bootstrap-step 7500",
            "--expected-world-size 8",
            '"completed_step": 7500',
            '"target_step": 12000',
        ):
            self.assertIn(expected, preflight)
        for expected in (
            "#SBATCH --nodes=2",
            "#SBATCH --gpus-per-node=4",
            "XWAM_M6_TOTAL_STEPS=12000",
            "XWAM_M6_BOOTSTRAP_STEP=7500",
            "gh200x8_96gb_gbs128.yaml",
            "rgb_cont7500_12000_seed42_8gpu",
        ):
            self.assertIn(expected, training)
        self.assertIn('BOOTSTRAP_STEP="${XWAM_M6_BOOTSTRAP_STEP:-}"', parent)
        self.assertIn('BOOTSTRAP_STEP="$BOOTSTRAP_STEP"', parent)
        self.assertIn('RESUME_STEP="$XWAM_PLAN_COMPLETED_STEP"', parent)
        self.assertIn(
            '"lr_continuation_expected_resume_step=$RESUME_STEP"', parent
        )
        self.assertIn("LR continuation guard: pass", runner)
        self.assertNotIn("rgbd", training.lower())


if __name__ == "__main__":
    unittest.main()
