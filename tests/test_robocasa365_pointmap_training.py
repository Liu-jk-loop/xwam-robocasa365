from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.resolve_robocasa365_eval_checkpoints import (
    resolve_exact_checkpoints,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_checkpoint(root: Path, step: int, *, ranks: int = 8) -> Path:
    checkpoint = root / f"epoch=0-step={step}.ckpt" / "checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "mp_rank_00_model_states.pt").write_bytes(b"model")
    for rank in range(ranks):
        (
            checkpoint
            / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
        ).write_bytes(b"optimizer")
    return checkpoint.parent


class RoboCasa365PointMapTrainingContractTests(unittest.TestCase):
    def test_pointmap_aux_model_and_loader_are_explicit(self) -> None:
        model = (
            REPO_ROOT
            / "configs/model/wan22_5b_robocasa365_atomic_pointmap.yaml"
        ).read_text(encoding="utf-8")
        data = (
            REPO_ROOT / "configs/data/robocasa365_close_fridge_pointmap.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("use_depth: false", model)
        self.assertIn("use_pointmap: true", model)
        self.assertIn("depth_loss_weight: 0.0", model)
        self.assertIn("pointmap_loss_weight: 1.0", model)
        self.assertIn("clean_action_ratio: 0.0", model)
        self.assertIn("pointmap_cache_root:", data)
        self.assertIn("pointmap_cache_manifest:", data)
        self.assertIn("pointmap_cache_audit:", data)

    def test_loader_uses_mmap_and_returns_float_pointmaps(self) -> None:
        loader = (REPO_ROOT / "data/robocasa365_dataset.py").read_text(
            encoding="utf-8"
        )
        augmentation = (REPO_ROOT / "data/augmentation.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('np.load(path, mmap_mode="r", allow_pickle=False)', loader)
        self.assertIn('data["pointmaps"] = pointmaps', loader)
        self.assertIn('auxiliary_mode="bilinear" if pointmaps is not None', augmentation)
        self.assertIn("if depths is not None and pointmaps is not None", augmentation)

    def test_four_gpu_gate_requires_batch_loss_update_and_resume(self) -> None:
        job = (
            REPO_ROOT
            / "deployment/clariden/smoke_close_fridge_pointmap_train_resume_xwam.sbatch"
        ).read_text(encoding="utf-8")
        batch_position = job.index("scripts/audit_robocasa365_batch.py")
        initial_position = job.index("phase=initial_step_0_to_2")
        resume_position = job.index("phase=resume_step_2_to_4")
        self.assertLess(batch_position, initial_position)
        self.assertLess(initial_position, resume_position)
        self.assertIn("--pointmap-cache-audit", job)
        self.assertIn("--expect-pointmap", job)
        self.assertIn("--allow-dirty-git", job)
        self.assertIn("trainer_max_steps=4", job)
        self.assertNotIn("rm -", job)

    def test_formal_training_is_two_node_gbs128_ratio0_for_1500_steps(self) -> None:
        hardware = (
            REPO_ROOT
            / "configs/hardware/gh200x8_96gb_gbs128_pointmap_single_task.yaml"
        ).read_text(encoding="utf-8")
        experiment = (
            REPO_ROOT
            / "configs/experiment/robocasa365_close_fridge_pointmap_formal.yaml"
        ).read_text(encoding="utf-8")
        job = (
            REPO_ROOT
            / "deployment/clariden/train_close_fridge_pointmap_xwam.sbatch"
        ).read_text(encoding="utf-8")
        for expected in (
            "batch_size_per_gpu: 4",
            "accumulate_grad_batches: 4",
            "global_batch_size: 128",
            "formal_world_size: 8",
            "formal_num_nodes: 2",
        ):
            self.assertIn(expected, hardware)
        for expected in (
            "num_training_steps: 1500",
            "trainer_max_steps: 1500",
            "clean_action_ratio: 0.0",
            "save_interval: 250",
            "save_top_k: 5",
            "durable_save_interval: 500",
        ):
            self.assertIn(expected, experiment)
        for expected in (
            "#SBATCH --nodes=2",
            "#SBATCH --gpus-per-node=4",
            "TOTAL_STEPS=1500",
            "--expected-world-size 8",
            "python -m torch.distributed.run",
            '"dataset.pointmap_cache_root=$POINTMAP_CACHE_ROOT"',
            '"pointmap_loss_positive"',
            '"git_commit_recorded"',
            'CURRENT_COMMIT="$(git rev-parse HEAD)"',
            '"expect_pointmap": true',
        ):
            self.assertIn(expected, job)
        self.assertNotIn("rm -", job)

    def test_policy_inference_remains_rgb_only(self) -> None:
        policy = (
            REPO_ROOT / "evaluation/robocasa365_policy_server.py"
        ).read_text(encoding="utf-8")
        self.assertIn("training_use_pointmap", policy)
        self.assertIn("runner = XWAMRunner(config=config, run_depth=False)", policy)
        self.assertIn('"online_depth_required": False', policy)

    def test_pointmap_evaluation_resolves_three_complete_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = {
                f"step{step}": _make_checkpoint(root, step)
                for step in (500, 1000, 1500)
            }
            _make_checkpoint(root, 1250, ranks=7)
            report = resolve_exact_checkpoints(
                [root],
                {"step500": 500, "step1000": 1000, "step1500": 1500},
                expected_world_size=8,
            )
            self.assertTrue(report["ok"], report["errors"])
            self.assertEqual(
                {
                    group: record["path"]
                    for group, record in report["groups"].items()
                },
                {
                    group: str(checkpoint.resolve())
                    for group, checkpoint in expected.items()
                },
            )
            incomplete = resolve_exact_checkpoints(
                [root],
                {"step1250": 1250},
                expected_world_size=8,
            )
            self.assertFalse(incomplete["ok"])
            self.assertIn("完整step 1250", incomplete["errors"][0])

    def test_pointmap_evaluation_runs_three_steps_with_matched_seeds(self) -> None:
        job = (
            REPO_ROOT
            / "deployment/clariden/eval_close_fridge_pointmap_xwam.sbatch"
        ).read_text(encoding="utf-8")
        for expected in (
            "#SBATCH --gpus-per-node=4",
            "close_fridge_pointmap_aux_ratio00_seed42_8gpu",
            "--group-step step500=500",
            "--group-step step1000=1000",
            "--group-step step1500=1500",
            "--expected-world-size 8",
            "RUN_NAMES=(",
            "step500_seed42 step500_seed7",
            "step1000_seed42 step1000_seed7",
            "step1500_seed42 step1500_seed7",
            "RUN_SEEDS=(42 7 42 7 42 7)",
            "RUN_GPUS=(0 1 2 3 0 1)",
            'EVAL_ROOT="$DEPLOY_IOPS/x-wam-eval/close-fridge-pointmap/$EVAL_ID"',
            '"training_use_pointmap": true',
            '"training_auxiliary_geometry": "pointmap"',
            '"training_modality": "pointmap_aux"',
            '"online_inference_modalities": ["rgb", "proprio"]',
            '"online_pointmap_required": False',
        ):
            self.assertIn(expected, job)
        self.assertEqual(job.count("--server-id 5"), 1)
        self.assertEqual(job.count("--client-id 5"), 1)
        self.assertIn("TOPOLOGY_5=", job)
        self.assertIn("CHECKPOINT_5=", job)
        self.assertIn("GPU_5=", job)
        self.assertNotIn("POINTMAP_CACHE_ROOT", job)
        self.assertNotIn("trainer_max_steps", job)


if __name__ == "__main__":
    unittest.main()
