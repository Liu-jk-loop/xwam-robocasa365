from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from project_tools.h100_training import (
    _metrics_audit,
    validate_m6_formal_training_contract,
)
from project_tools.robocasa365_depth_encoding import file_sha256
from project_tools.robocasa365_pointmap_cache import (
    POINTMAP_RENDER_POLICY,
    pointmap_cache_path,
    pointmap_sidecar_path,
    validate_task_pointmap_cache,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _top_level_scalars(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if not line or line[0].isspace() or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition(":")
        if separator:
            values[key.strip()] = value.strip()
    return values


class Atomic9PointMapTrainingTests(unittest.TestCase):
    def test_formal_contract_matches_rgbd_topology_and_schedule(self) -> None:
        contract = validate_m6_formal_training_contract(
            {
                "formal_modality": "pointmap_aux",
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
                "use_depth": False,
                "use_pointmap": True,
                "depth_loss_weight": 0.0,
                "pointmap_loss_weight": 1.0,
                "formal_fixed_training_steps": 14000,
                "num_train_epochs": 8,
                "num_training_steps": 14000,
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
        self.assertEqual(contract["formal_modality"], "pointmap_aux")
        self.assertEqual(contract["global_batch_size"], 128)
        self.assertEqual(contract["fixed_training_steps"], 14000)
        self.assertTrue(all(contract["checks"].values()))

    def test_formal_metrics_require_positive_pointmap_and_zero_depth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "train.log"
            log.write_text(
                "[METRICS] Step: 0 - train/video_loss: 1.0, "
                "train/action_loss: 0.5, train/proprio_loss: 0.25, "
                "train/action_proprio_supervision_ratio: 1.0, "
                "train/depth_loss: 0.0, train/pointmap_loss: 0.125, "
                "train/loss: 1.875\n",
                encoding="utf-8",
            )
            report = _metrics_audit(log, expect_pointmap=True)
            self.assertTrue(all(report["checks"].values()), report)
            log.write_text(
                log.read_text(encoding="utf-8").replace(
                    "train/pointmap_loss: 0.125", "train/pointmap_loss: 0.0"
                ),
                encoding="utf-8",
            )
            report = _metrics_audit(log, expect_pointmap=True)
            self.assertFalse(report["checks"]["pointmap_loss_positive"])

    def test_atomic9_entries_match_rgbd_training_contract(self) -> None:
        cache_job = (
            REPO_ROOT / "deployment/clariden/build_atomic9_pointmap_cache_xwam.sbatch"
        ).read_text(encoding="utf-8")
        preflight = (
            REPO_ROOT
            / "deployment/clariden/prepare_atomic9_pointmap_14000step_preflight_xwam.sbatch"
        ).read_text(encoding="utf-8")
        train_job = (
            REPO_ROOT
            / "deployment/clariden/train_atomic9_ratio00_pointmap_xwam_8gpu.sbatch"
        ).read_text(encoding="utf-8")
        data = (
            REPO_ROOT
            / "configs/data/robocasa365_atomic9_fastwam_overlap_pointmap.yaml"
        ).read_text(encoding="utf-8")
        hardware = (
            REPO_ROOT / "configs/hardware/gh200x8_96gb_gbs128_pointmap.yaml"
        ).read_text(encoding="utf-8")
        experiment = (
            REPO_ROOT
            / "configs/experiment/robocasa365_atomic9_ratio00_gh200_pointmap.yaml"
        ).read_text(encoding="utf-8")

        for expected in (
            "#SBATCH --gpus-per-node=4",
            "--ntasks=4",
            "--gpus-per-task=1",
            "--num-shards 4",
            "--generate-only",
            "--index-only",
            "robocasa365_atomic9_fastwam_overlap.json",
            "robocasa365_atomic9_pointmap_cache_manifest.json",
        ):
            self.assertIn(expected, cache_job)
        for expected in (
            "--expected-task-count 9",
            "--fixed-training-steps 14000",
            "--expected-total-steps 14000",
            "--milestone-epoch 5",
        ):
            self.assertIn(expected, preflight)
        for expected in (
            "#SBATCH --nodes=2",
            "XWAM_M6_EXPECT_DEPTH=false",
            "XWAM_M6_EXPECT_POINTMAP=true",
            "XWAM_M6_POINTMAP_CACHE_ROOT=",
            "XWAM_M6_POINTMAP_MANIFEST=",
            "XWAM_M6_POINTMAP_AUDIT=",
            'XWAM_M6_TOTAL_STEPS="$XWAM_SCHEDULE_TOTAL_STEPS"',
            'XWAM_M6_MILESTONE_STEP="$XWAM_SCHEDULE_MILESTONE_STEP"',
        ):
            self.assertIn(expected, train_job)
        self.assertIn("expected_task_count: 9", data)
        self.assertIn("expected_sampling: natural_proportional", data)
        self.assertIn("batch_size_per_gpu: 4", hardware)
        self.assertIn("accumulate_grad_batches: 4", hardware)
        self.assertIn("global_batch_size: 128", hardware)
        self.assertIn("formal_modality: pointmap_aux", hardware)
        self.assertIn("lr: 1e-5", experiment)
        self.assertIn("num_warmup_steps: 200", experiment)
        self.assertIn("num_training_steps: 14000", experiment)
        self.assertIn("formal_fixed_training_steps: 14000", experiment)
        self.assertIn("clean_action_ratio: 0.0", experiment)
        self.assertIn("save_interval: 500", experiment)
        self.assertIn("save_top_k: 5", experiment)
        self.assertIn("durable_save_interval: 3000", experiment)

        rgbd_hardware = _top_level_scalars(
            (
                REPO_ROOT / "configs/hardware/gh200x8_96gb_gbs128_rgbd.yaml"
            ).read_text(encoding="utf-8")
        )
        pointmap_hardware = _top_level_scalars(hardware)
        for key in (
            "precision",
            "devices",
            "batch_size_per_gpu",
            "accumulate_grad_batches",
            "global_batch_size",
            "num_workers_per_gpu",
            "prefetch_factor",
            "use_gradient_checkpointing",
            "deepspeed_stage",
            "deepspeed_offload_optimizer",
            "deepspeed_fp32_optimizer_states",
            "formal_world_size",
            "formal_num_nodes",
            "formal_devices_per_node",
        ):
            self.assertEqual(pointmap_hardware[key], rgbd_hardware[key], key)

        rgbd_experiment = _top_level_scalars(
            (
                REPO_ROOT
                / "configs/experiment/robocasa365_atomic9_ratio00_gh200_rgbd.yaml"
            ).read_text(encoding="utf-8")
        )
        pointmap_experiment = _top_level_scalars(experiment)
        for key in (
            "seed",
            "lr",
            "weight_decay",
            "num_warmup_steps",
            "num_train_epochs",
            "num_training_steps",
            "formal_fixed_training_steps",
            "gradient_clip_val",
            "gradient_clip_algorithm",
            "clean_action_ratio",
            "text_dropout_prob",
            "train_shuffle",
            "persist_generator_state",
            "save_interval",
            "save_top_k",
            "durable_save_interval",
            "durable_save_top_k",
            "milestone_checkpoint_filename",
            "log_interval",
            "val_interval",
        ):
            self.assertEqual(pointmap_experiment[key], rgbd_experiment[key], key)

    def test_shared_formal_entry_passes_pointmap_paths_to_dataset(self) -> None:
        shared = (
            REPO_ROOT / "deployment/clariden/train_m6_formal_xwam.sbatch"
        ).read_text(encoding="utf-8")
        for expected in (
            "XWAM_M6_EXPECT_POINTMAP:-false",
            "XWAM_M6_POINTMAP_CACHE_ROOT:-",
            "XWAM_M6_POINTMAP_MANIFEST:-",
            "XWAM_M6_POINTMAP_AUDIT:-",
            'EXPECT_POINTMAP="$EXPECT_POINTMAP"',
            '"dataset.pointmap_cache_root=$POINTMAP_CACHE_ROOT"',
            '"dataset.pointmap_cache_manifest=$POINTMAP_MANIFEST"',
            '"dataset.pointmap_cache_audit=$POINTMAP_AUDIT"',
        ):
            self.assertIn(expected, shared)

    def test_multitask_index_selects_and_validates_one_task(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_root = root / "cache"
            task_name = "CloseFridge"
            camera_key = "observation.images.robot0_agentview_left"
            array_path = pointmap_cache_path(
                cache_root,
                task_name=task_name,
                episode_index=0,
                camera_key=camera_key,
                chunks_size=1000,
            )
            array_path.parent.mkdir(parents=True)
            np.save(array_path, np.zeros((1, 3, 256, 320), dtype=np.float16))
            contract_sha256 = "1" * 64
            camera = {
                "camera_key": camera_key,
                "array_path": str(array_path.resolve()),
                "shape": [1, 3, 256, 320],
            }
            pointmap_sidecar_path(array_path).write_text(
                json.dumps(camera), encoding="utf-8"
            )
            child_manifest_path = root / "child-manifest.json"
            child_audit_path = root / "child-audit.json"
            child_manifest = {
                "ok": True,
                "result": "pass",
                "scope": "atomic_only",
                "task_name": task_name,
                "cache_root": str(cache_root.resolve()),
                "contract_sha256": contract_sha256,
                "render_policy": POINTMAP_RENDER_POLICY,
                "contract": {"storage": {"shape_per_frame": [3, 256, 320]}},
                "episodes": [
                    {
                        "episode_index": 0,
                        "episode_length": 1,
                        "cameras": [camera],
                    }
                ],
            }
            child_manifest_path.write_text(
                json.dumps(child_manifest), encoding="utf-8"
            )
            child_audit = {
                "ok": True,
                "result": "pass",
                "scope": "atomic_only",
                "task_name": task_name,
                "cache_root": str(cache_root.resolve()),
                "manifest_path": str(child_manifest_path.resolve()),
                "contract_sha256": contract_sha256,
                "render_policy": POINTMAP_RENDER_POLICY,
                "checks": {"all_artifacts": True},
            }
            child_audit_path.write_text(json.dumps(child_audit), encoding="utf-8")
            entry = {
                "task_name": task_name,
                "manifest_path": str(child_manifest_path.resolve()),
                "manifest_sha256": file_sha256(child_manifest_path),
                "audit_path": str(child_audit_path.resolve()),
                "audit_sha256": file_sha256(child_audit_path),
            }
            index_manifest_path = root / "index-manifest.json"
            index_manifest = {
                "ok": True,
                "result": "pass",
                "scope": "atomic_only",
                "task_count": 1,
                "task_names": [task_name],
                "cache_root": str(cache_root.resolve()),
                "contract_sha256": contract_sha256,
                "render_policy": POINTMAP_RENDER_POLICY,
                "tasks": [entry],
            }
            index_manifest_path.write_text(
                json.dumps(index_manifest), encoding="utf-8"
            )
            index_audit_path = root / "index-audit.json"
            index_audit = {
                "ok": True,
                "result": "pass",
                "scope": "atomic_only",
                "task_count": 1,
                "task_names": [task_name],
                "cache_root": str(cache_root.resolve()),
                "manifest_path": str(index_manifest_path.resolve()),
                "manifest_sha256": file_sha256(index_manifest_path),
                "contract_sha256": contract_sha256,
                "render_policy": POINTMAP_RENDER_POLICY,
                "tasks": [entry],
                "checks": {"all_tasks": True},
            }
            index_audit_path.write_text(json.dumps(index_audit), encoding="utf-8")

            result = validate_task_pointmap_cache(
                cache_root=cache_root,
                manifest_path=index_manifest_path,
                audit_path=index_audit_path,
                task_name=task_name,
                episode_lengths={0: 1},
                camera_keys=(camera_key,),
                chunks_size=1000,
            )
            self.assertEqual(result["task_name"], task_name)
            self.assertEqual(result["multitask_index"]["task_count"], 1)


if __name__ == "__main__":
    unittest.main()
