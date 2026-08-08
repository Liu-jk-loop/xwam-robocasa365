from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from project_tools.h100_training import (
    build_m6_preflight_report,
    resolve_epoch_schedule,
)


def _digest(payload: dict) -> str:
    canonical = dict(payload)
    canonical.pop("manifest_digest", None)
    encoded = json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class M6H100TrainingTest(unittest.TestCase):
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

    def test_training_entry_uses_epoch_aligned_sampler_and_h100_guards(self) -> None:
        root = Path(__file__).resolve().parents[1]
        entrypoint = (root / "scripts/train_sft.py").read_text()
        for token in (
            "EpochAlignedDistributedSampler",
            "use_distributed_sampler=train_sampler is None",
            "validate_h100_training_contract",
            "OptimizerStateDtypeAudit",
            "final_checkpoint_save_complete",
        ):
            self.assertIn(token, entrypoint)


if __name__ == "__main__":
    unittest.main()
