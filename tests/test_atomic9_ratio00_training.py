from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from project_tools.h100_training import (
    build_m6_preflight_report,
    load_epoch_schedule_from_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS = [
    "OpenStandMixerHead",
    "PickPlaceSinkToCounter",
    "TurnOnElectricKettle",
    "CloseFridge",
    "TurnOnMicrowave",
    "OpenDrawer",
    "CoffeeSetupMug",
    "PickPlaceDrawerToCounter",
    "CloseBlenderLid",
]


def _digest(payload: dict) -> str:
    canonical = dict(payload)
    canonical.pop("manifest_digest", None)
    encoded = json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _stats_block(dimension: int) -> dict[str, list[float]]:
    return {
        "q01": [0.0] * dimension,
        "q99": [1.0] * dimension,
        "min": [0.0] * dimension,
        "max": [1.0] * dimension,
    }


class Atomic9Ratio00TrainingTest(unittest.TestCase):
    def test_task_manifest_preserves_the_frozen_order(self) -> None:
        payload = json.loads(
            (
                REPO_ROOT
                / "configs/tasks/robocasa365_atomic9_fastwam_overlap.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(payload["tasks"], TASKS)
        self.assertEqual(set(payload["horizons"]), set(TASKS))

    def test_fixed_step_schedule_records_sample_draws_and_effective_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {
                "schema_version": 1,
                "scope": "atomic_only",
                "split": "pretrain",
                "sampling": "natural_proportional",
                "total_valid_clips": 217_600,
                "tasks": [
                    {"task_name": task, "valid_clips": 1} for task in TASKS
                ],
                "ok": True,
                "result": "pass",
            }
            manifest["manifest_digest"] = _digest(manifest)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            schedule = load_epoch_schedule_from_manifest(
                manifest_path,
                global_batch_size=128,
                num_train_epochs=5,
                expected_task_count=9,
                fixed_training_steps=8500,
            )
            self.assertEqual(schedule["schedule_mode"], "fixed_steps")
            self.assertEqual(schedule["num_training_steps"], 8500)
            self.assertEqual(schedule["trainer_max_steps"], 8500)
            self.assertEqual(schedule["planned_sample_draws"], 1_088_000)
            self.assertEqual(schedule["steps_per_epoch"], 1700)
            self.assertEqual(schedule["effective_num_train_epochs"], 5.0)

            stats = {
                "scope": "atomic_only",
                "split": "pretrain",
                "task_count": 9,
                "manifest_digest": manifest["manifest_digest"],
                "total_frames": 100,
                "observation.state": _stats_block(16),
                "action": _stats_block(12),
                "ok": True,
                "result": "pass",
            }
            stats_path = root / "stats.json"
            stats_path.write_text(json.dumps(stats), encoding="utf-8")
            report = build_m6_preflight_report(
                manifest_path,
                stats_path,
                global_batch_size=128,
                num_train_epochs=5,
                expected_task_count=9,
                fixed_training_steps=8500,
            )
            self.assertTrue(report["ok"])
            self.assertEqual(report["schedule"]["fixed_training_steps"], 8500)

            with self.assertRaisesRegex(ValueError, "必须包含18个任务"):
                load_epoch_schedule_from_manifest(
                    manifest_path,
                    global_batch_size=128,
                    num_train_epochs=5,
                )

    def test_training_entry_is_isolated_from_atomic18(self) -> None:
        wrapper = (
            REPO_ROOT
            / "deployment/clariden/train_atomic9_ratio00_xwam_8gpu.sbatch"
        ).read_text(encoding="utf-8")
        experiment = (
            REPO_ROOT
            / "configs/experiment/robocasa365_atomic9_ratio00_gh200_rgb.yaml"
        ).read_text(encoding="utf-8")
        data = (
            REPO_ROOT
            / "configs/data/robocasa365_atomic9_fastwam_overlap.yaml"
        ).read_text(encoding="utf-8")
        for expected in (
            "XWAM_M6_TOTAL_STEPS=8500",
            "gh200x8_96gb_gbs128.yaml",
            "robocasa365_atomic9_ratio00_gh200_rgb.yaml",
            "robocasa365_atomic9_fastwam_overlap.yaml",
        ):
            self.assertIn(expected, wrapper)
        self.assertNotIn("atomic_seen18", wrapper)
        self.assertIn("clean_action_ratio: 0.0", experiment)
        self.assertIn("formal_fixed_training_steps: 8500", experiment)
        self.assertIn("expected_task_count: 9", data)
        self.assertIn("expected_sampling: natural_proportional", data)

    def test_parent_formal_entry_has_backward_compatible_overrides(self) -> None:
        parent = (
            REPO_ROOT / "deployment/clariden/train_m6_formal_xwam.sbatch"
        ).read_text(encoding="utf-8")
        for expected in (
            "XWAM_M6_MANIFEST:-",
            "XWAM_M6_STATS:-",
            "XWAM_M6_PREFLIGHT:-",
            "XWAM_M6_TOTAL_STEPS:-$DEFAULT_TOTAL_STEPS",
            "XWAM_M6_DATA_CONFIG:-",
            "XWAM_M6_EXPERIMENT_CONFIG:-",
        ):
            self.assertIn(expected, parent)

    def test_ratio05_control_changes_only_ratio_identity_and_stop_step(self) -> None:
        ratio00_path = (
            REPO_ROOT
            / "configs/experiment/robocasa365_atomic9_ratio00_gh200_rgb.yaml"
        )
        ratio05_path = (
            REPO_ROOT
            / "configs/experiment/robocasa365_atomic9_ratio05_gh200_rgb.yaml"
        )
        ratio00 = ratio00_path.read_text(encoding="utf-8")
        ratio05 = ratio05_path.read_text(encoding="utf-8")
        self.assertIn("clean_action_ratio: 0.0", ratio00)
        self.assertIn("clean_action_ratio: 0.5", ratio05)
        self.assertIn("formal_fixed_training_steps: 8500", ratio05)

        ignored_keys = {
            "exp_name",
            "clean_action_ratio",
            "wandb_name",
        }

        def normalized(text: str) -> list[str]:
            return [
                line
                for line in text.rstrip().splitlines()
                if not line.startswith("# Atomic9正式对照")
                and line.partition(":")[0] not in ignored_keys
            ]

        self.assertEqual(normalized(ratio00), normalized(ratio05))

        wrapper = (
            REPO_ROOT
            / "deployment/clariden/train_atomic9_ratio05_xwam_8gpu.sbatch"
        ).read_text(encoding="utf-8")
        for expected in (
            "XWAM_M6_TOTAL_STEPS=7500",
            "gh200x8_96gb_gbs128.yaml",
            "robocasa365_atomic9_ratio05_gh200_rgb.yaml",
            "robocasa365_atomic9_fastwam_overlap_ratio05_rgb_seed42_8gpu",
            'RATIO00_EVIDENCE="$DEPLOY_STORE/manifests/xwam/atomic9_ratio00"',
            'RATIO05_EVIDENCE="$DEPLOY_STORE/manifests/xwam/atomic9_ratio05"',
        ):
            self.assertIn(expected, wrapper)

        preflight = (
            REPO_ROOT
            / "deployment/clariden/prepare_atomic9_ratio05_preflight_xwam.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("--fixed-training-steps 8500", preflight)
        self.assertIn("--expected-task-count 9", preflight)
        self.assertNotIn("compute_robocasa365_global_stats.py", preflight)


if __name__ == "__main__":
    unittest.main()
