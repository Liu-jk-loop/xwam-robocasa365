from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from data.robocasa365_contract import inspect_dataset, load_task_manifest, resolve_lerobot_root


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "configs" / "tasks" / "robocasa365_atomic_seen.json"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


class RoboCasa365ContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.dataset_root = Path(self.temp_dir.name) / "CloseFridge" / "lerobot"
        meta = self.dataset_root / "meta"
        features = {
            "observation.state": {"dtype": "float64", "shape": [16]},
            "action": {"dtype": "float64", "shape": [12]},
        }
        for camera in (
            "observation.images.robot0_agentview_left",
            "observation.images.robot0_agentview_right",
            "observation.images.robot0_eye_in_hand",
        ):
            features[camera] = {"dtype": "video", "shape": [256, 256, 3]}
        write_json(
            meta / "info.json",
            {
                "codebase_version": "v2.1",
                "fps": 20,
                "total_episodes": 1,
                "total_frames": 8,
                "features": features,
            },
        )
        write_jsonl(meta / "tasks.jsonl", [{"task_index": 0, "task": "close the fridge"}])
        write_jsonl(meta / "episodes.jsonl", [{"episode_index": 0, "tasks": ["close the fridge"], "length": 8}])
        write_json(meta / "modality.json", {"state": {}, "action": {}, "video": {}})
        write_json(meta / "embodiment.json", {"robot_type": "PandaOmron"})

    def test_resolves_parent_or_lerobot_directory(self) -> None:
        self.assertEqual(resolve_lerobot_root(self.dataset_root), self.dataset_root.resolve())
        self.assertEqual(resolve_lerobot_root(self.dataset_root.parent), self.dataset_root.resolve())

    def test_valid_atomic_seen_metadata(self) -> None:
        manifest = load_task_manifest(MANIFEST_PATH)
        summary = inspect_dataset(self.dataset_root.parent, manifest=manifest)
        self.assertTrue(summary.ok, summary.errors)
        self.assertEqual(summary.task_name, "CloseFridge")
        self.assertEqual(summary.state_dim, 16)
        self.assertEqual(summary.action_dim, 12)
        self.assertEqual(summary.task_descriptions, 1)
        self.assertEqual(summary.episode_records, 1)

    def test_rejects_task_outside_atomic_seen_manifest(self) -> None:
        manifest = load_task_manifest(MANIFEST_PATH)
        summary = inspect_dataset(self.dataset_root, manifest=manifest, task_name="PrepareCoffee")
        self.assertFalse(summary.ok)
        self.assertTrue(any("不属于 atomic-only" in error for error in summary.errors))

    def test_reports_missing_camera_and_wrong_action_dimension(self) -> None:
        info_path = self.dataset_root / "meta" / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["features"]["action"]["shape"] = [7]
        del info["features"]["observation.images.robot0_eye_in_hand"]
        write_json(info_path, info)

        summary = inspect_dataset(self.dataset_root, task_name="CloseFridge")
        self.assertFalse(summary.ok)
        self.assertTrue(any("action 维度应为 12" in error for error in summary.errors))
        self.assertTrue(any("缺少相机特征" in error for error in summary.errors))

    def test_requires_parquet_and_all_camera_videos_when_requested(self) -> None:
        manifest = load_task_manifest(MANIFEST_PATH)
        missing = inspect_dataset(
            self.dataset_root,
            manifest=manifest,
            require_data=True,
            require_videos=True,
        )
        self.assertFalse(missing.ok)
        self.assertTrue(any("Parquet" in error for error in missing.errors))
        self.assertEqual(len([error for error in missing.errors if "MP4" in error]), 3)

        parquet = self.dataset_root / "data" / "chunk-000" / "episode_000000.parquet"
        parquet.parent.mkdir(parents=True, exist_ok=True)
        parquet.touch()
        for camera in (
            "observation.images.robot0_agentview_left",
            "observation.images.robot0_agentview_right",
            "observation.images.robot0_eye_in_hand",
        ):
            video = self.dataset_root / "videos" / "chunk-000" / camera / "episode_000000.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.touch()

        complete = inspect_dataset(
            self.dataset_root,
            manifest=manifest,
            require_data=True,
            require_videos=True,
        )
        self.assertTrue(complete.ok, complete.errors)
        self.assertEqual(complete.data_files, 1)
        self.assertEqual(set(complete.video_files_by_camera.values()), {1})

    def test_audit_cli_emits_machine_readable_report(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "audit_robocasa365_dataset.py"),
                "--dataset",
                str(self.dataset_root),
                "--task-name",
                "CloseFridge",
                "--task-manifest",
                str(MANIFEST_PATH),
            ],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["ok"])
        self.assertEqual(report["task_name"], "CloseFridge")


if __name__ == "__main__":
    unittest.main()
