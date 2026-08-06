from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from data.robocasa365_contract import DatasetContractError
from data.robocasa365_index import (
    ClipIndex,
    EpisodeRecord,
    build_clip_spec,
    episode_data_path,
    episode_video_path,
    load_episode_records,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


class RoboCasa365IndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name) / "CloseFridge" / "lerobot"
        write_json(
            self.root / "meta" / "info.json",
            {
                "codebase_version": "v2.1",
                "total_episodes": 2,
                "fps": 20,
                "chunks_size": 2,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            },
        )
        write_jsonl(
            self.root / "meta" / "tasks.jsonl",
            [
                {"task_index": 0, "task": "close the refrigerator"},
                {"task_index": 1, "task": "shut the fridge door"},
            ],
        )
        write_jsonl(
            self.root / "meta" / "episodes.jsonl",
            [
                {"episode_index": 0, "length": 40, "tasks": ["close the refrigerator"]},
                {"episode_index": 2, "length": 36, "tasks": [1]},
            ],
        )

    def test_loads_prompts_and_resolves_v21_paths(self) -> None:
        root, info, episodes = load_episode_records(self.root.parent)
        self.assertEqual(root, self.root.resolve())
        self.assertEqual(episodes[0].prompts, ("close the refrigerator",))
        self.assertEqual(episodes[1].prompts, ("shut the fridge door",))
        self.assertEqual(
            episode_data_path(root, info, 2),
            root / "data" / "chunk-001" / "episode_000002.parquet",
        )
        self.assertEqual(
            episode_video_path(root, info, 2, "observation.images.left"),
            root / "videos" / "chunk-001" / "observation.images.left" / "episode_000002.mp4",
        )

    def test_clip_index_and_temporal_window_are_exact(self) -> None:
        episodes = (
            EpisodeRecord(episode_index=0, length=40, prompts=("a",)),
            EpisodeRecord(episode_index=2, length=36, prompts=("b",)),
        )
        index = ClipIndex(episodes, required_span=32)
        self.assertEqual(len(index), 12)
        self.assertEqual(index[0], (0, 0))
        self.assertEqual(index[7], (0, 7))
        self.assertEqual(index[8], (2, 0))
        self.assertEqual(index[-1], (2, 3))

        clip = build_clip_spec(2, 3, sequence_length=9, frame_skip=4, action_skip=1)
        self.assertEqual(clip.frame_ids, (3, 7, 11, 15, 19, 23, 27, 31, 35))
        self.assertEqual(clip.action_ids, tuple(range(3, 35)))

    def test_rejects_non_v2_episode_layout(self) -> None:
        info_path = self.root / "meta" / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["codebase_version"] = "v3.0"
        write_json(info_path, info)
        with self.assertRaisesRegex(DatasetContractError, "只支持 LeRobot v2"):
            load_episode_records(self.root)

    def test_rejects_action_skip_that_breaks_fixed_ratio(self) -> None:
        with self.assertRaisesRegex(ValueError, "必须能被"):
            build_clip_spec(0, 0, sequence_length=9, frame_skip=4, action_skip=3)


if __name__ == "__main__":
    unittest.main()
