from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from project_tools.robocasa365_depth_preflight import inspect_depth_replay_inputs


class RoboCasa365DepthPreflightTests(unittest.TestCase):
    def _dataset(
        self,
        root: Path,
        *,
        state_frames: int = 4,
        state_widths: tuple[int, ...] = (17,),
    ) -> Path:
        lerobot = root / "lerobot"
        (lerobot / "meta").mkdir(parents=True)
        (lerobot / "extras").mkdir(parents=True)
        info = {
            "codebase_version": "v2.1",
            "total_episodes": len(state_widths),
            "features": {
                name: {"dtype": "video", "shape": [256, 256, 3]}
                for name in (
                    "observation.images.robot0_agentview_left",
                    "observation.images.robot0_agentview_right",
                    "observation.images.robot0_eye_in_hand",
                )
            },
        }
        (lerobot / "meta" / "info.json").write_text(json.dumps(info))
        (lerobot / "meta" / "tasks.jsonl").write_text(
            json.dumps({"task_index": 0, "task": "close the fridge"}) + "\n"
        )
        (lerobot / "meta" / "episodes.jsonl").write_text(
            "".join(
                json.dumps({"episode_index": index, "length": 4, "tasks": [0]})
                + "\n"
                for index in range(len(state_widths))
            )
        )
        (lerobot / "extras" / "dataset_meta.json").write_text(
            json.dumps({"env_args": {"env_name": "CloseFridge"}})
        )
        for index, state_width in enumerate(state_widths):
            episode = lerobot / "extras" / f"episode_{index:06d}"
            episode.mkdir(parents=True)
            np.savez_compressed(
                episode / "states.npz",
                states=np.ones((state_frames, state_width), dtype=np.float64),
            )
            (episode / "ep_meta.json").write_text(json.dumps({"lang": "close"}))
            with gzip.open(episode / "model.xml.gz", "wb") as handle:
                handle.write(b"<mujoco model='fixture'></mujoco>")
        return root

    def test_valid_official_extras_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = inspect_depth_replay_inputs(
                self._dataset(Path(tmp)),
                task_name="CloseFridge",
                episodes_to_decode=1,
            )
        self.assertTrue(report["ok"])
        self.assertEqual(report["state_widths"], [17])
        self.assertEqual(report["depth_target_contract"]["semantic_target"], "inverse_depth")
        self.assertEqual(
            report["depth_target_contract"]["pixel_encoding_formula_status"],
            "requires_render_probe_validation",
        )

    def test_state_width_may_vary_when_each_episode_has_its_own_mjcf(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = inspect_depth_replay_inputs(
                self._dataset(Path(tmp), state_widths=(17, 19, 23)),
                task_name="CloseFridge",
                episodes_to_decode=3,
            )
        self.assertTrue(report["ok"])
        self.assertEqual(report["state_widths"], [17, 19, 23])
        self.assertEqual(report["state_width_contract"], "per_episode_mjcf")
        self.assertTrue(any("model.xml.gz" in message for message in report["warnings"]))

    def test_state_frame_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = inspect_depth_replay_inputs(
                self._dataset(Path(tmp), state_frames=3),
                task_name="CloseFridge",
                episodes_to_decode=1,
            )
        self.assertFalse(report["ok"])
        self.assertTrue(any("帧数" in message for message in report["errors"]))

    def test_missing_episode_file_fails_before_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._dataset(Path(tmp))
            (dataset / "lerobot" / "extras" / "episode_000000" / "model.xml.gz").unlink()
            report = inspect_depth_replay_inputs(
                dataset,
                task_name="CloseFridge",
                episodes_to_decode=1,
            )
        self.assertFalse(report["ok"])
        self.assertEqual(report["episodes_with_complete_file_set"], 0)
        self.assertEqual(report["missing_episode_files"][0]["missing"], ["model.xml.gz"])


if __name__ == "__main__":
    unittest.main()
