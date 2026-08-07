from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data.robocasa365_schema import load_panda_omron_schema
from evaluation.robocasa365_benchmark import (
    BenchmarkContractError,
    aggregate_episode_records,
    flat_action_to_gym_dict,
    load_atomic_task_manifest,
    load_random_smoke_config,
    pack_online_cameras,
    pack_online_state,
    sample_random_flat_action,
    tile_camera_views,
    validate_gym_action_space,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_MANIFEST = REPO_ROOT / "configs" / "tasks" / "robocasa365_atomic_seen.json"
SCHEMA_PATH = REPO_ROOT / "configs" / "schemas" / "robocasa365_panda_omron_v1.json"
CONFIG_PATH = REPO_ROOT / "configs" / "evaluation" / "robocasa365_close_fridge_m4_random_smoke.json"


class RoboCasa365BenchmarkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = load_panda_omron_schema(SCHEMA_PATH)

    def test_atomic_seen_horizons_are_complete_and_versioned(self) -> None:
        manifest = load_atomic_task_manifest(TASK_MANIFEST)
        self.assertEqual(len(manifest["tasks"]), 18)
        self.assertEqual(manifest["horizons"]["CloseFridge"], 900)
        self.assertEqual(set(manifest["tasks"]), set(manifest["horizons"]))

    def test_random_smoke_config_is_atomic_fixed_scene_and_short_horizon(self) -> None:
        config = load_random_smoke_config(CONFIG_PATH, REPO_ROOT)
        self.assertEqual(config["task"], "CloseFridge")
        self.assertEqual(config["official_horizon"], 900)
        self.assertEqual(config["rollout"]["max_steps"], 20)
        self.assertEqual(config["scene"]["obj_instance_split"], "target")
        self.assertEqual((config["scene"]["layout_id"], config["scene"]["style_id"]), (1, 1))
        self.assertEqual(config["rollout"]["depth_mode"], "disabled")

    def test_online_observation_packs_exact_16d_and_three_cameras(self) -> None:
        observation = {}
        for component in self.schema.state.components:
            observation[f"state.{component.name}"] = np.arange(component.size, dtype=np.float32)
        for name in ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"):
            observation[f"video.{name}"] = np.zeros((256, 256, 3), dtype=np.uint8)
        state = pack_online_state(observation, self.schema)
        cameras = pack_online_cameras(
            observation,
            ["video.robot0_agentview_left", "video.robot0_agentview_right", "video.robot0_eye_in_hand"],
            [256, 256, 3],
        )
        tiled = tile_camera_views(cameras)
        self.assertEqual(state.shape, (16,))
        self.assertEqual(cameras.shape, (3, 256, 256, 3))
        self.assertEqual(tiled.shape, (256, 768, 3))

    def test_flat_action_maps_every_named_dimension_without_reordering_loss(self) -> None:
        flat = np.arange(12, dtype=np.float32)
        action = flat_action_to_gym_dict(flat, self.schema)
        np.testing.assert_array_equal(action["action.base_motion"], flat[0:4])
        np.testing.assert_array_equal(action["action.control_mode"], flat[4:5])
        np.testing.assert_array_equal(action["action.end_effector_position"], flat[5:8])
        np.testing.assert_array_equal(action["action.end_effector_rotation"], flat[8:11])
        np.testing.assert_array_equal(action["action.gripper_close"], flat[11:12])
        self.assertEqual(sum(value.size for value in action.values()), 12)

    def test_random_action_exercises_base_and_discrete_components(self) -> None:
        action = sample_random_flat_action(
            np.random.default_rng(0), self.schema, base_motion_mode="sample"
        )
        self.assertGreater(np.linalg.norm(action[0:4]), 0.0)
        self.assertIn(float(action[4]), (-1.0, 1.0))
        self.assertIn(float(action[11]), (-1.0, 1.0))

    def test_gym_action_validation_rejects_out_of_range_component(self) -> None:
        class Box:
            shape = (1,)
            low = np.asarray([-1.0], dtype=np.float32)
            high = np.asarray([1.0], dtype=np.float32)

        class DictSpace:
            spaces = {f"action.{component.name}": Box() for component in self.schema.action.components}

        space = DictSpace()
        action = {
            f"action.{component.name}": np.zeros(component.size, dtype=np.float32)
            for component in self.schema.action.components
        }
        for component in self.schema.action.components:
            space.spaces[f"action.{component.name}"].shape = (component.size,)
            space.spaces[f"action.{component.name}"].low = np.full(component.size, -1.0, dtype=np.float32)
            space.spaces[f"action.{component.name}"].high = np.full(component.size, 1.0, dtype=np.float32)
        validate_gym_action_space(action, space)
        action["action.base_motion"][0] = 1.5
        with self.assertRaises(BenchmarkContractError):
            validate_gym_action_space(action, space)

    def test_contract_rejects_missing_camera_and_composite_scope(self) -> None:
        with self.assertRaises(BenchmarkContractError):
            pack_online_cameras({}, ["left", "right", "hand"], [256, 256, 3])
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tasks.json"
            path.write_text(
                json.dumps({"scope": "composite", "tasks": ["A"], "horizons": {"A": 1}}),
                encoding="utf-8",
            )
            with self.assertRaises(BenchmarkContractError):
                load_atomic_task_manifest(path)

    def test_random_smoke_rejects_scene_or_camera_drift(self) -> None:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.json"
            payload["scene"]["randomize_cameras"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(BenchmarkContractError):
                load_random_smoke_config(path, REPO_ROOT)

            payload["scene"]["randomize_cameras"] = False
            payload["video"]["camera_keys"] = list(reversed(payload["video"]["camera_keys"]))
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(BenchmarkContractError):
                load_random_smoke_config(path, REPO_ROOT)

    def test_episode_aggregation_is_recomputable(self) -> None:
        summary = aggregate_episode_records(
            [
                {
                    "episode_id": "e0",
                    "result": "pass",
                    "success": True,
                    "steps": 3,
                    "end_reason": "success",
                    "action_diagnostics": {"base_nonzero_steps": 3},
                },
                {
                    "episode_id": "e1",
                    "result": "pass",
                    "success": False,
                    "steps": 5,
                    "end_reason": "max_steps",
                    "action_diagnostics": {"base_nonzero_steps": 5},
                },
            ]
        )
        self.assertEqual(summary["successes"], 1)
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["total_steps"], 8)
        self.assertEqual(summary["base_nonzero_steps"], 8)

    def test_episode_aggregation_detects_missing_rollouts(self) -> None:
        summary = aggregate_episode_records(
            [
                {
                    "episode_id": "e0",
                    "result": "pass",
                    "success": False,
                    "steps": 2,
                    "end_reason": "max_steps",
                }
            ],
            expected_episodes=2,
        )
        self.assertEqual(summary["episodes_missing"], 1)
        self.assertEqual(summary["end_reason_counts"]["missing"], 1)

    def test_cli_help_does_not_import_robocasa(self) -> None:
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "evaluation" / "run_robocasa365_random_rollout.py"), "--help"],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--max-steps", result.stdout)


if __name__ == "__main__":
    unittest.main()
