from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data.robocasa365_contract import DatasetContractError
from data.robocasa365_schema import (
    PandaOmronTensorCodec,
    load_panda_omron_schema,
    load_statistics,
    validate_dataset_modality,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "configs" / "schemas" / "robocasa365_panda_omron_v1.json"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def modality_payload() -> dict:
    return {
        "state": {
            "base_position": {"original_key": "observation.state", "start": 0, "end": 3},
            "base_rotation": {"original_key": "observation.state", "start": 3, "end": 7},
            "end_effector_position_relative": {"original_key": "observation.state", "start": 7, "end": 10},
            "end_effector_rotation_relative": {"original_key": "observation.state", "start": 10, "end": 14},
            "gripper_qpos": {"original_key": "observation.state", "start": 14, "end": 16},
        },
        "action": {
            "base_motion": {"original_key": "action", "start": 0, "end": 4},
            "control_mode": {"original_key": "action", "start": 4, "end": 5},
            "end_effector_position": {"original_key": "action", "start": 5, "end": 8},
            "end_effector_rotation": {"original_key": "action", "start": 8, "end": 11},
            "gripper_close": {"original_key": "action", "start": 11, "end": 12},
        },
        "video": {
            "robot0_eye_in_hand": {"original_key": "observation.images.robot0_eye_in_hand"},
            "robot0_agentview_left": {"original_key": "observation.images.robot0_agentview_left"},
            "robot0_agentview_right": {"original_key": "observation.images.robot0_agentview_right"},
        },
    }


class RoboCasa365SchemaTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name) / "CloseFridge" / "lerobot"
        write_json(self.root / "meta" / "info.json", {"codebase_version": "v2.1"})
        write_json(self.root / "meta" / "modality.json", modality_payload())

        state_q01 = np.arange(16, dtype=np.float64) - 8.0
        state_q99 = state_q01 + 2.0
        action_q01 = np.full(12, -0.5, dtype=np.float64)
        action_q99 = np.full(12, 0.5, dtype=np.float64)
        write_json(
            self.root / "meta" / "stats.json",
            {
                "observation.state": {
                    "q01": state_q01.tolist(),
                    "q99": state_q99.tolist(),
                    "min": (state_q01 - 1).tolist(),
                    "max": (state_q99 + 1).tolist(),
                },
                "action": {
                    "q01": action_q01.tolist(),
                    "q99": action_q99.tolist(),
                    "min": np.full(12, -1.0).tolist(),
                    "max": np.full(12, 1.0).tolist(),
                },
            },
        )

    def test_real_modality_contract_matches_versioned_schema(self) -> None:
        schema = load_panda_omron_schema(SCHEMA_PATH)
        evidence = validate_dataset_modality(self.root.parent, schema)
        self.assertEqual(schema.state.dimension, 16)
        self.assertEqual(schema.action.dimension, 12)
        self.assertEqual(len(evidence["file_sha256"]), 64)
        self.assertEqual([component.name for component in schema.action.components][1], "control_mode")

    def test_codec_roundtrip_preserves_all_dimensions(self) -> None:
        schema = load_panda_omron_schema(SCHEMA_PATH)
        codec = PandaOmronTensorCodec(
            schema,
            load_statistics(self.root, schema.state),
            load_statistics(self.root, schema.action),
        )
        state = np.stack([np.arange(16, dtype=np.float32) - 7.0, np.arange(16, dtype=np.float32) - 6.5])
        state[:, 3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        state[:, 10:14] = np.array([0.5, 0.5, 0.5, 0.5])
        action = np.zeros((2, 12), dtype=np.float32)
        action[:, 4] = np.array([-1.0, 1.0])
        action[:, 5:11] = 0.25
        action[:, 11] = -1.0

        state_encoded = codec.encode_state(state, clip=False)
        action_encoded = codec.encode_action(action, clip=False)
        np.testing.assert_allclose(codec.decode_state(state_encoded), state, atol=1e-6)
        np.testing.assert_allclose(codec.decode_action(action_encoded), action, atol=1e-6)
        self.assertEqual(action_encoded.shape[-1], 12)

        continuous_mode = action_encoded.copy()
        continuous_mode[:, 4] = np.array([-0.1, 0.1])
        decoded = codec.decode_action(continuous_mode, discretize_control_mode=True)
        np.testing.assert_array_equal(decoded[:, 4], np.array([-1.0, 1.0]))

    def test_rejects_changed_action_slice(self) -> None:
        payload = modality_payload()
        payload["action"]["control_mode"]["start"] = 5
        write_json(self.root / "meta" / "modality.json", payload)
        schema = load_panda_omron_schema(SCHEMA_PATH)
        with self.assertRaisesRegex(DatasetContractError, "不一致"):
            validate_dataset_modality(self.root, schema)

    def test_clipped_encoding_stays_in_model_range(self) -> None:
        schema = load_panda_omron_schema(SCHEMA_PATH)
        codec = PandaOmronTensorCodec(
            schema,
            load_statistics(self.root, schema.state),
            load_statistics(self.root, schema.action),
        )
        state = np.full((3, 16), 100.0, dtype=np.float32)
        action = np.full((3, 12), 100.0, dtype=np.float32)
        self.assertLessEqual(float(np.abs(codec.encode_state(state)).max()), 1.0)
        self.assertLessEqual(float(np.abs(codec.encode_action(action)).max()), 1.0)


if __name__ == "__main__":
    unittest.main()
