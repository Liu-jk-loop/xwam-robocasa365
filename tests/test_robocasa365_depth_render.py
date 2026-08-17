from __future__ import annotations

import unittest

import numpy as np

from project_tools.robocasa365_depth_render import (
    DepthRenderProbeError,
    diagnostic_inverse_depth_rgb,
    metric_depth_statistics,
    parse_frame_fractions,
    resolve_frame_indices,
    rgb_alignment_metrics,
    set_replay_episode_state,
)


class _FakeState:
    def __init__(self, values: np.ndarray):
        self.values = values

    def flatten(self) -> np.ndarray:
        return self.values.copy()


class _FakeSim:
    def __init__(self, width: int):
        self.values = np.zeros(width, dtype=np.float64)

    def get_state(self) -> _FakeState:
        return _FakeState(self.values)

    def set_state_from_flattened(self, values: np.ndarray) -> None:
        self.values = np.asarray(values, dtype=np.float64).copy()

    def forward(self) -> None:
        pass


class _FakeEnvironment:
    def __init__(self, width: int):
        self.sim = _FakeSim(width)

    def update_state(self) -> None:
        pass


class RoboCasa365DepthRenderTests(unittest.TestCase):
    def test_frame_fractions_resolve_start_middle_end(self):
        fractions = parse_frame_fractions("0,0.5,1")
        self.assertEqual(resolve_frame_indices(5, fractions), (0, 2, 4))

    def test_invalid_frame_fraction_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            parse_frame_fractions("0,1.1")

    def test_rgb_alignment_applies_mujoco_vertical_flip(self):
        source = np.zeros((4, 3, 3), dtype=np.uint8)
        source[0] = 10
        source[1] = 40
        source[2] = 100
        source[3] = 220
        metrics = rgb_alignment_metrics(source, source[::-1])
        self.assertEqual(metrics["rgb_mae"], 0.0)
        self.assertGreater(metrics["unflipped_rgb_mae"], 0.0)
        self.assertTrue(metrics["vertical_flip_applied"])

    def test_metric_depth_contract(self):
        normalized = np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32)
        metric = np.array([[0.1, 0.2], [0.4, 10.0]], dtype=np.float32)
        report = metric_depth_statistics(normalized, metric)
        self.assertTrue(report["normalized_in_unit_interval"])
        self.assertEqual(report["metric_finite_fraction"], 1.0)
        self.assertEqual(report["metric_positive_fraction"], 1.0)
        self.assertGreater(report["metric_std_m"], 0.0)

    def test_inverse_depth_preview_is_explicitly_diagnostic(self):
        preview, contract = diagnostic_inverse_depth_rgb(
            np.array([[0.2, 0.4], [1.0, 2.0]], dtype=np.float32)
        )
        self.assertEqual(preview.shape, (2, 2, 3))
        self.assertEqual(preview.dtype, np.uint8)
        self.assertFalse(contract["training_encoding"])
        self.assertEqual(contract["status"], "diagnostic_per_frame_q01_q99")

    def test_state_is_checked_against_current_episode_model(self):
        state = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        self.assertEqual(set_replay_episode_state(_FakeEnvironment(4), state), 0.0)

    def test_state_from_another_episode_model_is_rejected(self):
        with self.assertRaisesRegex(DepthRenderProbeError, "state=4, model=3"):
            set_replay_episode_state(
                _FakeEnvironment(3),
                np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
            )


if __name__ == "__main__":
    unittest.main()
