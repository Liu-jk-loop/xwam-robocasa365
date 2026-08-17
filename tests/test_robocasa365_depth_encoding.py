from __future__ import annotations

import copy
import unittest

import numpy as np

from project_tools.robocasa365_depth_encoding import (
    DepthEncodingError,
    decoded_frame_audit,
    encode_inverse_metric_depth,
    freeze_global_inverse_depth_encoding,
    sample_inverse_metric_depth,
    validate_frozen_encoding,
)


def _spec() -> dict[str, object]:
    return freeze_global_inverse_depth_encoding(
        [np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)],
        quantile_low=0.0,
        quantile_high=1.0,
        calibration={"seed": 42},
    )


class RoboCasa365DepthEncodingTests(unittest.TestCase):
    def test_global_encoding_is_nearer_brighter_and_three_channel(self):
        metric = np.array([[1.0, 0.5], [0.25, 2.0]], dtype=np.float32)
        encoded, report = encode_inverse_metric_depth(metric, _spec())
        self.assertEqual(encoded.shape, (2, 2, 3))
        self.assertEqual(encoded.dtype, np.uint8)
        self.assertTrue(np.array_equal(encoded[..., 0], encoded[..., 1]))
        self.assertGreater(encoded[1, 0, 0], encoded[0, 0, 0])
        self.assertEqual(report["invalid_pixels"], 0)

    def test_invalid_depth_maps_to_zero(self):
        metric = np.array([[0.0, np.nan], [-1.0, 1.0]], dtype=np.float32)
        encoded, report = encode_inverse_metric_depth(metric, _spec())
        self.assertTrue(np.all(encoded[:1] == 0))
        self.assertEqual(encoded[1, 0, 0], 0)
        self.assertEqual(report["invalid_pixels"], 3)

    def test_encoding_digest_detects_mutation(self):
        spec = copy.deepcopy(_spec())
        spec["inverse_depth_high_per_m"] = 9.0
        with self.assertRaisesRegex(DepthEncodingError, "digest"):
            validate_frozen_encoding(spec)

    def test_sampling_is_deterministic_for_same_seed(self):
        depth = np.linspace(0.1, 5.0, 100, dtype=np.float32).reshape(10, 10)
        first = sample_inverse_metric_depth(
            depth, pixels_per_frame=10, rng=np.random.default_rng(7)
        )
        second = sample_inverse_metric_depth(
            depth, pixels_per_frame=10, rng=np.random.default_rng(7)
        )
        np.testing.assert_array_equal(first, second)

    def test_decoded_audit_tolerates_small_h264_error(self):
        expected = np.full((4, 4, 3), 100, dtype=np.uint8)
        decoded = expected.copy()
        decoded[..., 1] += 1
        report = decoded_frame_audit(
            decoded, expected, max_mae=1.0, max_channel_delta=2
        )
        self.assertTrue(report["ok"])


if __name__ == "__main__":
    unittest.main()
