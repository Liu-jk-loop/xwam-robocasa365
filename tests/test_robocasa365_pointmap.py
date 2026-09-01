from __future__ import annotations

import unittest

import numpy as np

from project_tools.robocasa365_pointmap import (
    DEPTH_VALID_MAX_M,
    DEPTH_VALID_MIN_M,
    FLEXPI_REFERENCE_COMMIT,
    POINTMAP_CONTRACT_NAME,
    PointMapContractError,
    audit_pointmap_frame,
    denormalize_camera_xyz,
    metric_depth_to_camera_xyz,
    normalize_camera_xyz,
    pointmap_contract,
    resize_nearest_hwc,
    summarize_intrinsics,
    validate_camera_intrinsics,
)


class RoboCasa365PointMapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.intrinsics = np.asarray(
            [[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def test_contract_freezes_flexpi_camera_xyz_and_direct_float16_cache(self):
        contract = pointmap_contract(
            source_height=256,
            source_width=256,
            target_height=256,
            target_width=320,
        )
        self.assertEqual(contract["name"], POINTMAP_CONTRACT_NAME)
        self.assertEqual(contract["reference"]["commit"], FLEXPI_REFERENCE_COMMIT)
        self.assertEqual(contract["coordinate_frame"], "opencv_camera_xyz_meters")
        self.assertEqual(contract["valid_depth_m"]["minimum_exclusive"], 0.01)
        self.assertEqual(contract["valid_depth_m"]["maximum_exclusive"], 2.0)
        self.assertEqual(contract["planned_cache"]["dtype"], "float16")
        self.assertEqual(contract["planned_cache"]["compression"], "none_npy_memmap")
        self.assertEqual(contract["spatial"]["target_shape_hw"], [256, 320])

    def test_metric_depth_unprojects_to_camera_xyz_and_reprojects(self):
        depth = np.ones((3, 3), dtype=np.float32)
        xyz, valid = metric_depth_to_camera_xyz(depth, self.intrinsics)
        np.testing.assert_allclose(xyz[1, 1], [0.0, 0.0, 1.0], atol=1e-7)
        np.testing.assert_allclose(xyz[1, 2], [0.01, 0.0, 1.0], atol=1e-7)
        self.assertTrue(valid.all())

        _, report = audit_pointmap_frame(
            depth,
            self.intrinsics,
            target_height=3,
            target_width=4,
        )
        self.assertTrue(report["checks"]["reprojection"])
        self.assertTrue(report["checks"]["depth_roundtrip"])
        self.assertTrue(report["ok"])

    def test_strict_depth_bounds_and_invalid_sentinel_match_flexpi(self):
        depth = np.asarray(
            [
                [DEPTH_VALID_MIN_M, DEPTH_VALID_MIN_M + 0.001],
                [DEPTH_VALID_MAX_M - 0.001, DEPTH_VALID_MAX_M],
            ],
            dtype=np.float32,
        )
        intrinsics = np.asarray(
            [[100.0, 0.0, 0.5], [0.0, 100.0, 0.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        xyz, valid = metric_depth_to_camera_xyz(depth, intrinsics)
        np.testing.assert_array_equal(valid, [[False, True], [True, False]])
        np.testing.assert_array_equal(xyz[~valid], np.zeros((2, 3), dtype=np.float32))
        normalized, _ = normalize_camera_xyz(xyz)
        np.testing.assert_allclose(
            normalized[~valid],
            np.asarray([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
        )

    def test_normalization_roundtrip_returns_clipped_metric_xyz(self):
        xyz = np.asarray(
            [[[0.75, -0.75, 1.75], [0.25, -0.25, 0.75]]],
            dtype=np.float32,
        )
        normalized, clipped = normalize_camera_xyz(xyz)
        self.assertGreaterEqual(float(normalized.min()), -1.0)
        self.assertLessEqual(float(normalized.max()), 1.0)
        np.testing.assert_allclose(denormalize_camera_xyz(normalized), clipped)
        np.testing.assert_allclose(clipped[0, 0], [0.5, -0.5, 1.5])

    def test_nearest_floor_resize_has_deterministic_indices(self):
        source = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
        resized = resize_nearest_hwc(source, 2, 4)
        np.testing.assert_array_equal(resized[:, 0], source[:, 0])
        np.testing.assert_array_equal(resized[:, 1], source[:, 0])
        np.testing.assert_array_equal(resized[:, 2], source[:, 1])
        np.testing.assert_array_equal(resized[:, 3], source[:, 1])

    def test_float16_cache_roundtrip_is_audited_in_metric_space(self):
        depth = np.linspace(0.1, 1.4, 9, dtype=np.float32).reshape(3, 3)
        arrays, report = audit_pointmap_frame(
            depth,
            self.intrinsics,
            target_height=4,
            target_width=5,
            max_float16_roundtrip_error_m=0.002,
        )
        self.assertEqual(arrays["pointmap_normalized_float16"].dtype, np.float16)
        self.assertEqual(arrays["pointmap_normalized_float16"].shape, (3, 4, 5))
        self.assertTrue(report["checks"]["float16_roundtrip"])
        self.assertLessEqual(
            report["float16_cache_roundtrip_error_m"]["maximum"],
            0.002,
        )

    def test_bad_intrinsics_are_rejected_before_unprojection(self):
        bad = self.intrinsics.copy()
        bad[0, 0] = 0.0
        report = validate_camera_intrinsics(bad, height=3, width=3)
        self.assertFalse(report["ok"])
        with self.assertRaisesRegex(PointMapContractError, "相机内参合同失败"):
            metric_depth_to_camera_xyz(np.ones((3, 3), dtype=np.float32), bad)

    def test_three_camera_intrinsics_must_be_stable_and_complete(self):
        matrices = {
            "left": [self.intrinsics.copy(), self.intrinsics.copy()],
            "right": [self.intrinsics.copy(), self.intrinsics.copy()],
            "wrist": [self.intrinsics.copy(), self.intrinsics.copy()],
        }
        report = summarize_intrinsics(matrices, expected_observations=2)
        self.assertTrue(report["ok"])
        matrices["wrist"][1][0, 0] += 0.1
        report = summarize_intrinsics(matrices, expected_observations=2)
        self.assertFalse(report["ok"])
        self.assertFalse(report["cameras"]["wrist"]["ok"])


if __name__ == "__main__":
    unittest.main()
