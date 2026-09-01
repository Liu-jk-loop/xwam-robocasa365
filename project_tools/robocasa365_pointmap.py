"""RoboCasa365 camera-space PointMap contract and dependency-light audits.

The numerical contract follows the public Flex-Pi pointmap encoder at commit
``20c1b2b71ea35a415d5d47c39b04443cfadad7a1`` while keeping the cache layout
specific to this X-WAM adaptation.  All functions in this module are NumPy
only so the representation can be tested on the local workstation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


POINTMAP_CONTRACT_NAME = "robocasa365_camera_xyz_flexpi_v1"
POINTMAP_SCHEMA_VERSION = 1
FLEXPI_REFERENCE_COMMIT = "20c1b2b71ea35a415d5d47c39b04443cfadad7a1"
DEPTH_VALID_MIN_M = 0.01
DEPTH_VALID_MAX_M = 2.0
POINTMAP_MIN_M = np.asarray([-0.5, -0.5, 0.0], dtype=np.float32)
POINTMAP_MAX_M = np.asarray([0.5, 0.5, 1.5], dtype=np.float32)


class PointMapContractError(RuntimeError):
    """Raised when PointMap inputs violate the frozen numerical contract."""


def pointmap_contract(
    *,
    source_height: int,
    source_width: int,
    target_height: int,
    target_width: int,
) -> dict[str, Any]:
    """Return the versioned contract embedded in reports and future caches."""

    for name, value in (
        ("source_height", source_height),
        ("source_width", source_width),
        ("target_height", target_height),
        ("target_width", target_width),
    ):
        if int(value) <= 0:
            raise ValueError(f"{name} 必须为正数，实际为 {value}")
    return {
        "schema_version": POINTMAP_SCHEMA_VERSION,
        "name": POINTMAP_CONTRACT_NAME,
        "reference": {
            "project": "Flex-Pi",
            "repository": "https://github.com/geyan21/flex-pi",
            "commit": FLEXPI_REFERENCE_COMMIT,
            "source_file": "src/flexpi/models/pointmap_encoder.py",
        },
        "coordinate_frame": "opencv_camera_xyz_meters",
        "unprojection": {
            "x": "(u - cx) / fx * z",
            "y": "(v - cy) / fy * z",
            "z": "metric_depth_m",
            "pixel_origin": "top_left",
            "pixel_coordinates": "integer_centers",
            "extrinsics_required": False,
        },
        "valid_depth_m": {
            "minimum_exclusive": DEPTH_VALID_MIN_M,
            "maximum_exclusive": DEPTH_VALID_MAX_M,
            "invalid_xyz_m": [0.0, 0.0, 0.0],
        },
        "normalization": {
            "clip_min_xyz_m": POINTMAP_MIN_M.tolist(),
            "clip_max_xyz_m": POINTMAP_MAX_M.tolist(),
            "formula": "2 * (clip(xyz, min, max) - min) / (max - min) - 1",
            "range": [-1.0, 1.0],
            "invalid_normalized_xyz": [0.0, 0.0, -1.0],
        },
        "spatial": {
            "source_shape_hw": [int(source_height), int(source_width)],
            "target_shape_hw": [int(target_height), int(target_width)],
            "order": "unproject_at_source_then_resize_normalized_xyz",
            "resize": "nearest_floor",
        },
        "planned_cache": {
            "dtype": "float16",
            "layout": "per_episode_per_camera_[T,3,H,W]",
            "compression": "none_npy_memmap",
            "values": "normalized_xyz",
        },
    }


def validate_camera_intrinsics(
    intrinsics: np.ndarray,
    *,
    height: int,
    width: int,
) -> dict[str, Any]:
    """Validate one OpenCV-style 3x3 K matrix at the source depth grid."""

    matrix = np.asarray(intrinsics, dtype=np.float64)
    checks = {
        "shape_3x3": matrix.shape == (3, 3),
        "finite": bool(np.isfinite(matrix).all()),
    }
    if matrix.shape == (3, 3) and checks["finite"]:
        checks.update(
            {
                "positive_focal_lengths": bool(
                    matrix[0, 0] > 0.0 and matrix[1, 1] > 0.0
                ),
                "zero_skew": bool(abs(float(matrix[0, 1])) <= 1e-9),
                "homogeneous_row": bool(
                    np.allclose(matrix[2], np.asarray([0.0, 0.0, 1.0]), atol=1e-9)
                ),
                "principal_point_inside_image": bool(
                    0.0 <= matrix[0, 2] <= max(width - 1, 0)
                    and 0.0 <= matrix[1, 2] <= max(height - 1, 0)
                ),
            }
        )
    else:
        checks.update(
            {
                "positive_focal_lengths": False,
                "zero_skew": False,
                "homogeneous_row": False,
                "principal_point_inside_image": False,
            }
        )
    errors = [name for name, passed in checks.items() if not passed]
    return {
        "matrix": matrix.tolist() if matrix.shape == (3, 3) else None,
        "source_shape_hw": [int(height), int(width)],
        "checks": checks,
        "errors": errors,
        "ok": not errors,
    }


def metric_depth_to_camera_xyz(
    metric_depth_m: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject metric depth to HWC camera-space XYZ and a valid-pixel mask."""

    depth = np.asarray(metric_depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise PointMapContractError(f"metric depth 必须为二维，实际为 {depth.shape}")
    height, width = depth.shape
    intrinsics_report = validate_camera_intrinsics(
        intrinsics,
        height=height,
        width=width,
    )
    if not intrinsics_report["ok"]:
        raise PointMapContractError(
            "相机内参合同失败：" + ", ".join(intrinsics_report["errors"])
        )
    matrix = np.asarray(intrinsics, dtype=np.float32)
    u = np.arange(width, dtype=np.float32)[None, :]
    v = np.arange(height, dtype=np.float32)[:, None]
    valid = (
        np.isfinite(depth) & (depth > DEPTH_VALID_MIN_M) & (depth < DEPTH_VALID_MAX_M)
    )
    z = np.where(valid, depth, 0.0)
    x = (u - matrix[0, 2]) / matrix[0, 0] * z
    y = (v - matrix[1, 2]) / matrix[1, 1] * z
    xyz = np.stack((x, y, z), axis=-1).astype(np.float32, copy=False)
    return xyz, valid


def normalize_camera_xyz(pointmap_xyz_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Clip metric HWC XYZ and map it to the frozen [-1, 1] VAE range."""

    xyz = np.asarray(pointmap_xyz_m, dtype=np.float32)
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise PointMapContractError(f"PointMap 必须为 HWC XYZ，实际为 {xyz.shape}")
    if not np.isfinite(xyz).all():
        raise PointMapContractError("PointMap 包含 NaN/Inf")
    clipped = np.clip(xyz, POINTMAP_MIN_M, POINTMAP_MAX_M)
    normalized = (
        2.0 * (clipped - POINTMAP_MIN_M) / (POINTMAP_MAX_M - POINTMAP_MIN_M) - 1.0
    )
    return normalized.astype(np.float32, copy=False), clipped.astype(
        np.float32, copy=False
    )


def denormalize_camera_xyz(normalized_pointmap: np.ndarray) -> np.ndarray:
    normalized = np.asarray(normalized_pointmap, dtype=np.float32)
    if normalized.ndim != 3 or normalized.shape[-1] != 3:
        raise PointMapContractError(
            f"normalized PointMap 必须为 HWC XYZ，实际为 {normalized.shape}"
        )
    return (
        (normalized + 1.0) * 0.5 * (POINTMAP_MAX_M - POINTMAP_MIN_M) + POINTMAP_MIN_M
    ).astype(np.float32, copy=False)


def resize_nearest_hwc(
    array: np.ndarray, target_height: int, target_width: int
) -> np.ndarray:
    """Dependency-light equivalent of nearest resize for an HWC tensor."""

    source = np.asarray(array)
    if source.ndim != 3:
        raise ValueError(f"nearest resize 输入必须为 HWC，实际为 {source.shape}")
    if target_height <= 0 or target_width <= 0:
        raise ValueError("target height/width 必须为正数")
    source_height, source_width = source.shape[:2]
    rows = np.floor(
        np.arange(target_height, dtype=np.float64) * source_height / target_height
    ).astype(np.int64)
    columns = np.floor(
        np.arange(target_width, dtype=np.float64) * source_width / target_width
    ).astype(np.int64)
    rows = np.clip(rows, 0, source_height - 1)
    columns = np.clip(columns, 0, source_width - 1)
    return source[rows[:, None], columns[None, :]]


def _axis_summary(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "minimum": np.min(values, axis=0).astype(float).tolist(),
        "maximum": np.max(values, axis=0).astype(float).tolist(),
        "mean": np.mean(values, axis=0).astype(float).tolist(),
    }


def audit_pointmap_frame(
    metric_depth_m: np.ndarray,
    intrinsics: np.ndarray,
    *,
    target_height: int,
    target_width: int,
    max_reprojection_error_px: float = 1e-3,
    max_float16_roundtrip_error_m: float = 0.002,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build one normalized float16 PointMap and audit all reversible steps."""

    if not math.isfinite(max_reprojection_error_px) or max_reprojection_error_px < 0:
        raise ValueError("max_reprojection_error_px 必须为非负有限数")
    if (
        not math.isfinite(max_float16_roundtrip_error_m)
        or max_float16_roundtrip_error_m < 0
    ):
        raise ValueError("max_float16_roundtrip_error_m 必须为非负有限数")

    depth = np.asarray(metric_depth_m, dtype=np.float32)
    height, width = depth.shape
    intrinsics_report = validate_camera_intrinsics(
        intrinsics,
        height=height,
        width=width,
    )
    xyz, valid = metric_depth_to_camera_xyz(depth, intrinsics)
    normalized, clipped_xyz = normalize_camera_xyz(xyz)

    matrix = np.asarray(intrinsics, dtype=np.float64)
    valid_xyz = xyz[valid].astype(np.float64)
    valid_depth = depth[valid].astype(np.float64)
    if valid_xyz.size:
        projected_u = matrix[0, 0] * valid_xyz[:, 0] / valid_xyz[:, 2] + matrix[0, 2]
        projected_v = matrix[1, 1] * valid_xyz[:, 1] / valid_xyz[:, 2] + matrix[1, 2]
        grid_v, grid_u = np.nonzero(valid)
        reprojection_error = np.sqrt(
            np.square(projected_u - grid_u) + np.square(projected_v - grid_v)
        )
        depth_error = np.abs(valid_xyz[:, 2] - valid_depth)
    else:
        reprojection_error = np.asarray([], dtype=np.float64)
        depth_error = np.asarray([], dtype=np.float64)

    target_normalized = resize_nearest_hwc(
        normalized,
        target_height,
        target_width,
    ).astype(np.float32, copy=False)
    target_clipped_xyz = resize_nearest_hwc(
        clipped_xyz,
        target_height,
        target_width,
    ).astype(np.float32, copy=False)
    cache_float16 = target_normalized.astype(np.float16)
    roundtrip_xyz = denormalize_camera_xyz(cache_float16.astype(np.float32))
    roundtrip_error = np.abs(roundtrip_xyz - target_clipped_xyz)

    valid_count = int(valid.sum())
    valid_values = xyz[valid]
    clip_low = (
        (valid_values < POINTMAP_MIN_M).mean(axis=0) if valid_count else np.ones(3)
    )
    clip_high = (
        (valid_values > POINTMAP_MAX_M).mean(axis=0) if valid_count else np.ones(3)
    )
    invalid_expected = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    invalid_contract = bool(
        np.allclose(normalized[~valid], invalid_expected, atol=1e-7)
    )

    checks = {
        "intrinsics": bool(intrinsics_report["ok"]),
        "valid_pixels_present": valid_count > 0,
        "xyz_finite": bool(np.isfinite(xyz).all()),
        "invalid_xyz_zero": bool(np.all(xyz[~valid] == 0.0)),
        "normalized_finite": bool(np.isfinite(target_normalized).all()),
        "normalized_range": bool(
            np.min(target_normalized) >= -1.000001
            and np.max(target_normalized) <= 1.000001
        ),
        "invalid_normalized_sentinel": invalid_contract,
        "target_shape": target_normalized.shape
        == (int(target_height), int(target_width), 3),
        "reprojection": bool(
            reprojection_error.size
            and float(np.max(reprojection_error)) <= max_reprojection_error_px
        ),
        "depth_roundtrip": bool(
            depth_error.size and float(np.max(depth_error)) <= 1e-6
        ),
        "float16_roundtrip": bool(
            float(np.max(roundtrip_error)) <= max_float16_roundtrip_error_m
        ),
    }
    errors = [name for name, passed in checks.items() if not passed]
    report = {
        "contract_name": POINTMAP_CONTRACT_NAME,
        "source_shape_hw": [int(height), int(width)],
        "target_shape_chw": [3, int(target_height), int(target_width)],
        "intrinsics": intrinsics_report,
        "valid_pixels": valid_count,
        "invalid_pixels": int(valid.size - valid_count),
        "valid_fraction": float(np.mean(valid)),
        "valid_depth_m": {
            "minimum": float(np.min(valid_depth)) if valid_count else None,
            "maximum": float(np.max(valid_depth)) if valid_count else None,
        },
        "valid_xyz_m": _axis_summary(valid_values) if valid_count else None,
        "clipped_fraction": {
            "low_xyz": np.asarray(clip_low, dtype=float).tolist(),
            "high_xyz": np.asarray(clip_high, dtype=float).tolist(),
        },
        "reprojection_error_px": {
            "maximum": float(np.max(reprojection_error))
            if reprojection_error.size
            else None,
            "mean": float(np.mean(reprojection_error))
            if reprojection_error.size
            else None,
            "threshold": float(max_reprojection_error_px),
        },
        "depth_roundtrip_error_m": {
            "maximum": float(np.max(depth_error)) if depth_error.size else None,
        },
        "float16_cache_roundtrip_error_m": {
            "maximum": float(np.max(roundtrip_error)),
            "mean": float(np.mean(roundtrip_error)),
            "threshold": float(max_float16_roundtrip_error_m),
        },
        "cache_dtype": str(cache_float16.dtype),
        "checks": checks,
        "errors": errors,
        "ok": not errors,
    }
    arrays = {
        "camera_intrinsics": np.asarray(intrinsics, dtype=np.float64),
        "pointmap_xyz_m": xyz,
        "pointmap_valid_mask": valid,
        "pointmap_normalized_float16": np.moveaxis(cache_float16, -1, 0),
    }
    return arrays, report


def summarize_intrinsics(
    matrices_by_camera: dict[str, list[np.ndarray]],
    *,
    expected_observations: int,
    max_drift: float = 1e-6,
) -> dict[str, Any]:
    """Require stable per-camera K across all sampled frames and episodes."""

    if expected_observations < 0:
        raise PointMapContractError("expected_observations 必须为非负整数")
    if not math.isfinite(max_drift) or max_drift < 0.0:
        raise PointMapContractError("max_drift 必须为有限非负数")

    cameras: dict[str, Any] = {}
    errors: list[str] = []
    for camera_name, matrices in matrices_by_camera.items():
        count = len(matrices)
        if count:
            stack = np.stack([np.asarray(item, dtype=np.float64) for item in matrices])
            drift = float(np.max(np.abs(stack - stack[0])))
            matrix = stack[0].tolist()
        else:
            drift = math.inf
            matrix = None
        ok = count == expected_observations and drift <= max_drift
        if not ok:
            errors.append(
                f"{camera_name}: observations={count}/{expected_observations}, "
                f"max_abs_drift={drift}"
            )
        cameras[camera_name] = {
            "observations": count,
            "expected_observations": expected_observations,
            "matrix": matrix,
            "max_abs_drift": drift,
            "max_allowed_drift": max_drift,
            "ok": ok,
        }
    if len(cameras) != 3:
        errors.append(f"期望三路相机内参，实际为 {sorted(cameras)}")
    return {
        "coordinate_frame": "opencv_camera_xyz_meters",
        "camera_count": len(cameras),
        "cameras": cameras,
        "errors": errors,
        "ok": not errors,
    }
