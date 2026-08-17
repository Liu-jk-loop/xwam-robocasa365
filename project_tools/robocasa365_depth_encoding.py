"""Versioned, dependency-light depth encoding helpers for RoboCasa365."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ENCODING_NAME = "robocasa365_inverse_metric_global_q_v1"
ENCODING_SCHEMA_VERSION = 1


class DepthEncodingError(RuntimeError):
    """Raised when a frozen encoding contract is missing or inconsistent."""


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def depth_cache_video_path(
    cache_root: str | Path,
    *,
    task_name: str,
    episode_index: int,
    camera_key: str,
    chunks_size: int,
) -> Path:
    """Resolve the cache layout shared by the generator and training loader."""

    if chunks_size <= 0:
        raise ValueError("chunks_size 必须为正整数")
    if episode_index < 0:
        raise ValueError("episode_index 不能为负数")
    return (
        Path(cache_root).expanduser().resolve()
        / task_name
        / "videos"
        / f"chunk-{episode_index // chunks_size:03d}"
        / camera_key
        / f"episode_{episode_index:06d}.mp4"
    )


def depth_cache_sidecar_path(video_path: str | Path) -> Path:
    return Path(video_path).with_suffix(".depth.json")


def read_frozen_depth_encoding(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DepthEncodingError(f"无法读取depth encoding：{resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DepthEncodingError(f"depth encoding顶层必须为对象：{resolved}")
    validate_frozen_encoding(payload)
    return payload


def validate_task_depth_cache(
    *,
    cache_root: str | Path,
    encoding_path: str | Path,
    manifest_path: str | Path,
    task_name: str,
    episode_lengths: dict[int, int],
    camera_keys: Iterable[str],
    chunks_size: int,
) -> dict[str, Any]:
    """Validate one task's immutable cache contract without decoding videos."""

    root = Path(cache_root).expanduser().resolve()
    encoding_file = Path(encoding_path).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve()
    encoding = read_frozen_depth_encoding(encoding_file)
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DepthEncodingError(f"无法读取depth cache manifest：{manifest_file}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise DepthEncodingError(f"depth cache manifest顶层必须为对象：{manifest_file}")
    if manifest.get("ok") is not True or manifest.get("result") != "pass":
        raise DepthEncodingError(f"depth cache manifest不是PASS：{manifest_file}")
    if manifest.get("encoding_sha256") != encoding["encoding_sha256"]:
        raise DepthEncodingError("depth cache manifest与encoding digest不一致")
    if Path(str(manifest.get("cache_root", ""))).expanduser().resolve() != root:
        raise DepthEncodingError(
            f"depth cache root漂移：manifest={manifest.get('cache_root')}, runtime={root}"
        )

    task_reports = [
        item
        for item in manifest.get("tasks", [])
        if isinstance(item, dict) and item.get("task_name") == task_name
    ]
    if len(task_reports) != 1:
        raise DepthEncodingError(
            f"depth cache manifest中任务{task_name}应恰好出现一次，实际为{len(task_reports)}"
        )
    task_report = task_reports[0]
    if task_report.get("ok") is not True:
        raise DepthEncodingError(f"任务{task_name}的depth cache不是PASS")
    episode_reports = {
        int(item["episode_index"]): item
        for item in task_report.get("episodes", [])
        if isinstance(item, dict) and "episode_index" in item
    }
    expected_episode_indices = set(episode_lengths)
    if set(episode_reports) != expected_episode_indices:
        raise DepthEncodingError(
            "depth cache episode集合不一致："
            f"missing={sorted(expected_episode_indices - set(episode_reports))[:8]}, "
            f"extra={sorted(set(episode_reports) - expected_episode_indices)[:8]}"
        )

    expected_cameras = tuple(str(key) for key in camera_keys)
    video_paths: dict[tuple[int, str], Path] = {}
    sidecar_paths: dict[tuple[int, str], Path] = {}
    for episode_index, episode_length in episode_lengths.items():
        episode_report = episode_reports[episode_index]
        if int(episode_report.get("episode_length", -1)) != int(episode_length):
            raise DepthEncodingError(
                f"episode_{episode_index:06d}长度与cache manifest不一致"
            )
        cameras = {
            str(item.get("camera_key")): item
            for item in episode_report.get("cameras", [])
            if isinstance(item, dict)
        }
        if set(cameras) != set(expected_cameras):
            raise DepthEncodingError(
                f"episode_{episode_index:06d}相机集合不一致：{sorted(cameras)}"
            )
        for camera_key in expected_cameras:
            report = cameras[camera_key]
            video_path = depth_cache_video_path(
                root,
                task_name=task_name,
                episode_index=episode_index,
                camera_key=camera_key,
                chunks_size=chunks_size,
            )
            sidecar_path = depth_cache_sidecar_path(video_path)
            if Path(str(report.get("video_path", ""))).expanduser().resolve() != video_path:
                raise DepthEncodingError(f"cache video路径漂移：{video_path}")
            if report.get("encoding_sha256") != encoding["encoding_sha256"]:
                raise DepthEncodingError(f"cache video encoding漂移：{video_path}")
            if int(report.get("frame_count", -1)) != int(episode_length):
                raise DepthEncodingError(f"cache video帧数漂移：{video_path}")
            if report.get("ok") is not True:
                raise DepthEncodingError(f"cache video审计未通过：{video_path}")
            if not video_path.is_file() or not sidecar_path.is_file():
                raise DepthEncodingError(f"cache video或sidecar缺失：{video_path}")
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if not isinstance(sidecar, dict) or sidecar != report:
                raise DepthEncodingError(f"cache sidecar与manifest记录不一致：{sidecar_path}")
            video_paths[(episode_index, camera_key)] = video_path
            sidecar_paths[(episode_index, camera_key)] = sidecar_path

    return {
        "cache_root": str(root),
        "encoding_path": str(encoding_file),
        "encoding_name": encoding["encoding_name"],
        "encoding_sha256": encoding["encoding_sha256"],
        "manifest_path": str(manifest_file),
        "task_name": task_name,
        "episode_count": len(episode_lengths),
        "video_count": len(video_paths),
        "video_paths": video_paths,
        "sidecar_paths": sidecar_paths,
    }


def sample_inverse_metric_depth(
    metric_depth: np.ndarray,
    *,
    pixels_per_frame: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select a deterministic random subset of valid inverse-depth pixels."""

    if pixels_per_frame <= 0:
        raise ValueError("pixels_per_frame 必须为正整数")
    depth = np.asarray(metric_depth, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 0.0)
    values = 1.0 / depth[valid]
    if not values.size:
        raise DepthEncodingError("metric depth 没有有限正值，无法标定")
    if values.size > pixels_per_frame:
        indices = rng.choice(values.size, size=pixels_per_frame, replace=False)
        values = values[indices]
    return np.asarray(values, dtype=np.float32)


def freeze_global_inverse_depth_encoding(
    samples: Iterable[np.ndarray],
    *,
    quantile_low: float,
    quantile_high: float,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    """Freeze one task/camera-independent inverse-depth range."""

    if not 0.0 <= quantile_low < quantile_high <= 1.0:
        raise ValueError("quantile 必须满足 0 <= low < high <= 1")
    arrays = [np.asarray(item, dtype=np.float32).reshape(-1) for item in samples]
    if not arrays or not any(item.size for item in arrays):
        raise DepthEncodingError("没有可用于标定的 inverse-depth 样本")
    values = np.concatenate([item for item in arrays if item.size])
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise DepthEncodingError("标定样本必须全部为有限正数")
    lower, upper = np.quantile(values, [quantile_low, quantile_high])
    if not math.isfinite(float(lower)) or not math.isfinite(float(upper)) or upper <= lower:
        raise DepthEncodingError(
            f"全局 inverse-depth 标定范围退化：lower={lower}, upper={upper}"
        )
    payload: dict[str, Any] = {
        "schema_version": ENCODING_SCHEMA_VERSION,
        "encoding_name": ENCODING_NAME,
        "semantic": "inverse_metric_depth",
        "source_unit": "meter",
        "nearer_is_brighter": True,
        "output": {
            "dtype": "uint8",
            "channels": 3,
            "channel_contract": "repeat_grayscale",
            "range": [0, 255],
            "video_codec": "h264",
            "pixel_format": "yuv420p",
        },
        "formula": (
            "round(255 * clip(((1 / depth_m) - inverse_depth_low) / "
            "(inverse_depth_high - inverse_depth_low), 0, 1))"
        ),
        "invalid_depth_value": 0,
        "quantile_low": float(quantile_low),
        "quantile_high": float(quantile_high),
        "inverse_depth_low_per_m": float(lower),
        "inverse_depth_high_per_m": float(upper),
        "calibration_sample_count": int(values.size),
        "calibration_inverse_depth_min_per_m": float(np.min(values)),
        "calibration_inverse_depth_max_per_m": float(np.max(values)),
        "calibration": calibration,
        "upstream_compatibility": {
            "storage_and_loader_contract": "X-WAM three-channel uint8 depth MP4",
            "numeric_formula_claim": "project-defined; upstream formula is not public",
        },
    }
    payload["encoding_sha256"] = canonical_json_sha256(payload)
    return payload


def validate_frozen_encoding(spec: dict[str, Any]) -> None:
    if spec.get("encoding_name") != ENCODING_NAME:
        raise DepthEncodingError(f"不支持的 depth encoding：{spec.get('encoding_name')!r}")
    expected_digest = spec.get("encoding_sha256")
    unsigned = dict(spec)
    unsigned.pop("encoding_sha256", None)
    actual_digest = canonical_json_sha256(unsigned)
    if expected_digest != actual_digest:
        raise DepthEncodingError(
            f"depth encoding digest 不一致：expected={expected_digest}, actual={actual_digest}"
        )
    lower = float(spec["inverse_depth_low_per_m"])
    upper = float(spec["inverse_depth_high_per_m"])
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        raise DepthEncodingError(f"depth encoding 范围非法：{lower}, {upper}")


def encode_inverse_metric_depth(
    metric_depth: np.ndarray,
    spec: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode metric depth using one already-frozen global range."""

    validate_frozen_encoding(spec)
    depth = np.asarray(metric_depth, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"metric_depth 必须是二维数组，实际为 {depth.shape}")
    valid = np.isfinite(depth) & (depth > 0.0)
    inverse = np.zeros_like(depth)
    inverse[valid] = 1.0 / depth[valid]
    lower = float(spec["inverse_depth_low_per_m"])
    upper = float(spec["inverse_depth_high_per_m"])
    scaled = np.zeros_like(depth)
    scaled[valid] = np.clip((inverse[valid] - lower) / (upper - lower), 0.0, 1.0)
    gray = np.rint(scaled * 255.0).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    valid_count = int(valid.sum())
    return rgb, {
        "pixels": int(depth.size),
        "valid_pixels": valid_count,
        "invalid_pixels": int(depth.size - valid_count),
        "clipped_low_pixels": int(np.sum(valid & (inverse <= lower))),
        "clipped_high_pixels": int(np.sum(valid & (inverse >= upper))),
        "encoded_min": int(gray.min()),
        "encoded_max": int(gray.max()),
    }


def decoded_frame_audit(
    decoded: np.ndarray,
    expected: np.ndarray,
    *,
    max_mae: float,
    max_channel_delta: int,
) -> dict[str, Any]:
    """Audit H.264 round-trip error and repeated-grayscale integrity."""

    actual = np.asarray(decoded)
    target = np.asarray(expected)
    errors: list[str] = []
    if actual.shape != target.shape:
        errors.append(f"shape 不一致：decoded={actual.shape}, expected={target.shape}")
        return {"errors": errors, "ok": False}
    if actual.dtype != np.uint8:
        errors.append(f"decoded dtype 必须为 uint8，实际为 {actual.dtype}")
    delta = np.abs(actual.astype(np.int16) - target.astype(np.int16))
    channel_delta = np.max(
        np.abs(actual.astype(np.int16) - actual[..., :1].astype(np.int16))
    )
    mae = float(np.mean(delta))
    if mae > max_mae:
        errors.append(f"H.264 round-trip MAE {mae:.4f} 超过 {max_mae:.4f}")
    if int(channel_delta) > max_channel_delta:
        errors.append(
            f"通道最大差 {int(channel_delta)} 超过 {max_channel_delta}"
        )
    return {
        "shape": [int(value) for value in actual.shape],
        "dtype": str(actual.dtype),
        "decoded_min": int(actual.min()),
        "decoded_max": int(actual.max()),
        "roundtrip_mae": mae,
        "roundtrip_max_abs": int(delta.max()),
        "max_channel_delta": int(channel_delta),
        "errors": errors,
        "ok": not errors,
    }
