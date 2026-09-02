"""Dependency-light contracts for resumable RoboCasa365 PointMap caches."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from project_tools.robocasa365_depth_encoding import canonical_json_sha256, file_sha256
from project_tools.robocasa365_pointmap import POINTMAP_CONTRACT_NAME, pointmap_contract


POINTMAP_CACHE_SCHEMA_VERSION = 1
POINTMAP_RENDER_POLICY = "use_forced_opaque_depth_for_pointmap_keep_original_rgb"
POINTMAP_CACHE_DTYPE = np.dtype("float16")


class PointMapCacheError(RuntimeError):
    """Raised when a PointMap cache artifact violates the frozen contract."""


def validate_transparent_policy_evidence(
    report: dict[str, Any], *, report_sha256: str
) -> dict[str, Any]:
    """Require the conclusive P0.1 report before any formal P1 generation."""

    task = report.get("task")
    conclusion = task.get("conclusion") if isinstance(task, dict) else None
    checks = {
        "report_pass": report.get("ok") is True and report.get("result") == "pass",
        "atomic_only": report.get("scope") == "atomic_only",
        "close_blender_lid": isinstance(task, dict)
        and task.get("task_name") == "CloseBlenderLid",
        "three_episodes_passed": isinstance(task, dict)
        and int(task.get("episodes_passed", -1)) == 3,
        "target_visible": isinstance(task, dict)
        and int(task.get("target_visible_pixels", 0)) > 0,
        "target_depth_changed": isinstance(task, dict)
        and int(task.get("changed_target_pixels", 0)) > 0,
        "forced_opaque_required": isinstance(conclusion, dict)
        and conclusion.get("status") == "forced_opaque_required",
        "formal_policy": isinstance(conclusion, dict)
        and conclusion.get("formal_cache_policy") == POINTMAP_RENDER_POLICY,
        "report_sha256": isinstance(report_sha256, str) and len(report_sha256) == 64,
    }
    errors = [name for name, passed in checks.items() if not passed]
    if errors:
        raise PointMapCacheError("P0.1透明表面策略证据未通过：" + ", ".join(errors))
    git = report.get("git") if isinstance(report.get("git"), dict) else {}
    return {
        "phase": report.get("phase"),
        "task_name": task["task_name"],
        "episodes_passed": int(task["episodes_passed"]),
        "target_visible_pixels": int(task["target_visible_pixels"]),
        "changed_target_pixels": int(task["changed_target_pixels"]),
        "conclusion": conclusion["status"],
        "formal_cache_policy": conclusion["formal_cache_policy"],
        "report_sha256": report_sha256,
        "git_commit": git.get("commit"),
    }


def pointmap_cache_path(
    cache_root: str | Path,
    *,
    task_name: str,
    episode_index: int,
    camera_key: str,
    chunks_size: int,
) -> Path:
    if chunks_size <= 0:
        raise ValueError("chunks_size 必须为正整数")
    if episode_index < 0:
        raise ValueError("episode_index 不能为负数")
    return (
        Path(cache_root).expanduser().resolve()
        / task_name
        / "pointmaps"
        / f"chunk-{episode_index // chunks_size:03d}"
        / camera_key
        / f"episode_{episode_index:06d}.npy"
    )


def pointmap_sidecar_path(array_path: str | Path) -> Path:
    return Path(array_path).with_suffix(".pointmap.json")


def frozen_pointmap_cache_contract(
    *,
    source_height: int,
    source_width: int,
    target_height: int,
    target_width: int,
    policy_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "cache_schema_version": POINTMAP_CACHE_SCHEMA_VERSION,
        "pointmap": pointmap_contract(
            source_height=source_height,
            source_width=source_width,
            target_height=target_height,
            target_width=target_width,
        ),
        "render_policy": {
            "name": POINTMAP_RENDER_POLICY,
            "geometry_depth": "temporarily_force_all_visible_translucent_geom_and_material_alpha_to_one",
            "rgb": "source_dataset_rgb_is_immutable_and_not_rewritten",
            "policy_source": "CloseBlenderLid P0.1 transparent-surface audit",
            "policy_evidence": policy_evidence,
        },
        "storage": {
            "format": "numpy_npy_v1",
            "compression": "none",
            "dtype": "float16",
            "layout": "TCHW",
            "shape_per_frame": [3, int(target_height), int(target_width)],
            "access": "numpy_mmap",
            "publication": "atomic_replace_after_complete_episode_camera",
        },
    }
    payload["contract_sha256"] = canonical_json_sha256(payload)
    return payload


def expected_array_shape(
    frame_count: int, *, target_height: int, target_width: int
) -> tuple[int, int, int, int]:
    if frame_count <= 0 or target_height <= 0 or target_width <= 0:
        raise ValueError("frame_count/target height/target width 必须为正整数")
    return int(frame_count), 3, int(target_height), int(target_width)


def audit_pointmap_cache_array(
    path: str | Path,
    *,
    expected_shape: tuple[int, int, int, int],
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Audit one uncompressed array frame by frame without loading it into RAM."""

    resolved = Path(path).expanduser().resolve()
    errors: list[str] = []
    minimum = math.inf
    maximum = -math.inf
    finite = True
    try:
        array = np.load(resolved, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        return {
            "path": str(resolved),
            "errors": [f"无法memory-map PointMap：{type(exc).__name__}: {exc}"],
            "ok": False,
        }

    shape = tuple(int(item) for item in array.shape)
    if shape != tuple(int(item) for item in expected_shape):
        errors.append(f"shape={shape}，expected={tuple(expected_shape)}")
    if array.dtype != POINTMAP_CACHE_DTYPE:
        errors.append(f"dtype={array.dtype}，expected=float16")
    if shape == tuple(expected_shape) and array.dtype == POINTMAP_CACHE_DTYPE:
        for frame_index in range(shape[0]):
            frame = np.asarray(array[frame_index])
            if not np.isfinite(frame).all():
                finite = False
                errors.append(f"frame {frame_index} 包含 NaN/Inf")
                break
            minimum = min(minimum, float(np.min(frame)))
            maximum = max(maximum, float(np.max(frame)))
        if finite and (minimum < -1.0 or maximum > 1.0):
            errors.append(f"range=[{minimum}, {maximum}] 超出 [-1,1]")
    actual_sha256 = file_sha256(resolved)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        errors.append("array sha256 与 sidecar 不一致")
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "shape": list(shape),
        "dtype": str(array.dtype),
        "layout": "TCHW",
        "finite": finite,
        "minimum": None if minimum == math.inf else minimum,
        "maximum": None if maximum == -math.inf else maximum,
        "sha256": actual_sha256,
        "errors": errors,
        "ok": not errors,
    }


def validate_resume_pair(
    array_path: str | Path,
    *,
    expected: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Return a valid sidecar or reasons why this exact pair must be rebuilt."""

    array_file = Path(array_path).expanduser().resolve()
    sidecar_file = pointmap_sidecar_path(array_file)
    reasons: list[str] = []
    if not array_file.is_file():
        reasons.append("array_missing")
    if not sidecar_file.is_file():
        reasons.append("sidecar_missing")
    if reasons:
        return None, reasons
    try:
        sidecar = json.loads(sidecar_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"sidecar_unreadable:{type(exc).__name__}"]
    if not isinstance(sidecar, dict):
        return None, ["sidecar_not_object"]
    for key, value in expected.items():
        if sidecar.get(key) != value:
            reasons.append(f"sidecar_mismatch:{key}")
    if sidecar.get("ok") is not True:
        reasons.append("sidecar_not_pass")
    recorded_sha256 = sidecar.get("array_sha256")
    if not isinstance(recorded_sha256, str) or len(recorded_sha256) != 64:
        reasons.append("sidecar_array_sha256_invalid")
    if reasons:
        return None, reasons
    audit = audit_pointmap_cache_array(
        array_file,
        expected_shape=tuple(int(item) for item in expected["shape"]),
        expected_sha256=recorded_sha256,
    )
    if not audit["ok"]:
        return None, [f"array_invalid:{message}" for message in audit["errors"]]
    if sidecar.get("array_audit") != audit:
        return None, ["sidecar_array_audit_mismatch"]
    return sidecar, []


def validate_manifest_artifact(
    record: dict[str, Any],
    *,
    expected_contract_sha256: str,
) -> dict[str, Any]:
    """Reopen one manifest record and prove its sidecar and array still agree."""

    errors: list[str] = []
    array_path = Path(str(record.get("array_path", ""))).expanduser().resolve()
    sidecar_path = pointmap_sidecar_path(array_path)
    if record.get("contract_name") != POINTMAP_CONTRACT_NAME:
        errors.append("contract_name 漂移")
    if record.get("contract_sha256") != expected_contract_sha256:
        errors.append("contract_sha256 漂移")
    if not sidecar_path.is_file():
        errors.append(f"sidecar 缺失：{sidecar_path}")
        sidecar = None
    else:
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            sidecar = None
            errors.append(f"sidecar 无法读取：{exc}")
    if sidecar != record:
        errors.append("manifest记录与sidecar不一致")
    shape_value = record.get("shape", [])
    if not isinstance(shape_value, list) or len(shape_value) != 4:
        errors.append("shape 合同缺失")
        array_audit = {"ok": False, "errors": ["shape 合同缺失"]}
    else:
        array_audit = audit_pointmap_cache_array(
            array_path,
            expected_shape=tuple(int(item) for item in shape_value),
            expected_sha256=str(record.get("array_sha256", "")),
        )
        if not array_audit["ok"]:
            errors.extend(array_audit["errors"])
        if record.get("array_audit") != array_audit:
            errors.append("sidecar array_audit 与复查结果不一致")
    return {
        "array_path": str(array_path),
        "sidecar_path": str(sidecar_path),
        "array_audit": array_audit,
        "errors": errors,
        "ok": not errors,
    }
