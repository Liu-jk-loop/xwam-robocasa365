"""Same-state transparent-object PointMap diagnostics for RoboCasa365.

This module does not change a training cache.  It renders each recorded state
twice: once with the original materials and once after temporarily promoting
visible translucent geom/material alpha to one.  The comparison determines
whether transparent task geometry is missing from the original metric depth
and therefore from the PointMap supervision target.
"""

from __future__ import annotations

import gzip
import json
import math
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from data.robocasa365_contract import resolve_lerobot_root
from data.robocasa365_index import load_episode_records
from project_tools.robocasa365_depth_render import (
    CAMERA_KEY_TO_NAME,
    create_replay_environment,
    diagnostic_inverse_depth_rgb,
    load_replay_episode_model,
    resolve_frame_indices,
    set_replay_episode_state,
)
from project_tools.robocasa365_pointmap import (
    DEPTH_VALID_MAX_M,
    DEPTH_VALID_MIN_M,
    audit_pointmap_frame,
    resize_nearest_hwc,
)


DEFAULT_TARGET_NAME_PATTERNS = (r"blender.*lid", r"lid.*blender")


class TransparentPointMapAuditError(RuntimeError):
    """Raised when the transparent-object diagnostic cannot be completed."""


def alpha_change_mask(rgba: np.ndarray, *, epsilon: float = 1e-6) -> np.ndarray:
    """Select visible translucent rows while preserving alpha-zero hidden geoms."""

    array = np.asarray(rgba)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(f"RGBA 必须为 [N,4]，实际为 {array.shape}")
    if not math.isfinite(epsilon) or epsilon < 0.0 or epsilon >= 0.5:
        raise ValueError("epsilon 必须为 [0, 0.5) 内的有限数")
    alpha = np.asarray(array[:, 3], dtype=np.float64)
    return np.isfinite(alpha) & (alpha > epsilon) & (alpha < 1.0 - epsilon)


@contextmanager
def temporarily_force_visible_alpha_opaque(model: Any) -> Iterator[dict[str, Any]]:
    """Promote visible translucent alpha to one and restore it unconditionally."""

    backups: list[tuple[np.ndarray, np.ndarray]] = []
    attributes: dict[str, Any] = {}
    for attribute in ("geom_rgba", "mat_rgba"):
        if not hasattr(model, attribute):
            attributes[attribute] = {
                "available": False,
                "changed_count": 0,
                "changed_indices": [],
            }
            continue
        target = getattr(model, attribute)
        original = np.asarray(target).copy()
        changed = alpha_change_mask(original)
        backups.append((target, original))
        target[changed, 3] = 1.0
        attributes[attribute] = {
            "available": True,
            "changed_count": int(changed.sum()),
            "changed_indices": np.flatnonzero(changed).astype(int).tolist(),
            "original_alpha": original[changed, 3].astype(float).tolist(),
        }
    try:
        yield {
            "attributes": attributes,
            "changed_count": sum(
                int(item["changed_count"]) for item in attributes.values()
            ),
        }
    finally:
        for target, original in reversed(backups):
            target[...] = original


def _model_object_name(model: Any, kind: str, object_id: int) -> str | None:
    method = getattr(model, f"{kind}_id2name", None)
    if callable(method):
        value = method(int(object_id))
        return None if value is None else str(value)
    try:
        import mujoco

        enum = {
            "geom": mujoco.mjtObj.mjOBJ_GEOM,
            "body": mujoco.mjtObj.mjOBJ_BODY,
            "material": mujoco.mjtObj.mjOBJ_MATERIAL,
        }[kind]
        value = mujoco.mj_id2name(model, enum, int(object_id))
        return None if value is None else str(value)
    except (ImportError, KeyError, TypeError):
        return None


def describe_geom(model: Any, geom_id: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "geom_id": int(geom_id),
        "geom_name": _model_object_name(model, "geom", geom_id),
    }
    if hasattr(model, "geom_bodyid") and 0 <= geom_id < len(model.geom_bodyid):
        body_id = int(model.geom_bodyid[geom_id])
        row.update(
            body_id=body_id,
            body_name=_model_object_name(model, "body", body_id),
        )
    if hasattr(model, "geom_matid") and 0 <= geom_id < len(model.geom_matid):
        material_id = int(model.geom_matid[geom_id])
        row.update(
            material_id=material_id,
            material_name=(
                _model_object_name(model, "material", material_id)
                if material_id >= 0
                else None
            ),
        )
    if hasattr(model, "geom_rgba") and 0 <= geom_id < len(model.geom_rgba):
        row["geom_rgba"] = np.asarray(model.geom_rgba[geom_id]).astype(float).tolist()
    return row


def _as_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            result.extend(_as_strings(item))
        return result
    return []


def _model_name_to_id(model: Any, kind: str, name: str) -> int:
    method = getattr(model, f"{kind}_name2id", None)
    if callable(method):
        try:
            return int(method(str(name)))
        except (KeyError, TypeError, ValueError):
            return -1
    try:
        import mujoco

        enum = {
            "geom": mujoco.mjtObj.mjOBJ_GEOM,
            "body": mujoco.mjtObj.mjOBJ_BODY,
        }[kind]
        return int(mujoco.mj_name2id(model, enum, str(name)))
    except (ImportError, KeyError, TypeError, ValueError):
        return -1


def _descendant_body_ids(model: Any, root_body_id: int) -> set[int]:
    if root_body_id < 0 or not hasattr(model, "body_parentid"):
        return set()
    parents = np.asarray(model.body_parentid, dtype=np.int64).reshape(-1)
    descendants = {int(root_body_id)}
    changed = True
    while changed:
        changed = False
        for body_id, parent_id in enumerate(parents):
            if body_id not in descendants and int(parent_id) in descendants:
                descendants.add(int(body_id))
                changed = True
    return descendants


def entity_geom_ids(model: Any, entity: Any) -> set[int]:
    """Resolve all geoms of one RoboCasa model entity without scene-wide fallback."""

    if entity is None:
        return set()
    ids: set[int] = set()
    for attribute in (
        "visual_geoms",
        "contact_geoms",
        "collision_geoms",
        "all_geoms",
    ):
        for name in _as_strings(getattr(entity, attribute, None)):
            geom_id = _model_name_to_id(model, "geom", name)
            if geom_id >= 0:
                ids.add(geom_id)
    root_names = _as_strings(getattr(entity, "root_body", None))
    for root_name in root_names:
        root_body_id = _model_name_to_id(model, "body", root_name)
        descendants = _descendant_body_ids(model, root_body_id)
        if descendants and hasattr(model, "geom_bodyid"):
            for geom_id, body_id in enumerate(np.asarray(model.geom_bodyid)):
                if int(body_id) in descendants:
                    ids.add(int(geom_id))
    return ids


def find_target_geom_ids(
    model: Any,
    patterns: tuple[str, ...] = DEFAULT_TARGET_NAME_PATTERNS,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Find target geoms from geom/body/material names using strict regex fallback."""

    normalized = tuple(item.strip() for item in patterns if item.strip())
    if not normalized:
        raise ValueError("target geom name patterns 不能为空")
    geom_count = int(getattr(model, "ngeom", 0))
    if geom_count <= 0 and hasattr(model, "geom_rgba"):
        geom_count = len(model.geom_rgba)
    ids: list[int] = []
    rows: list[dict[str, Any]] = []
    for geom_id in range(geom_count):
        row = describe_geom(model, geom_id)
        searchable = " ".join(
            str(row.get(key) or "")
            for key in ("geom_name", "body_name", "material_name")
        )
        if any(
            re.search(pattern, searchable, flags=re.IGNORECASE)
            for pattern in normalized
        ):
            ids.append(geom_id)
            rows.append(row)
    return ids, rows


def resolve_close_blender_lid_geom_ids(
    env: Any,
    patterns: tuple[str, ...] = DEFAULT_TARGET_NAME_PATTERNS,
) -> tuple[list[int], list[dict[str, Any]], str]:
    """Prefer the task's explicit blender_lid entity; use strict name fallback."""

    model = env.sim.model
    blender = getattr(env, "blender", None)
    lid = getattr(blender, "blender_lid", None)
    ids = sorted(entity_geom_ids(model, lid))
    source = "base_env.blender.blender_lid"
    if not ids:
        ids, _ = find_target_geom_ids(model, patterns)
        source = "strict_blender_lid_name_fallback"
    return ids, [describe_geom(model, item) for item in ids], source


def target_alpha_change_count(
    model: Any,
    target_geom_ids: list[int],
    opacity_report: dict[str, Any],
) -> int:
    """Count changed geom/material alpha entries belonging to the target entity."""

    attributes = opacity_report["attributes"]
    changed_geoms = set(attributes["geom_rgba"]["changed_indices"])
    target_ids = set(int(item) for item in target_geom_ids)
    count = len(changed_geoms & target_ids)
    changed_materials = set(attributes["mat_rgba"]["changed_indices"])
    if hasattr(model, "geom_matid"):
        target_materials = {
            int(model.geom_matid[geom_id])
            for geom_id in target_ids
            if 0 <= geom_id < len(model.geom_matid)
            and int(model.geom_matid[geom_id]) >= 0
        }
        count += len(changed_materials & target_materials)
    return count


def transparent_depth_delta(
    original_depth_m: np.ndarray,
    opaque_depth_m: np.ndarray,
    opaque_segmentation: np.ndarray,
    target_geom_ids: list[int],
    *,
    minimum_change_m: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Measure target-only depth changes between original and forced-opaque renders."""

    if not math.isfinite(minimum_change_m) or minimum_change_m <= 0.0:
        raise ValueError("minimum_change_m 必须为正有限数")
    original = np.asarray(original_depth_m, dtype=np.float32)
    opaque = np.asarray(opaque_depth_m, dtype=np.float32)
    segmentation = np.asarray(opaque_segmentation)
    if original.shape != opaque.shape or original.ndim != 2:
        raise ValueError("original/opaque depth 必须是同shape二维数组")
    if segmentation.shape != (*original.shape, 2):
        raise ValueError(
            f"segmentation 必须为 {(*original.shape, 2)}，实际为 {segmentation.shape}"
        )

    target = np.isin(segmentation[..., 1], np.asarray(target_geom_ids, dtype=np.int64))
    original_valid = (
        np.isfinite(original)
        & (original > DEPTH_VALID_MIN_M)
        & (original < DEPTH_VALID_MAX_M)
    )
    opaque_valid = (
        np.isfinite(opaque)
        & (opaque > DEPTH_VALID_MIN_M)
        & (opaque < DEPTH_VALID_MAX_M)
    )
    newly_valid = target & opaque_valid & ~original_valid
    both_valid = target & opaque_valid & original_valid
    closer = both_valid & ((original - opaque) > minimum_change_m)
    farther = both_valid & ((opaque - original) > minimum_change_m)
    changed = newly_valid | closer | farther
    target_visible = target & opaque_valid
    delta = np.abs(original - opaque)
    target_delta = delta[target_visible]
    report = {
        "target_geom_ids": [int(item) for item in target_geom_ids],
        "target_visible_pixels": int(target_visible.sum()),
        "newly_valid_pixels": int(newly_valid.sum()),
        "closer_pixels": int(closer.sum()),
        "farther_pixels": int(farther.sum()),
        "changed_pixels": int(changed.sum()),
        "changed_fraction_of_target": (
            float(changed.sum() / target_visible.sum())
            if target_visible.any()
            else None
        ),
        "target_depth_abs_delta_m": {
            "maximum": float(np.max(target_delta)) if target_delta.size else None,
            "mean": float(np.mean(target_delta)) if target_delta.size else None,
        },
        "minimum_change_m": float(minimum_change_m),
        "target_visible": bool(target_visible.any()),
    }
    arrays = {
        "target_mask": target,
        "target_visible_mask": target_visible,
        "newly_valid_mask": newly_valid,
        "closer_mask": closer,
        "farther_mask": farther,
        "changed_mask": changed,
    }
    return arrays, report


def transparency_conclusion(
    *,
    matched_target_geoms: int,
    alpha_changed_count: int,
    target_visible_pixels: int,
    changed_target_pixels: int,
) -> dict[str, Any]:
    """Return a non-ambiguous policy decision without treating a found issue as a crash."""

    if matched_target_geoms <= 0:
        status = "inconclusive_no_target_geom_match"
    elif alpha_changed_count <= 0:
        status = "inconclusive_no_translucent_alpha"
    elif target_visible_pixels <= 0:
        status = "inconclusive_target_not_visible_in_samples"
    elif changed_target_pixels > 0:
        status = "forced_opaque_required"
    else:
        status = "original_depth_matches_forced_opaque"
    return {
        "status": status,
        "conclusive": status
        in {"forced_opaque_required", "original_depth_matches_forced_opaque"},
        "formal_cache_policy": (
            "use_forced_opaque_depth_for_pointmap_keep_original_rgb"
            if status == "forced_opaque_required"
            else (
                "original_depth_is_acceptable"
                if status == "original_depth_matches_forced_opaque"
                else "undecided"
            )
        ),
    }


def _read_episode_payload(
    root: Path, episode_index: int
) -> tuple[np.ndarray, str, dict[str, Any]]:
    episode_dir = root / "extras" / f"episode_{episode_index:06d}"
    with np.load(episode_dir / "states.npz", allow_pickle=False) as archive:
        states = np.asarray(archive["states"])
    with gzip.open(episode_dir / "model.xml.gz", "rt", encoding="utf-8") as handle:
        model_xml = handle.read()
    ep_meta = json.loads((episode_dir / "ep_meta.json").read_text(encoding="utf-8"))
    if not isinstance(ep_meta, dict):
        raise TransparentPointMapAuditError("ep_meta.json 顶层必须是对象")
    return states, model_xml, ep_meta


def _render_pointmap_variant(
    env: Any,
    *,
    camera_name: str,
    height: int,
    width: int,
    target_height: int,
    target_width: int,
    max_reprojection_error_px: float,
    max_float16_roundtrip_error_m: float,
) -> dict[str, Any]:
    from robosuite.utils.camera_utils import (
        get_camera_intrinsic_matrix,
        get_camera_segmentation,
        get_real_depth_map,
    )

    rgb_bottom_up, normalized_bottom_up = env.sim.render(
        camera_name=camera_name,
        height=height,
        width=width,
        depth=True,
    )
    rgb = np.asarray(rgb_bottom_up)[::-1]
    normalized_depth = np.asarray(normalized_bottom_up, dtype=np.float32)[::-1]
    metric_depth = np.asarray(
        get_real_depth_map(env.sim, normalized_depth), dtype=np.float32
    )
    segmentation = np.asarray(
        get_camera_segmentation(env.sim, camera_name, height, width),
        dtype=np.int32,
    )
    intrinsics = np.asarray(
        get_camera_intrinsic_matrix(env.sim, camera_name, height, width),
        dtype=np.float64,
    )
    pointmap_arrays, pointmap_report = audit_pointmap_frame(
        metric_depth,
        intrinsics,
        target_height=target_height,
        target_width=target_width,
        max_reprojection_error_px=max_reprojection_error_px,
        max_float16_roundtrip_error_m=max_float16_roundtrip_error_m,
    )
    return {
        "rgb": rgb,
        "normalized_depth": normalized_depth,
        "metric_depth_m": metric_depth,
        "segmentation": segmentation,
        "camera_intrinsics": intrinsics,
        "pointmap_arrays": pointmap_arrays,
        "pointmap_report": pointmap_report,
    }


def _pointmap_preview(pointmap_chw: np.ndarray, height: int, width: int) -> np.ndarray:
    normalized = np.moveaxis(np.asarray(pointmap_chw), 0, -1).astype(np.float32)
    preview = np.rint(np.clip((normalized + 1.0) * 127.5, 0.0, 255.0)).astype(np.uint8)
    if preview.shape[:2] != (height, width):
        preview = resize_nearest_hwc(preview, height, width)
    return preview


def _change_overlay(rgb: np.ndarray, arrays: dict[str, np.ndarray]) -> np.ndarray:
    result = np.asarray(rgb, dtype=np.float32).copy()
    tint = np.zeros_like(result)
    tint[arrays["newly_valid_mask"]] = (40, 120, 255)
    tint[arrays["closer_mask"]] = (255, 30, 220)
    tint[arrays["farther_mask"]] = (255, 70, 40)
    changed = arrays["changed_mask"]
    result[changed] = 0.35 * result[changed] + 0.65 * tint[changed]
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


def audit_transparent_task_pointmaps(
    dataset_path: str | Path,
    *,
    task_name: str,
    episodes_per_task: int,
    frame_fractions: tuple[float, ...],
    height: int,
    width: int,
    target_height: int,
    target_width: int,
    target_name_patterns: tuple[str, ...],
    minimum_depth_change_m: float,
    max_reprojection_error_px: float,
    max_float16_roundtrip_error_m: float,
    artifacts_root: str | Path,
) -> dict[str, Any]:
    """Audit original versus forced-opaque PointMaps for one transparent task."""

    if episodes_per_task <= 0:
        raise ValueError("episodes_per_task 必须为正数")
    root, _, episodes = load_episode_records(dataset_path)
    root = resolve_lerobot_root(root)
    selected = episodes[:episodes_per_task]
    if not selected:
        raise TransparentPointMapAuditError(f"任务 {task_name} 没有 episode")
    artifacts_dir = Path(artifacts_root) / task_name
    env = None
    episode_reports: list[dict[str, Any]] = []
    errors: list[str] = []
    runtime: dict[str, Any] = {}
    matched_geom_ids: set[int] = set()
    target_alpha_changed_max = 0
    total_target_visible = 0
    total_target_changed = 0
    try:
        env, robosuite_module, runtime = create_replay_environment(root)
        import imageio.v2 as imageio

        for episode in selected:
            episode_errors: list[str] = []
            frame_reports: list[dict[str, Any]] = []
            episode_target_visible = 0
            episode_target_changed = 0
            try:
                states, model_xml, ep_meta = _read_episode_payload(
                    root, episode.episode_index
                )
                if states.ndim != 2 or states.shape[0] != episode.length:
                    raise TransparentPointMapAuditError(
                        f"states shape={states.shape} 与 episode length={episode.length} 不匹配"
                    )
                load_replay_episode_model(env, robosuite_module, model_xml, ep_meta)
                target_geom_ids, target_geoms, target_geom_source = (
                    resolve_close_blender_lid_geom_ids(
                        env,
                        target_name_patterns,
                    )
                )
                matched_geom_ids.update(target_geom_ids)
                if not target_geom_ids:
                    episode_errors.append("没有匹配到 blender/lid 目标 geom")
                for frame_index in resolve_frame_indices(
                    episode.length, frame_fractions
                ):
                    frame_errors: list[str] = []
                    state_roundtrip_max_abs = set_replay_episode_state(
                        env, states[frame_index]
                    )
                    if state_roundtrip_max_abs > 1e-9:
                        frame_errors.append(
                            "state 恢复后回读偏差超过 1e-9："
                            f"{state_roundtrip_max_abs:.6g}"
                        )
                    originals = {
                        camera_name: _render_pointmap_variant(
                            env,
                            camera_name=camera_name,
                            height=height,
                            width=width,
                            target_height=target_height,
                            target_width=target_width,
                            max_reprojection_error_px=max_reprojection_error_px,
                            max_float16_roundtrip_error_m=max_float16_roundtrip_error_m,
                        )
                        for camera_name in CAMERA_KEY_TO_NAME.values()
                    }
                    with temporarily_force_visible_alpha_opaque(
                        env.sim.model
                    ) as opacity:
                        env.sim.forward()
                        opaque = {
                            camera_name: _render_pointmap_variant(
                                env,
                                camera_name=camera_name,
                                height=height,
                                width=width,
                                target_height=target_height,
                                target_width=target_width,
                                max_reprojection_error_px=max_reprojection_error_px,
                                max_float16_roundtrip_error_m=(
                                    max_float16_roundtrip_error_m
                                ),
                            )
                            for camera_name in CAMERA_KEY_TO_NAME.values()
                        }
                    env.sim.forward()
                    current_target_alpha_changed = target_alpha_change_count(
                        env.sim.model,
                        target_geom_ids,
                        opacity,
                    )
                    target_alpha_changed_max = max(
                        target_alpha_changed_max,
                        current_target_alpha_changed,
                    )
                    if current_target_alpha_changed <= 0:
                        frame_errors.append("目标geom/material没有可提升的半透明alpha")

                    camera_reports: list[dict[str, Any]] = []
                    for camera_name in CAMERA_KEY_TO_NAME.values():
                        original = originals[camera_name]
                        forced = opaque[camera_name]
                        delta_arrays, delta_report = transparent_depth_delta(
                            original["metric_depth_m"],
                            forced["metric_depth_m"],
                            forced["segmentation"],
                            target_geom_ids,
                            minimum_change_m=minimum_depth_change_m,
                        )
                        total_target_visible += delta_report["target_visible_pixels"]
                        total_target_changed += delta_report["changed_pixels"]
                        episode_target_visible += delta_report["target_visible_pixels"]
                        episode_target_changed += delta_report["changed_pixels"]
                        if not original["pointmap_report"]["ok"]:
                            frame_errors.append(
                                f"{camera_name}: original PointMap合同失败"
                            )
                        if not forced["pointmap_report"]["ok"]:
                            frame_errors.append(
                                f"{camera_name}: forced-opaque PointMap合同失败"
                            )

                        artifact_base = (
                            artifacts_dir
                            / f"episode_{episode.episode_index:06d}"
                            / f"frame_{frame_index:06d}"
                            / camera_name
                        )
                        artifact_base.parent.mkdir(parents=True, exist_ok=True)
                        arrays_path = artifact_base.with_suffix(".npz")
                        comparison_path = artifact_base.with_name(
                            f"{artifact_base.name}_transparent_comparison.png"
                        )
                        np.savez_compressed(
                            arrays_path,
                            original_rgb=original["rgb"],
                            original_depth_m=original["metric_depth_m"],
                            opaque_depth_m=forced["metric_depth_m"],
                            opaque_segmentation=forced["segmentation"],
                            original_pointmap_float16=original["pointmap_arrays"][
                                "pointmap_normalized_float16"
                            ],
                            opaque_pointmap_float16=forced["pointmap_arrays"][
                                "pointmap_normalized_float16"
                            ],
                            **delta_arrays,
                        )
                        original_depth_preview, _ = diagnostic_inverse_depth_rgb(
                            original["metric_depth_m"]
                        )
                        opaque_depth_preview, _ = diagnostic_inverse_depth_rgb(
                            forced["metric_depth_m"]
                        )
                        imageio.imwrite(
                            comparison_path,
                            np.concatenate(
                                [
                                    original["rgb"],
                                    original_depth_preview,
                                    opaque_depth_preview,
                                    _change_overlay(original["rgb"], delta_arrays),
                                    _pointmap_preview(
                                        original["pointmap_arrays"][
                                            "pointmap_normalized_float16"
                                        ],
                                        height,
                                        width,
                                    ),
                                    _pointmap_preview(
                                        forced["pointmap_arrays"][
                                            "pointmap_normalized_float16"
                                        ],
                                        height,
                                        width,
                                    ),
                                ],
                                axis=1,
                            ),
                        )
                        camera_reports.append(
                            {
                                "camera_name": camera_name,
                                "original_pointmap": original["pointmap_report"],
                                "forced_opaque_pointmap": forced["pointmap_report"],
                                "transparency": delta_report,
                                "arrays": str(arrays_path),
                                "comparison_png": str(comparison_path),
                                "ok": bool(
                                    original["pointmap_report"]["ok"]
                                    and forced["pointmap_report"]["ok"]
                                ),
                            }
                        )
                    frame_reports.append(
                        {
                            "frame_index": frame_index,
                            "state_roundtrip_max_abs": state_roundtrip_max_abs,
                            "opacity_mutation": opacity,
                            "target_alpha_changed_count": (
                                current_target_alpha_changed
                            ),
                            "cameras": camera_reports,
                            "errors": frame_errors,
                            "ok": not frame_errors,
                        }
                    )
                    episode_errors.extend(frame_errors)
                if episode_target_visible <= 0:
                    episode_errors.append(
                        "该episode抽样帧中目标未出现在forced-opaque segmentation"
                    )
            except Exception as exc:
                episode_errors.append(f"{type(exc).__name__}: {exc}")
                target_geoms = []
                target_geom_source = None
            report = {
                "episode_index": episode.episode_index,
                "target_geoms": target_geoms,
                "target_geom_source": target_geom_source,
                "target_visible_pixels": episode_target_visible,
                "changed_target_pixels": episode_target_changed,
                "frames": frame_reports,
                "errors": episode_errors,
                "ok": not episode_errors,
            }
            episode_reports.append(report)
            errors.extend(
                f"episode_{episode.episode_index:06d}: {message}"
                for message in episode_errors
            )
    except Exception as exc:
        errors.append(f"环境创建失败：{type(exc).__name__}: {exc}")
    finally:
        if env is not None:
            env.close()

    conclusion = transparency_conclusion(
        matched_target_geoms=len(matched_geom_ids),
        alpha_changed_count=target_alpha_changed_max,
        target_visible_pixels=total_target_visible,
        changed_target_pixels=total_target_changed,
    )
    if not conclusion["conclusive"]:
        errors.append(f"透明物体诊断不充分：{conclusion['status']}")
    ok = bool(episode_reports) and not errors
    return {
        "schema_version": 1,
        "phase": "POINTMAP-P0-transparent-surface-audit",
        "scope": "atomic_only",
        "task_name": task_name,
        "lerobot_root": str(root),
        "runtime": runtime,
        "episodes_requested": len(selected),
        "episodes_passed": sum(item["ok"] for item in episode_reports),
        "frame_fractions": list(frame_fractions),
        "target_name_patterns": list(target_name_patterns),
        "matched_target_geom_ids": sorted(matched_geom_ids),
        "target_alpha_changed_count_max": target_alpha_changed_max,
        "target_visible_pixels": total_target_visible,
        "changed_target_pixels": total_target_changed,
        "conclusion": conclusion,
        "episodes": episode_reports,
        "artifacts_dir": str(artifacts_dir),
        "errors": errors,
        "ok": ok,
        "result": "pass" if ok else "fail",
    }
