"""RoboCasa365 per-episode MJCF/state RGB-D replay validation."""

from __future__ import annotations

import copy
import gzip
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from data.robocasa365_contract import resolve_lerobot_root
from data.robocasa365_index import episode_video_path, load_episode_records


CAMERA_KEY_TO_NAME = {
    "observation.images.robot0_agentview_left": "robot0_agentview_left",
    "observation.images.robot0_agentview_right": "robot0_agentview_right",
    "observation.images.robot0_eye_in_hand": "robot0_eye_in_hand",
}


class DepthRenderProbeError(RuntimeError):
    """Raised when one replay/render contract cannot be satisfied."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DepthRenderProbeError(f"无法读取 JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DepthRenderProbeError(f"JSON 顶层必须是对象：{path}")
    return payload


def parse_frame_fractions(value: str) -> tuple[float, ...]:
    """Parse comma-separated [0, 1] frame positions."""

    try:
        fractions = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"无法解析帧位置 {value!r}") from exc
    if not fractions:
        raise ValueError("帧位置不能为空")
    if any(not math.isfinite(item) or item < 0.0 or item > 1.0 for item in fractions):
        raise ValueError(f"帧位置必须是 [0, 1] 内的有限数：{fractions}")
    return tuple(dict.fromkeys(fractions))


def resolve_frame_indices(frame_count: int, fractions: Iterable[float]) -> tuple[int, ...]:
    if frame_count <= 0:
        raise ValueError(f"frame_count 必须为正数，实际为 {frame_count}")
    last = frame_count - 1
    return tuple(dict.fromkeys(int(round(float(fraction) * last)) for fraction in fractions))


def rgb_alignment_metrics(source: np.ndarray, rendered_bottom_up: np.ndarray) -> dict[str, Any]:
    """Compare a dataset RGB frame with MuJoCo output before/after vertical flip."""

    source_array = np.asarray(source)
    rendered_array = np.asarray(rendered_bottom_up)
    if source_array.shape != rendered_array.shape:
        raise ValueError(
            f"RGB shape 不一致：source={source_array.shape}, rendered={rendered_array.shape}"
        )
    if source_array.ndim != 3 or source_array.shape[2] != 3:
        raise ValueError(f"RGB 必须为 HWC 三通道，实际为 {source_array.shape}")

    def metrics(candidate: np.ndarray) -> tuple[float, float, float | None]:
        delta = source_array.astype(np.float32) - candidate.astype(np.float32)
        mae = float(np.mean(np.abs(delta)))
        rmse = float(np.sqrt(np.mean(np.square(delta))))
        psnr = None if rmse == 0.0 else float(20.0 * math.log10(255.0 / rmse))
        return mae, rmse, psnr

    raw_mae, raw_rmse, raw_psnr = metrics(rendered_array)
    flipped = rendered_array[::-1]
    flipped_mae, flipped_rmse, flipped_psnr = metrics(flipped)
    return {
        "vertical_flip_applied": True,
        "rgb_mae": flipped_mae,
        "rgb_rmse": flipped_rmse,
        "rgb_psnr_db": flipped_psnr,
        "unflipped_rgb_mae": raw_mae,
        "unflipped_rgb_rmse": raw_rmse,
        "unflipped_rgb_psnr_db": raw_psnr,
    }


def metric_depth_statistics(normalized_depth: np.ndarray, metric_depth: np.ndarray) -> dict[str, Any]:
    normalized = np.asarray(normalized_depth)
    metric = np.asarray(metric_depth)
    if normalized.shape != metric.shape or normalized.ndim != 2:
        raise ValueError(
            "depth 必须是 shape 相同的二维数组："
            f"normalized={normalized.shape}, metric={metric.shape}"
        )
    finite = np.isfinite(metric)
    positive = metric > 0.0
    valid = finite & positive
    valid_values = metric[valid]
    return {
        "shape": [int(value) for value in metric.shape],
        "normalized_dtype": str(normalized.dtype),
        "normalized_min": float(np.min(normalized)),
        "normalized_max": float(np.max(normalized)),
        "normalized_in_unit_interval": bool(
            np.isfinite(normalized).all()
            and np.min(normalized) >= 0.0
            and np.max(normalized) <= 1.0
        ),
        "metric_dtype": str(metric.dtype),
        "metric_finite_fraction": float(np.mean(finite)),
        "metric_positive_fraction": float(np.mean(positive)),
        "metric_valid_pixels": int(valid.sum()),
        "metric_min_m": float(np.min(valid_values)) if valid_values.size else None,
        "metric_max_m": float(np.max(valid_values)) if valid_values.size else None,
        "metric_mean_m": float(np.mean(valid_values)) if valid_values.size else None,
        "metric_std_m": float(np.std(valid_values)) if valid_values.size else None,
    }


def diagnostic_inverse_depth_rgb(metric_depth: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Make a diagnostic-only inverse-depth preview; never use it as training encoding."""

    depth = np.asarray(metric_depth, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 0.0)
    inverse = np.zeros_like(depth)
    inverse[valid] = 1.0 / depth[valid]
    if not valid.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8), {
            "status": "no_valid_depth",
            "training_encoding": False,
        }
    lower, upper = np.quantile(inverse[valid], [0.01, 0.99])
    if upper <= lower:
        scaled = np.zeros_like(depth)
    else:
        scaled = np.clip((inverse - lower) / (upper - lower), 0.0, 1.0)
    gray = np.rint(scaled * 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2), {
        "status": "diagnostic_per_frame_q01_q99",
        "training_encoding": False,
        "inverse_depth_q01": float(lower),
        "inverse_depth_q99": float(upper),
    }


def _read_video_frames(path: Path, frame_indices: tuple[int, ...]) -> dict[int, np.ndarray]:
    try:
        import imageio.v2 as imageio

        reader = imageio.get_reader(path)
        try:
            return {index: np.asarray(reader.get_data(index)) for index in frame_indices}
        finally:
            reader.close()
    except Exception as exc:  # decoder backends raise several non-portable exception types
        raise DepthRenderProbeError(f"视频帧解码失败 {path}: {type(exc).__name__}: {exc}") from exc


def _set_runtime_egl_device_after_import() -> dict[str, str]:
    current = os.environ.get("MUJOCO_EGL_DEVICE_ID", "0")
    physical = os.environ.setdefault("XWAM_ROBOCASA_IMPORT_EGL_DEVICE", current)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible and current != "0" and current not in visible.split(","):
        raise DepthRenderProbeError(
            "MUJOCO_EGL_DEVICE_ID 必须属于 CUDA_VISIBLE_DEVICES："
            f"egl={current!r}, visible={visible!r}"
        )
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    return {"import_physical_device": physical, "runtime_logical_device": "0"}


def _create_environment(root: Path) -> tuple[Any, Any, dict[str, Any]]:
    import robocasa  # noqa: F401 -- register RoboCasa environments
    import robosuite

    egl = _set_runtime_egl_device_after_import()
    dataset_meta = _read_json(root / "extras" / "dataset_meta.json")
    env_meta = dataset_meta.get("env_args")
    if not isinstance(env_meta, dict):
        raise DepthRenderProbeError("dataset_meta.json 缺少对象类型 env_args")
    env_name = env_meta.get("env_name")
    env_kwargs = copy.deepcopy(env_meta.get("env_kwargs"))
    if not isinstance(env_name, str) or not env_name:
        raise DepthRenderProbeError("dataset_meta.env_args.env_name 必须为非空字符串")
    if not isinstance(env_kwargs, dict):
        raise DepthRenderProbeError("dataset_meta.env_args.env_kwargs 必须为对象")
    env_kwargs.update(
        {
            "env_name": env_name,
            "has_renderer": False,
            "renderer": "mjviewer",
            "has_offscreen_renderer": True,
            "use_camera_obs": False,
            "camera_depths": False,
        }
    )
    environment = robosuite.make(**env_kwargs)
    runtime = {
        "env_name": env_name,
        "robocasa_module": getattr(robocasa, "__file__", None),
        "robosuite_module": getattr(robosuite, "__file__", None),
        "robosuite_version": getattr(robosuite, "__version__", None),
        "egl": egl,
    }
    return environment, robosuite, runtime


def _load_episode_model(env: Any, robosuite_module: Any, model_xml: str, ep_meta: dict[str, Any]) -> None:
    if hasattr(env, "set_attrs_from_ep_meta"):
        env.set_attrs_from_ep_meta(ep_meta)
    elif hasattr(env, "set_ep_meta"):
        env.set_ep_meta(ep_meta)
    else:
        raise DepthRenderProbeError("RoboCasa env 缺少 set_ep_meta/set_attrs_from_ep_meta")

    env.reset()
    try:
        minor_version = int(str(robosuite_module.__version__).split(".")[1])
    except (AttributeError, IndexError, ValueError) as exc:
        raise DepthRenderProbeError(
            f"无法解析 robosuite 版本：{getattr(robosuite_module, '__version__', None)!r}"
        ) from exc
    if minor_version <= 3:
        from robosuite.utils.mjcf_utils import postprocess_model_xml

        edited_xml = postprocess_model_xml(model_xml)
    else:
        edited_xml = env.edit_model_xml(model_xml)
    env.reset_from_xml_string(edited_xml)
    env.sim.reset()


def _set_episode_state(env: Any, state: np.ndarray) -> float:
    expected_width = int(np.asarray(env.sim.get_state().flatten()).size)
    actual_width = int(np.asarray(state).size)
    if actual_width != expected_width:
        raise DepthRenderProbeError(
            "state width 与该 episode MJCF 不匹配："
            f"state={actual_width}, model={expected_width}"
        )
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    if hasattr(env, "update_sites"):
        env.update_sites()
    if hasattr(env, "update_state"):
        env.update_state()
    restored = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
    delta = np.abs(restored - np.asarray(state, dtype=np.float64))
    return float(np.max(delta)) if delta.size else 0.0


def _probe_episode(
    env: Any,
    robosuite_module: Any,
    *,
    root: Path,
    info: dict[str, Any],
    episode_index: int,
    episode_length: int,
    frame_fractions: tuple[float, ...],
    height: int,
    width: int,
    max_rgb_mae: float,
    artifacts_dir: Path,
) -> dict[str, Any]:
    from robosuite.utils.camera_utils import get_real_depth_map
    import imageio.v2 as imageio

    episode_dir = root / "extras" / f"episode_{episode_index:06d}"
    with np.load(episode_dir / "states.npz", allow_pickle=False) as archive:
        states = np.asarray(archive["states"])
    ep_meta = _read_json(episode_dir / "ep_meta.json")
    with gzip.open(episode_dir / "model.xml.gz", "rt", encoding="utf-8") as handle:
        model_xml = handle.read()
    if states.ndim != 2 or states.shape[0] != episode_length:
        raise DepthRenderProbeError(
            f"episode_{episode_index:06d} states shape={states.shape} 与 length={episode_length} 不匹配"
        )

    _load_episode_model(env, robosuite_module, model_xml, ep_meta)
    model_state_width = int(np.asarray(env.sim.get_state().flatten()).size)
    if int(states.shape[1]) != model_state_width:
        raise DepthRenderProbeError(
            f"episode_{episode_index:06d} state width={states.shape[1]} "
            f"与自身 MJCF width={model_state_width} 不匹配"
        )

    frame_indices = resolve_frame_indices(episode_length, frame_fractions)
    source_by_camera = {
        key: _read_video_frames(
            episode_video_path(root, info, episode_index, key), frame_indices
        )
        for key in CAMERA_KEY_TO_NAME
    }
    frame_reports: list[dict[str, Any]] = []
    errors: list[str] = []
    for frame_index in frame_indices:
        try:
            state_roundtrip_max_abs = _set_episode_state(env, states[frame_index])
            frame_errors: list[str] = []
            if state_roundtrip_max_abs > 1e-9:
                frame_errors.append(
                    "state 恢复后回读偏差超过 1e-9："
                    f"{state_roundtrip_max_abs:.6g}"
                )
            camera_reports: list[dict[str, Any]] = []
            for camera_key, camera_name in CAMERA_KEY_TO_NAME.items():
                rendered_rgb, normalized_depth = env.sim.render(
                    height=height,
                    width=width,
                    camera_name=camera_name,
                    depth=True,
                )
                source_rgb = source_by_camera[camera_key][frame_index]
                alignment = rgb_alignment_metrics(source_rgb, rendered_rgb)
                normalized_depth = np.asarray(normalized_depth)[::-1]
                metric_depth = np.asarray(
                    get_real_depth_map(env.sim, normalized_depth), dtype=np.float32
                )
                depth_stats = metric_depth_statistics(normalized_depth, metric_depth)
                preview, preview_contract = diagnostic_inverse_depth_rgb(metric_depth)
                rendered_rgb = np.asarray(rendered_rgb)[::-1]

                camera_errors: list[str] = []
                if alignment["rgb_mae"] > max_rgb_mae:
                    camera_errors.append(
                        f"RGB MAE {alignment['rgb_mae']:.4f} 超过门限 {max_rgb_mae:.4f}"
                    )
                if not depth_stats["normalized_in_unit_interval"]:
                    camera_errors.append("MuJoCo normalized depth 不在 [0,1]")
                if depth_stats["metric_finite_fraction"] != 1.0:
                    camera_errors.append("metric depth 包含 NaN/Inf")
                if depth_stats["metric_positive_fraction"] != 1.0:
                    camera_errors.append("metric depth 包含非正值")
                if depth_stats["metric_std_m"] is None or depth_stats["metric_std_m"] <= 0.0:
                    camera_errors.append("metric depth 退化为常量")

                artifact_base = (
                    artifacts_dir
                    / f"episode_{episode_index:06d}"
                    / f"frame_{frame_index:06d}"
                    / camera_name
                )
                artifact_base.parent.mkdir(parents=True, exist_ok=True)
                arrays_path = artifact_base.with_suffix(".npz")
                preview_path = artifact_base.with_name(f"{artifact_base.name}_comparison.png")
                np.savez_compressed(
                    arrays_path,
                    source_rgb=np.asarray(source_rgb, dtype=np.uint8),
                    rendered_rgb=np.asarray(rendered_rgb, dtype=np.uint8),
                    normalized_depth=np.asarray(normalized_depth, dtype=np.float32),
                    metric_depth_m=metric_depth,
                )
                imageio.imwrite(
                    preview_path,
                    np.concatenate(
                        (
                            np.asarray(source_rgb, dtype=np.uint8),
                            np.asarray(rendered_rgb, dtype=np.uint8),
                            preview,
                        ),
                        axis=1,
                    ),
                )
                camera_reports.append(
                    {
                        "camera_key": camera_key,
                        "camera_name": camera_name,
                        "source_video": str(
                            episode_video_path(root, info, episode_index, camera_key)
                        ),
                        "alignment": alignment,
                        "depth": depth_stats,
                        "inverse_depth_preview": preview_contract,
                        "arrays": str(arrays_path),
                        "comparison_png": str(preview_path),
                        "errors": camera_errors,
                        "ok": not camera_errors,
                    }
                )
                frame_errors.extend(
                    f"{camera_name}: {message}"
                    for message in camera_errors
                )
            errors.extend(f"frame {frame_index}: {message}" for message in frame_errors)
            frame_reports.append(
                {
                    "frame_index": frame_index,
                    "state_roundtrip_max_abs": state_roundtrip_max_abs,
                    "cameras": camera_reports,
                    "errors": frame_errors,
                    "ok": not frame_errors and all(item["ok"] for item in camera_reports),
                }
            )
        except Exception as exc:
            message = f"frame {frame_index}: {type(exc).__name__}: {exc}"
            errors.append(message)
            frame_reports.append({"frame_index": frame_index, "errors": [message], "ok": False})

    return {
        "episode_index": episode_index,
        "episode_length": episode_length,
        "state_shape": [int(value) for value in states.shape],
        "model_state_width": model_state_width,
        "frame_indices": list(frame_indices),
        "frames": frame_reports,
        "errors": errors,
        "ok": not errors,
    }


def probe_task_depth_replay(
    dataset_path: str | Path,
    *,
    task_name: str,
    episodes_per_task: int,
    frame_fractions: tuple[float, ...],
    height: int,
    width: int,
    max_rgb_mae: float,
    artifacts_root: str | Path,
) -> dict[str, Any]:
    """Restore each selected episode's own model/state and render aligned RGB-D."""

    if episodes_per_task < 0:
        raise ValueError("episodes_per_task 不能为负数；0 表示全部")
    if height <= 0 or width <= 0:
        raise ValueError("render height/width 必须为正数")
    if not math.isfinite(max_rgb_mae) or max_rgb_mae < 0.0:
        raise ValueError("max_rgb_mae 必须为非负有限数")

    root, info, episodes = load_episode_records(dataset_path)
    root = resolve_lerobot_root(root)
    selected = episodes if episodes_per_task == 0 else episodes[:episodes_per_task]
    if not selected:
        raise DepthRenderProbeError(f"任务 {task_name} 没有可回放 episode")
    artifacts_dir = Path(artifacts_root) / task_name
    env = None
    episode_reports: list[dict[str, Any]] = []
    errors: list[str] = []
    runtime: dict[str, Any] = {}
    try:
        env, robosuite_module, runtime = _create_environment(root)
        for episode in selected:
            try:
                report = _probe_episode(
                    env,
                    robosuite_module,
                    root=root,
                    info=info,
                    episode_index=episode.episode_index,
                    episode_length=episode.length,
                    frame_fractions=frame_fractions,
                    height=height,
                    width=width,
                    max_rgb_mae=max_rgb_mae,
                    artifacts_dir=artifacts_dir,
                )
            except Exception as exc:
                report = {
                    "episode_index": episode.episode_index,
                    "errors": [f"{type(exc).__name__}: {exc}"],
                    "ok": False,
                }
            episode_reports.append(report)
            errors.extend(
                f"episode_{episode.episode_index:06d}: {message}"
                for message in report.get("errors", [])
            )
    except Exception as exc:
        errors.append(f"环境创建失败：{type(exc).__name__}: {exc}")
    finally:
        if env is not None:
            env.close()

    return {
        "schema_version": 1,
        "phase": "RGBD-P1-per-episode-render-probe",
        "task_name": task_name,
        "lerobot_root": str(root),
        "runtime": runtime,
        "episodes_requested": len(selected),
        "episodes_passed": sum(item.get("ok") is True for item in episode_reports),
        "state_widths": sorted(
            {
                int(item["model_state_width"])
                for item in episode_reports
                if item.get("model_state_width") is not None
            }
        ),
        "state_width_contract": "each state must match its own episode model.xml.gz",
        "render_shape": [height, width],
        "max_rgb_mae": max_rgb_mae,
        "episodes": episode_reports,
        "artifacts_dir": str(artifacts_dir),
        "errors": errors,
        "ok": bool(episode_reports) and not errors,
        "result": "pass" if episode_reports and not errors else "fail",
    }
