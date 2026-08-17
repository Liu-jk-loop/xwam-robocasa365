"""Dependency-light structural preflight for RoboCasa365 depth replay inputs."""

from __future__ import annotations

import gzip
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from data.robocasa365_index import load_episode_records


REQUIRED_CAMERAS = (
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
)
REQUIRED_EPISODE_FILES = ("states.npz", "ep_meta.json", "model.xml.gz")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return payload


def _camera_keys(info: dict[str, Any]) -> list[str]:
    features = info.get("features", {})
    if not isinstance(features, dict):
        return []
    return sorted(
        str(key)
        for key, value in features.items()
        if str(key).startswith("observation.images.")
        and isinstance(value, dict)
    )


def _inspect_episode(
    episode_dir: Path,
    *,
    episode_index: int,
    expected_frames: int,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    states_path = episode_dir / "states.npz"
    ep_meta_path = episode_dir / "ep_meta.json"
    model_path = episode_dir / "model.xml.gz"
    state_shape: list[int] | None = None
    state_dtype: str | None = None
    state_finite: bool | None = None
    npz_keys: list[str] = []
    xml_root: str | None = None
    ep_meta_keys: list[str] = []

    try:
        with np.load(states_path, allow_pickle=False) as archive:
            npz_keys = sorted(str(key) for key in archive.files)
            if "states" not in archive.files:
                raise ValueError(f"缺少官方 states 键，实际键为 {npz_keys}")
            states = archive["states"]
        state_shape = [int(value) for value in states.shape]
        state_dtype = str(states.dtype)
        if states.ndim != 2:
            errors.append(f"states 必须为二维数组，实际 shape={state_shape}")
        elif int(states.shape[0]) != int(expected_frames):
            errors.append(
                f"states 帧数 {states.shape[0]} 与 episodes.jsonl length={expected_frames} 不一致"
            )
        if not np.issubdtype(states.dtype, np.number):
            errors.append(f"states 必须是数值 dtype，实际为 {states.dtype}")
        else:
            state_finite = bool(np.isfinite(states).all())
            if not state_finite:
                errors.append("states 包含 NaN 或 Inf")
        unexpected_keys = sorted(set(npz_keys) - {"states"})
        if unexpected_keys:
            warnings.append(f"states.npz 包含额外键：{unexpected_keys}")
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f"states.npz 无法解析：{type(exc).__name__}: {exc}")

    try:
        ep_meta = _read_json(ep_meta_path)
        ep_meta_keys = sorted(str(key) for key in ep_meta)
        if not ep_meta:
            warnings.append("ep_meta.json 为空对象")
    except ValueError as exc:
        errors.append(str(exc))

    try:
        with gzip.open(model_path, "rb") as handle:
            xml_root = ET.fromstring(handle.read()).tag
        if xml_root != "mujoco":
            errors.append(f"model.xml.gz 根标签应为 mujoco，实际为 {xml_root!r}")
    except (OSError, EOFError, ET.ParseError) as exc:
        errors.append(f"model.xml.gz 无法解析：{type(exc).__name__}: {exc}")

    return {
        "episode_index": int(episode_index),
        "expected_frames": int(expected_frames),
        "episode_dir": str(episode_dir),
        "npz_keys": npz_keys,
        "state_shape": state_shape,
        "state_dtype": state_dtype,
        "state_finite": state_finite,
        "ep_meta_keys": ep_meta_keys,
        "xml_root": xml_root,
        "warnings": warnings,
        "errors": errors,
        "ok": not errors,
    }


def inspect_depth_replay_inputs(
    dataset_path: str | Path,
    *,
    task_name: str,
    episodes_to_decode: int = 3,
) -> dict[str, Any]:
    """Audit official replay extras without importing RoboCasa, MuJoCo, or Torch."""

    if episodes_to_decode < 0:
        raise ValueError("episodes_to_decode 不能为负数；0 表示解码全部 episode")

    root, info, episodes = load_episode_records(dataset_path)
    extras_root = root / "extras"
    errors: list[str] = []
    warnings: list[str] = []
    dataset_meta_path = extras_root / "dataset_meta.json"
    dataset_meta: dict[str, Any] = {}
    env_name: str | None = None

    if not dataset_meta_path.is_file():
        errors.append(f"缺少 dataset_meta.json：{dataset_meta_path}")
    else:
        try:
            dataset_meta = _read_json(dataset_meta_path)
            env_args = dataset_meta.get("env_args")
            if not isinstance(env_args, dict):
                errors.append("dataset_meta.json 缺少对象类型 env_args")
            else:
                raw_env_name = env_args.get("env_name")
                if isinstance(raw_env_name, str) and raw_env_name:
                    env_name = raw_env_name
                else:
                    errors.append("dataset_meta.env_args 缺少非空 env_name")
                if env_name is not None and env_name != task_name:
                    warnings.append(
                        f"dataset_meta env_name={env_name!r} 与任务名 {task_name!r} 不同；"
                        "实际回放时必须使用 dataset_meta 值"
                    )
        except ValueError as exc:
            errors.append(str(exc))

    cameras = _camera_keys(info)
    missing_cameras = sorted(set(REQUIRED_CAMERAS) - set(cameras))
    if missing_cameras:
        errors.append(f"缺少 X-WAM 三路相机：{missing_cameras}")

    missing_episode_files: list[dict[str, Any]] = []
    for episode in episodes:
        episode_dir = extras_root / f"episode_{episode.episode_index:06d}"
        missing = [
            name for name in REQUIRED_EPISODE_FILES if not (episode_dir / name).is_file()
        ]
        if missing:
            missing_episode_files.append(
                {"episode_index": episode.episode_index, "missing": missing}
            )
    if missing_episode_files:
        errors.append(
            f"{len(missing_episode_files)}/{len(episodes)} 个 episode 缺少回放文件"
        )

    candidates = [
        episode
        for episode in episodes
        if not any(
            item["episode_index"] == episode.episode_index
            for item in missing_episode_files
        )
    ]
    selected = candidates if episodes_to_decode == 0 else candidates[:episodes_to_decode]
    inspected = [
        _inspect_episode(
            extras_root / f"episode_{episode.episode_index:06d}",
            episode_index=episode.episode_index,
            expected_frames=episode.length,
        )
        for episode in selected
    ]
    for item in inspected:
        errors.extend(
            f"episode_{item['episode_index']:06d}: {message}"
            for message in item["errors"]
        )
        warnings.extend(
            f"episode_{item['episode_index']:06d}: {message}"
            for message in item["warnings"]
        )

    state_widths = {
        int(item["state_shape"][1])
        for item in inspected
        if item["state_shape"] is not None and len(item["state_shape"]) == 2
    }
    if len(state_widths) > 1:
        errors.append(f"抽查 episode 的 MuJoCo state width 不一致：{sorted(state_widths)}")

    ok = not errors and len(inspected) == len(selected)
    return {
        "schema_version": 1,
        "phase": "RGBD-P0-P1-structural-preflight",
        "task_name": task_name,
        "requested_root": str(Path(dataset_path).expanduser()),
        "lerobot_root": str(root),
        "dataset_meta_path": str(dataset_meta_path),
        "dataset_env_name": env_name,
        "camera_keys": cameras,
        "total_episodes": len(episodes),
        "episodes_with_complete_file_set": len(episodes) - len(missing_episode_files),
        "missing_episode_files": missing_episode_files,
        "episodes_decoded": len(inspected),
        "state_widths": sorted(state_widths),
        "inspected_episodes": inspected,
        "depth_target_contract": {
            "semantic_target": "inverse_depth",
            "xwam_public_storage": "three-channel grayscale H.264 yuv420p MP4",
            "xwam_public_resolution": [256, 256],
            "xwam_public_fps": 20,
            "xwam_loader_range": [-1.0, 1.0],
            "repeat_single_channel_to_rgb": True,
            "robocasa_source": "per-episode model.xml.gz + ep_meta.json + states.npz",
            "training_loader_must_not_import_mujoco": True,
            "pixel_encoding_formula_status": "requires_render_probe_validation",
        },
        "next_gate": (
            "在 RoboCasa 环境逐帧 reset_to(states)，渲染三路 RGB 和 inverse depth，"
            "验证与原 RGB 的像素/时间对齐及 X-WAM 8-bit 编码公式"
        ),
        "warnings": warnings,
        "errors": errors,
        "ok": ok,
        "result": "pass" if ok else "fail",
    }
