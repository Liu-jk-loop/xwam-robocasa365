#!/usr/bin/env python3
"""标定全局逆深度编码，生成三任务小缓存，并完成数值/时序/存储审计。"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.robocasa365_contract import load_task_manifest  # noqa: E402
from data.robocasa365_index import episode_video_path, load_episode_records  # noqa: E402
from data.robocasa365_multitask import resolve_task_dataset_directory  # noqa: E402
from project_tools.robocasa365_depth_encoding import (  # noqa: E402
    decoded_frame_audit,
    depth_cache_sidecar_path,
    depth_cache_video_path,
    encode_inverse_metric_depth,
    file_sha256,
    freeze_global_inverse_depth_encoding,
    read_frozen_depth_encoding,
    sample_inverse_metric_depth,
    validate_frozen_encoding,
)
from project_tools.robocasa365_depth_render import (  # noqa: E402
    CAMERA_KEY_TO_NAME,
    create_replay_environment,
    load_replay_episode_model,
    set_replay_episode_state,
)
from project_tools.training_run import collect_git_state, write_json_atomic  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须为对象：{path}")
    return payload


def _explicit_paths(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        task, separator, path = value.partition("=")
        if not separator or not task or not path:
            raise ValueError(f"--task-path 必须为 TaskName=/absolute/path：{value!r}")
        if task in parsed:
            raise ValueError(f"--task-path 重复指定任务：{task}")
        parsed[task] = path
    return parsed


def _select_tasks(args: argparse.Namespace) -> tuple[list[tuple[str, Path]], Path]:
    manifest_path = Path(args.task_manifest).expanduser().resolve()
    manifest = load_task_manifest(manifest_path)
    explicit = _explicit_paths(args.task_path)
    unknown = sorted(set(explicit) - set(manifest.tasks))
    if unknown:
        raise ValueError(f"--task-path 包含 pilot manifest 外任务：{unknown}")
    selected = [
        (
            task_name,
            Path(explicit[task_name]).expanduser().resolve()
            if task_name in explicit
            else resolve_task_dataset_directory(args.dataset_root, task_name),
        )
        for task_name in manifest.tasks
    ]
    return selected, manifest_path


def _episode_paths(root: Path, episode_index: int) -> dict[str, Path]:
    episode_dir = root / "extras" / f"episode_{episode_index:06d}"
    return {
        "states": episode_dir / "states.npz",
        "model": episode_dir / "model.xml.gz",
        "metadata": episode_dir / "ep_meta.json",
    }


def _load_episode_inputs(
    root: Path,
    episode_index: int,
    episode_length: int,
) -> tuple[np.ndarray, str, dict[str, Any], dict[str, Any]]:
    paths = _episode_paths(root, episode_index)
    with np.load(paths["states"], allow_pickle=False) as archive:
        states = np.asarray(archive["states"])
    if states.ndim != 2 or states.shape[0] != episode_length:
        raise ValueError(
            f"episode_{episode_index:06d} states={states.shape} "
            f"与 length={episode_length} 不一致"
        )
    with gzip.open(paths["model"], "rt", encoding="utf-8") as handle:
        model_xml = handle.read()
    ep_meta = _read_json(paths["metadata"])
    identity = {
        name: {"path": str(path), "sha256": file_sha256(path)}
        for name, path in paths.items()
    }
    return states, model_xml, ep_meta, identity


def _frame_indices(length: int, stride: int) -> tuple[int, ...]:
    if length <= 0 or stride <= 0:
        raise ValueError("episode length 和 frame stride 必须为正数")
    indices = list(range(0, length, stride))
    if indices[-1] != length - 1:
        indices.append(length - 1)
    return tuple(indices)


def _render_metric_depth(env: Any, camera_name: str, height: int, width: int) -> np.ndarray:
    from robosuite.utils.camera_utils import get_real_depth_map

    _, normalized_depth = env.sim.render(
        height=height,
        width=width,
        camera_name=camera_name,
        depth=True,
    )
    normalized_depth = np.asarray(normalized_depth)[::-1]
    return np.asarray(get_real_depth_map(env.sim, normalized_depth), dtype=np.float32)


def _calibrate(
    tasks: list[tuple[str, Path]],
    *,
    episodes_per_task: int,
    frame_stride: int,
    pixels_per_frame: int,
    height: int,
    width: int,
    seed: int,
    quantile_low: float,
    quantile_high: float,
    task_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    task_reports: list[dict[str, Any]] = []
    for task_name, dataset_path in tasks:
        root, _, episodes = load_episode_records(dataset_path)
        selected = episodes[:episodes_per_task]
        if len(selected) != episodes_per_task:
            raise ValueError(
                f"{task_name} 仅有 {len(selected)} 个 episode，少于 {episodes_per_task}"
            )
        env = None
        episode_reports: list[dict[str, Any]] = []
        try:
            env, robosuite_module, runtime = create_replay_environment(root)
            for episode in selected:
                states, model_xml, ep_meta, identity = _load_episode_inputs(
                    root, episode.episode_index, episode.length
                )
                load_replay_episode_model(env, robosuite_module, model_xml, ep_meta)
                indices = _frame_indices(episode.length, frame_stride)
                before = len(samples)
                for frame_index in indices:
                    error = set_replay_episode_state(env, states[frame_index])
                    if error > 1e-9:
                        raise RuntimeError(
                            f"{task_name}/episode_{episode.episode_index:06d}/frame_{frame_index} "
                            f"state round-trip error={error}"
                        )
                    for camera_name in CAMERA_KEY_TO_NAME.values():
                        metric_depth = _render_metric_depth(env, camera_name, height, width)
                        samples.append(
                            sample_inverse_metric_depth(
                                metric_depth,
                                pixels_per_frame=pixels_per_frame,
                                rng=rng,
                            )
                        )
                episode_reports.append(
                    {
                        "episode_index": episode.episode_index,
                        "episode_length": episode.length,
                        "frame_indices": list(indices),
                        "sample_arrays": len(samples) - before,
                        "source_identity": identity,
                    }
                )
            task_reports.append(
                {
                    "task_name": task_name,
                    "lerobot_root": str(root),
                    "runtime": runtime,
                    "episodes": episode_reports,
                }
            )
        finally:
            if env is not None:
                env.close()

    calibration = {
        "scope": "atomic_rgbd_pilot3",
        "task_manifest": str(task_manifest_path),
        "task_manifest_sha256": file_sha256(task_manifest_path),
        "tasks": task_reports,
        "episodes_per_task": episodes_per_task,
        "frame_stride": frame_stride,
        "pixels_per_frame_per_camera": pixels_per_frame,
        "camera_keys": list(CAMERA_KEY_TO_NAME),
        "render_shape": [height, width],
        "seed": seed,
    }
    spec = freeze_global_inverse_depth_encoding(
        samples,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
        calibration=calibration,
    )
    report = {
        "sample_arrays": len(samples),
        "sample_count": spec["calibration_sample_count"],
        "inverse_depth_low_per_m": spec["inverse_depth_low_per_m"],
        "inverse_depth_high_per_m": spec["inverse_depth_high_per_m"],
        "tasks": [item[0] for item in tasks],
        "episodes": sum(len(item["episodes"]) for item in task_reports),
    }
    return spec, report


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> Path:
    if path.exists():
        existing = _read_json(path)
        if existing != payload:
            raise RuntimeError(f"不可变 JSON 已存在但内容不同：{path}")
        return path
    return write_json_atomic(path, payload)


def _cache_video_path(
    cache_root: Path,
    task_name: str,
    info: dict[str, Any],
    episode_index: int,
    camera_key: str,
) -> Path:
    return depth_cache_video_path(
        cache_root,
        task_name=task_name,
        episode_index=episode_index,
        camera_key=camera_key,
        chunks_size=int(info.get("chunks_size", 1000)),
    )


def _sidecar_path(video_path: Path) -> Path:
    return depth_cache_sidecar_path(video_path)


def _valid_resume_sidecar(
    video_path: Path,
    *,
    encoding_sha256: str,
    source_identity: dict[str, Any],
    episode_length: int,
) -> dict[str, Any] | None:
    sidecar_path = _sidecar_path(video_path)
    if not video_path.is_file() or not sidecar_path.is_file():
        return None
    sidecar = _read_json(sidecar_path)
    expected = {
        "encoding_sha256": encoding_sha256,
        "source_identity": source_identity,
        "frame_count": episode_length,
    }
    for key, value in expected.items():
        if sidecar.get(key) != value:
            raise RuntimeError(f"缓存 sidecar 合同漂移：{sidecar_path} key={key}")
    actual_sha256 = file_sha256(video_path)
    if sidecar.get("video_sha256") != actual_sha256:
        raise RuntimeError(f"缓存视频 digest 不一致：{video_path}")
    if sidecar.get("ok") is not True:
        raise RuntimeError(f"缓存 sidecar 不是 PASS：{sidecar_path}")
    return sidecar


def _audit_video(
    video_path: Path,
    *,
    source_rgb_video: Path,
    expected_frames: dict[int, np.ndarray],
    episode_length: int,
    fps: float,
    max_roundtrip_mae: float,
    max_channel_delta: int,
) -> dict[str, Any]:
    import imageio.v2 as imageio

    reader = imageio.get_reader(video_path)
    try:
        metadata = reader.get_meta_data()
        frame_count = int(reader.count_frames())
        decoded_fps = float(metadata.get("fps", 0.0))
        probes = []
        for frame_index, expected in expected_frames.items():
            probes.append(
                {
                    "frame_index": frame_index,
                    **decoded_frame_audit(
                        np.asarray(reader.get_data(frame_index)),
                        expected,
                        max_mae=max_roundtrip_mae,
                        max_channel_delta=max_channel_delta,
                    ),
                }
            )
    finally:
        reader.close()
    source_reader = imageio.get_reader(source_rgb_video)
    try:
        source_metadata = source_reader.get_meta_data()
        source_frame_count = int(source_reader.count_frames())
        source_fps = float(source_metadata.get("fps", 0.0))
    finally:
        source_reader.close()
    errors = [
        f"frame_count={frame_count}，expected={episode_length}"
        for _ in [0]
        if frame_count != episode_length
    ]
    if not np.isclose(decoded_fps, fps, atol=1e-6):
        errors.append(f"fps={decoded_fps}，expected={fps}")
    if source_frame_count != episode_length:
        errors.append(
            f"source RGB frame_count={source_frame_count}，expected={episode_length}"
        )
    if not np.isclose(source_fps, fps, atol=1e-6):
        errors.append(f"source RGB fps={source_fps}，expected={fps}")
    errors.extend(
        f"frame {probe['frame_index']}: {message}"
        for probe in probes
        for message in probe["errors"]
    )
    return {
        "codec": "h264",
        "pixel_format": "yuv420p",
        "frame_count": frame_count,
        "fps": decoded_fps,
        "source_rgb_frame_count": source_frame_count,
        "source_rgb_fps": source_fps,
        "probe_frames": probes,
        "errors": errors,
        "ok": not errors,
    }


def _build_episode_cache(
    env: Any,
    robosuite_module: Any,
    *,
    task_name: str,
    root: Path,
    info: dict[str, Any],
    episode_index: int,
    episode_length: int,
    cache_root: Path,
    spec: dict[str, Any],
    height: int,
    width: int,
    fps: float,
    max_roundtrip_mae: float,
    max_channel_delta: int,
) -> dict[str, Any]:
    import imageio.v2 as imageio

    states, model_xml, ep_meta, source_identity = _load_episode_inputs(
        root, episode_index, episode_length
    )
    output_paths = {
        key: _cache_video_path(cache_root, task_name, info, episode_index, key)
        for key in CAMERA_KEY_TO_NAME
    }
    resumed = {
        key: _valid_resume_sidecar(
            path,
            encoding_sha256=str(spec["encoding_sha256"]),
            source_identity=source_identity,
            episode_length=episode_length,
        )
        for key, path in output_paths.items()
    }
    if all(item is not None for item in resumed.values()):
        cameras = list(resumed.values())
        return {
            "episode_index": episode_index,
            "episode_length": episode_length,
            "resumed": True,
            "resumed_camera_count": len(cameras),
            "generation_seconds": sum(
                {
                    float(item["generation_seconds_generation_pass"])
                    for item in cameras
                }
            ),
            "cameras": cameras,
            "ok": True,
        }

    missing_keys = [key for key, item in resumed.items() if item is None]
    for key in missing_keys:
        sidecar_path = _sidecar_path(output_paths[key])
        if sidecar_path.exists() and not output_paths[key].exists():
            raise RuntimeError(f"发现缺失视频的孤立 sidecar，拒绝覆盖：{sidecar_path}")

    load_replay_episode_model(env, robosuite_module, model_xml, ep_meta)
    probe_indices = set(_frame_indices(episode_length, max(1, episode_length // 2)))
    writers: dict[str, Any] = {}
    temporary_paths: dict[str, Path] = {}
    expected_frames: dict[str, dict[int, np.ndarray]] = {
        key: {} for key in missing_keys
    }
    stats = {
        key: {
            "pixels": 0,
            "valid_pixels": 0,
            "invalid_pixels": 0,
            "clipped_low_pixels": 0,
            "clipped_high_pixels": 0,
            "encoded_min": 255,
            "encoded_max": 0,
        }
        for key in missing_keys
    }
    start = time.monotonic()
    try:
        for key in missing_keys:
            output_path = output_paths[key]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                prefix=f".{output_path.stem}.", suffix=".mp4", dir=output_path.parent, delete=False
            )
            handle.close()
            temporary_paths[key] = Path(handle.name)
            writers[key] = imageio.get_writer(
                temporary_paths[key],
                fps=fps,
                codec="libx264",
                pixelformat="yuv420p",
                macro_block_size=None,
                ffmpeg_log_level="error",
            )
        for frame_index in range(episode_length):
            error = set_replay_episode_state(env, states[frame_index])
            if error > 1e-9:
                raise RuntimeError(f"frame {frame_index} state round-trip error={error}")
            for camera_key in missing_keys:
                camera_name = CAMERA_KEY_TO_NAME[camera_key]
                metric_depth = _render_metric_depth(env, camera_name, height, width)
                encoded, frame_stats = encode_inverse_metric_depth(metric_depth, spec)
                writers[camera_key].append_data(encoded)
                if frame_index in probe_indices:
                    expected_frames[camera_key][frame_index] = encoded
                aggregate = stats[camera_key]
                for name in (
                    "pixels",
                    "valid_pixels",
                    "invalid_pixels",
                    "clipped_low_pixels",
                    "clipped_high_pixels",
                ):
                    aggregate[name] += frame_stats[name]
                aggregate["encoded_min"] = min(
                    aggregate["encoded_min"], frame_stats["encoded_min"]
                )
                aggregate["encoded_max"] = max(
                    aggregate["encoded_max"], frame_stats["encoded_max"]
                )
        for writer in writers.values():
            writer.close()
        writers.clear()
        for key, temporary_path in temporary_paths.items():
            os.replace(temporary_path, output_paths[key])
    finally:
        for writer in writers.values():
            writer.close()
        for temporary_path in temporary_paths.values():
            if temporary_path.exists():
                temporary_path.unlink()
    generation_seconds = time.monotonic() - start

    cameras_by_key: dict[str, dict[str, Any]] = {
        key: item for key, item in resumed.items() if item is not None
    }
    for key in missing_keys:
        output_path = output_paths[key]
        source_video = episode_video_path(root, info, episode_index, key)
        audit = _audit_video(
            output_path,
            source_rgb_video=source_video,
            expected_frames=expected_frames[key],
            episode_length=episode_length,
            fps=fps,
            max_roundtrip_mae=max_roundtrip_mae,
            max_channel_delta=max_channel_delta,
        )
        camera_report = {
            "task_name": task_name,
            "episode_index": episode_index,
            "camera_key": key,
            "camera_name": CAMERA_KEY_TO_NAME[key],
            "video_path": str(output_path),
            "video_sha256": file_sha256(output_path),
            "video_bytes": output_path.stat().st_size,
            "source_rgb_video": str(source_video),
            "source_rgb_video_sha256": file_sha256(source_video),
            "source_identity": source_identity,
            "sidecar_path": str(_sidecar_path(output_path)),
            "encoding_sha256": spec["encoding_sha256"],
            "frame_count": episode_length,
            "generation_seconds_generation_pass": generation_seconds,
            "encoding_statistics": stats[key],
            "audit": audit,
            "errors": audit["errors"],
            "ok": audit["ok"],
        }
        _write_immutable_json(_sidecar_path(output_path), camera_report)
        cameras_by_key[key] = camera_report
    cameras = [cameras_by_key[key] for key in CAMERA_KEY_TO_NAME]
    return {
        "episode_index": episode_index,
        "episode_length": episode_length,
        "resumed": any(item is not None for item in resumed.values()),
        "resumed_camera_count": sum(item is not None for item in resumed.values()),
        "generation_seconds": sum(
            {
                float(item["generation_seconds_generation_pass"])
                for item in cameras
            }
        ),
        "cameras": cameras,
        "ok": all(item["ok"] for item in cameras),
    }


def _build_cache(
    tasks: list[tuple[str, Path]],
    *,
    episodes_per_task: int,
    cache_root: Path,
    spec: dict[str, Any],
    height: int,
    width: int,
    fps: float,
    max_roundtrip_mae: float,
    max_channel_delta: int,
) -> list[dict[str, Any]]:
    task_reports: list[dict[str, Any]] = []
    for task_name, dataset_path in tasks:
        root, info, episodes = load_episode_records(dataset_path)
        selected = episodes if episodes_per_task == 0 else episodes[:episodes_per_task]
        if not selected:
            raise ValueError(f"{task_name}没有可生成depth cache的episode")
        env = None
        episode_reports: list[dict[str, Any]] = []
        try:
            env, robosuite_module, runtime = create_replay_environment(root)
            for episode in selected:
                print(
                    f"[RGBD-P2] task={task_name} episode={episode.episode_index} "
                    f"frames={episode.length}",
                    flush=True,
                )
                episode_reports.append(
                    _build_episode_cache(
                        env,
                        robosuite_module,
                        task_name=task_name,
                        root=root,
                        info=info,
                        episode_index=episode.episode_index,
                        episode_length=episode.length,
                        cache_root=cache_root,
                        spec=spec,
                        height=height,
                        width=width,
                        fps=fps,
                        max_roundtrip_mae=max_roundtrip_mae,
                        max_channel_delta=max_channel_delta,
                    )
                )
        finally:
            if env is not None:
                env.close()
        task_reports.append(
            {
                "task_name": task_name,
                "lerobot_root": str(root),
                "runtime": runtime,
                "episodes": episode_reports,
                "ok": bool(episode_reports) and all(item["ok"] for item in episode_reports),
            }
        )
    return task_reports


def _summarize(
    task_reports: list[dict[str, Any]],
    *,
    spec: dict[str, Any],
    cache_root: Path,
    task_manifest_path: Path,
) -> dict[str, Any]:
    episodes = [episode for task in task_reports for episode in task["episodes"]]
    cameras = [camera for episode in episodes for camera in episode["cameras"]]
    total_bytes = sum(int(item["video_bytes"]) for item in cameras)
    total_video_frames = sum(int(item["frame_count"]) for item in cameras)
    generated_seconds = sum(float(item.get("generation_seconds", 0.0)) for item in episodes)
    errors = [
        f"{camera['task_name']}/episode_{camera['episode_index']:06d}/{camera['camera_name']}: {message}"
        for camera in cameras
        for message in camera.get("errors", [])
    ]
    ok = bool(cameras) and not errors and all(task["ok"] for task in task_reports)
    return {
        "schema_version": 1,
        "phase": "RGBD-P2-pilot-cache-and-audit",
        "scope": "atomic_only",
        "git": collect_git_state(REPO_ROOT),
        "task_manifest": str(task_manifest_path),
        "task_manifest_sha256": file_sha256(task_manifest_path),
        "cache_root": str(cache_root),
        "encoding_name": spec["encoding_name"],
        "encoding_sha256": spec["encoding_sha256"],
        "tasks": task_reports,
        "task_count": len(task_reports),
        "episode_count": len(episodes),
        "video_count": len(cameras),
        "total_video_frames": total_video_frames,
        "total_bytes": total_bytes,
        "bytes_per_video_frame": (
            float(total_bytes / total_video_frames) if total_video_frames else None
        ),
        "generation_seconds": generated_seconds,
        "generated_video_frames_per_second": (
            float(total_video_frames / generated_seconds) if generated_seconds else None
        ),
        "resumed_episode_count": sum(item.get("resumed") is True for item in episodes),
        "resumed_camera_count": sum(
            int(item.get("resumed_camera_count", 0)) for item in episodes
        ),
        "camera_contract": list(CAMERA_KEY_TO_NAME),
        "timing_contract": "depth frame i matches RGB frame i and episode state i",
        "errors": errors,
        "ok": ok,
        "result": "pass" if ok else "fail",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--task-manifest",
        default="configs/tasks/robocasa365_rgbd_pilot3.json",
    )
    parser.add_argument("--task-path", action="append", default=[])
    parser.add_argument("--episodes-per-task", type=int, default=3)
    parser.add_argument("--calibration-frame-stride", type=int, default=10)
    parser.add_argument("--calibration-pixels-per-frame", type=int, default=4096)
    parser.add_argument("--quantile-low", type=float, default=0.01)
    parser.add_argument("--quantile-high", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--max-roundtrip-mae", type=float, default=3.0)
    parser.add_argument("--max-channel-delta", type=int, default=3)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument(
        "--encoding-input",
        help="复用已经通过审计的冻结encoding；设置后不重新标定。",
    )
    parser.add_argument(
        "--encoding-output",
        help="新标定encoding的输出；复用模式可省略或指向相同不可变文件。",
    )
    parser.add_argument(
        "--expected-task-count",
        type=int,
        default=0,
        help="非零时要求task manifest恰好包含该数量任务。",
    )
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--audit-output", required=True)
    args = parser.parse_args()

    if args.episodes_per_task < 0:
        raise ValueError("--episodes-per-task 不能为负数；0表示全部")
    tasks, task_manifest_path = _select_tasks(args)
    if args.expected_task_count > 0 and len(tasks) != args.expected_task_count:
        raise ValueError(
            f"任务数与--expected-task-count不一致：{len(tasks)} != {args.expected_task_count}"
        )
    cache_root = Path(args.cache_root).expanduser().resolve()
    manifest_output = Path(args.manifest_output).expanduser().resolve()
    audit_output = Path(args.audit_output).expanduser().resolve()

    if args.encoding_input:
        encoding_input = Path(args.encoding_input).expanduser().resolve()
        print(f"[RGBD-cache] reusing frozen encoding: {encoding_input}", flush=True)
        spec = read_frozen_depth_encoding(encoding_input)
        encoding_output = (
            Path(args.encoding_output).expanduser().resolve()
            if args.encoding_output
            else encoding_input
        )
        _write_immutable_json(encoding_output, spec)
        calibration_report = {
            "mode": "reused_frozen_encoding",
            "encoding_input": str(encoding_input),
            "encoding_sha256": spec["encoding_sha256"],
        }
    else:
        if args.episodes_per_task == 0:
            raise ValueError("新标定模式不允许--episodes-per-task=0")
        if not args.encoding_output:
            raise ValueError("新标定模式必须提供--encoding-output")
        encoding_output = Path(args.encoding_output).expanduser().resolve()
        print("[RGBD-cache] calibrating frozen global inverse depth", flush=True)
        spec, calibration_report = _calibrate(
            tasks,
            episodes_per_task=args.episodes_per_task,
            frame_stride=args.calibration_frame_stride,
            pixels_per_frame=args.calibration_pixels_per_frame,
            height=args.height,
            width=args.width,
            seed=args.seed,
            quantile_low=args.quantile_low,
            quantile_high=args.quantile_high,
            task_manifest_path=task_manifest_path,
        )
        validate_frozen_encoding(spec)
        _write_immutable_json(encoding_output, spec)

    print("[RGBD-cache] building resumable depth cache", flush=True)
    task_reports = _build_cache(
        tasks,
        episodes_per_task=args.episodes_per_task,
        cache_root=cache_root,
        spec=spec,
        height=args.height,
        width=args.width,
        fps=args.fps,
        max_roundtrip_mae=args.max_roundtrip_mae,
        max_channel_delta=args.max_channel_delta,
    )
    summary = _summarize(
        task_reports,
        spec=spec,
        cache_root=cache_root,
        task_manifest_path=task_manifest_path,
    )
    manifest = {
        **summary,
        "phase": "RGBD-cache-manifest",
        "calibration": calibration_report,
    }
    audit = {
        **summary,
        "phase": "RGBD-cache-audit",
        "checks": {
            "frozen_encoding_digest": True,
            "all_three_cameras": all(
                {camera["camera_key"] for camera in episode["cameras"]}
                == set(CAMERA_KEY_TO_NAME)
                for task in task_reports
                for episode in task["episodes"]
            ),
            "frame_count_and_fps": all(
                camera["audit"]["ok"]
                for task in task_reports
                for episode in task["episodes"]
                for camera in episode["cameras"]
            ),
            "source_identity_bound": all(
                camera.get("source_identity")
                and camera.get("source_rgb_video_sha256")
                for task in task_reports
                for episode in task["episodes"]
                for camera in episode["cameras"]
            ),
            "storage_cost_reported": summary["total_bytes"] > 0,
            "resume_sidecars_written": all(
                Path(camera["sidecar_path"]).is_file()
                for task in task_reports
                for episode in task["episodes"]
                for camera in episode["cameras"]
            ),
        },
    }
    write_json_atomic(manifest_output, manifest)
    write_json_atomic(audit_output, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"depth_encoding={encoding_output}")
    print(f"depth_cache_manifest={manifest_output}")
    print(f"depth_cache_audit={audit_output}")
    return 0 if audit["ok"] and all(audit["checks"].values()) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        output: str | None = None
        if "--audit-output" in sys.argv:
            position = sys.argv.index("--audit-output")
            if position + 1 < len(sys.argv):
                output = sys.argv[position + 1]
        failure = {
            "schema_version": 1,
            "phase": "RGBD-cache-audit",
            "scope": "atomic_only",
            "git": collect_git_state(REPO_ROOT),
            "errors": [f"{type(exc).__name__}: {exc}"],
            "ok": False,
            "result": "fail",
        }
        if output:
            write_json_atomic(output, failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True), file=sys.stderr)
        raise SystemExit(1) from exc
