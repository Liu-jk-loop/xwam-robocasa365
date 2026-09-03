#!/usr/bin/env python3
"""生成可恢复的RoboCasa365 PointMap缓存，并逐文件完成独立审计。"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from datetime import datetime, timezone
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
from project_tools.robocasa365_depth_encoding import file_sha256  # noqa: E402
from project_tools.robocasa365_depth_render import (  # noqa: E402
    CAMERA_KEY_TO_NAME,
    create_replay_environment,
    load_replay_episode_model,
    set_replay_episode_state,
)
from project_tools.robocasa365_pointmap import (  # noqa: E402
    POINTMAP_CONTRACT_NAME,
    metric_depth_to_camera_xyz,
    normalize_camera_xyz,
    resize_nearest_hwc,
)
from project_tools.robocasa365_pointmap_cache import (  # noqa: E402
    POINTMAP_CACHE_SCHEMA_VERSION,
    POINTMAP_RENDER_POLICY,
    audit_pointmap_cache_array,
    expected_array_shape,
    frozen_pointmap_cache_contract,
    pointmap_cache_path,
    pointmap_sidecar_path,
    validate_manifest_artifact,
    validate_resume_pair,
    validate_transparent_policy_evidence,
)
from project_tools.robocasa365_transparent_pointmap import (  # noqa: E402
    temporarily_force_visible_alpha_opaque,
)
from project_tools.training_run import collect_git_state, write_json_atomic  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON顶层必须为对象：{path}")
    return payload


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": file_sha256(resolved),
    }


def _episode_paths(root: Path, episode_index: int) -> dict[str, Path]:
    episode_dir = root / "extras" / f"episode_{episode_index:06d}"
    return {
        "states": episode_dir / "states.npz",
        "model": episode_dir / "model.xml.gz",
        "metadata": episode_dir / "ep_meta.json",
    }


def _load_episode_inputs(
    root: Path,
    info: dict[str, Any],
    *,
    episode_index: int,
    episode_length: int,
) -> tuple[np.ndarray, str, dict[str, Any], dict[str, Any]]:
    paths = _episode_paths(root, episode_index)
    with np.load(paths["states"], allow_pickle=False) as archive:
        states = np.asarray(archive["states"])
    if states.ndim != 2 or states.shape[0] != episode_length:
        raise ValueError(
            f"episode_{episode_index:06d} states={states.shape} "
            f"与length={episode_length}不一致"
        )
    with gzip.open(paths["model"], "rt", encoding="utf-8") as handle:
        model_xml = handle.read()
    ep_meta = _read_json(paths["metadata"])
    identity = {
        name: _file_identity(path)
        for name, path in paths.items()
    }
    identity["source_rgb_videos"] = {
        camera_key: _file_identity(
            episode_video_path(root, info, episode_index, camera_key)
        )
        for camera_key in CAMERA_KEY_TO_NAME
    }
    return states, model_xml, ep_meta, identity


def _source_identity_for_camera(
    identity: dict[str, Any], camera_key: str
) -> dict[str, Any]:
    return {
        "states": identity["states"],
        "model": identity["model"],
        "metadata": identity["metadata"],
        "source_rgb_video": identity["source_rgb_videos"][camera_key],
    }


def _expected_sidecar(
    *,
    task_name: str,
    episode_index: int,
    episode_length: int,
    camera_key: str,
    camera_name: str,
    array_path: Path,
    contract: dict[str, Any],
    source_identity: dict[str, Any],
    target_height: int,
    target_width: int,
) -> dict[str, Any]:
    return {
        "schema_version": POINTMAP_CACHE_SCHEMA_VERSION,
        "task_name": task_name,
        "scope": "atomic_only",
        "episode_index": int(episode_index),
        "camera_key": camera_key,
        "camera_name": camera_name,
        "contract_name": POINTMAP_CONTRACT_NAME,
        "contract_sha256": contract["contract_sha256"],
        "render_policy": POINTMAP_RENDER_POLICY,
        "source_identity": source_identity,
        "frame_count": int(episode_length),
        "shape": list(
            expected_array_shape(
                episode_length,
                target_height=target_height,
                target_width=target_width,
            )
        ),
        "dtype": "float16",
        "layout": "TCHW",
        "array_path": str(array_path),
        "sidecar_path": str(pointmap_sidecar_path(array_path)),
    }


def _temporary_array_path(array_path: Path) -> Path:
    return array_path.with_name(f".{array_path.stem}.partial.npy")


def _publish_memmap(temporary: Path, final: Path, array: np.memmap) -> None:
    array.flush()
    memory_map = getattr(array, "_mmap", None)
    if memory_map is not None:
        memory_map.close()
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, final)


def _build_episode(
    env: Any,
    robosuite_module: Any,
    *,
    task_name: str,
    root: Path,
    info: dict[str, Any],
    episode_index: int,
    episode_length: int,
    cache_root: Path,
    contract: dict[str, Any],
    source_height: int,
    source_width: int,
    target_height: int,
    target_width: int,
    chunks_size: int,
    git_state: dict[str, Any],
    progress_interval: int,
) -> dict[str, Any]:
    from robosuite.utils.camera_utils import (
        get_camera_intrinsic_matrix,
        get_real_depth_map,
    )

    started = time.monotonic()
    states, model_xml, ep_meta, source_identity = _load_episode_inputs(
        root,
        info,
        episode_index=episode_index,
        episode_length=episode_length,
    )
    output_paths = {
        camera_key: pointmap_cache_path(
            cache_root,
            task_name=task_name,
            episode_index=episode_index,
            camera_key=camera_key,
            chunks_size=chunks_size,
        )
        for camera_key in CAMERA_KEY_TO_NAME
    }
    expected = {
        camera_key: _expected_sidecar(
            task_name=task_name,
            episode_index=episode_index,
            episode_length=episode_length,
            camera_key=camera_key,
            camera_name=camera_name,
            array_path=output_paths[camera_key],
            contract=contract,
            source_identity=_source_identity_for_camera(source_identity, camera_key),
            target_height=target_height,
            target_width=target_width,
        )
        for camera_key, camera_name in CAMERA_KEY_TO_NAME.items()
    }
    resumed: dict[str, dict[str, Any] | None] = {}
    rebuild_reasons: dict[str, list[str]] = {}
    for camera_key, array_path in output_paths.items():
        sidecar, reasons = validate_resume_pair(
            array_path,
            expected=expected[camera_key],
        )
        resumed[camera_key] = sidecar
        rebuild_reasons[camera_key] = reasons
    missing = [key for key, sidecar in resumed.items() if sidecar is None]
    if not missing:
        return {
            "episode_index": int(episode_index),
            "episode_length": int(episode_length),
            "resumed": True,
            "resumed_camera_count": len(CAMERA_KEY_TO_NAME),
            "regenerated_camera_count": 0,
            "rebuild_reasons": rebuild_reasons,
            "generation_seconds": float(time.monotonic() - started),
            "cameras": [resumed[key] for key in CAMERA_KEY_TO_NAME],
            "ok": True,
        }

    load_replay_episode_model(env, robosuite_module, model_xml, ep_meta)
    model_state_width = int(np.asarray(env.sim.get_state().flatten()).size)
    if int(states.shape[1]) != model_state_width:
        raise RuntimeError(
            f"episode_{episode_index:06d} state width={states.shape[1]} "
            f"与MJCF width={model_state_width}不一致"
        )
    intrinsics = {
        key: np.asarray(
            get_camera_intrinsic_matrix(
                env.sim, camera_name, source_height, source_width
            ),
            dtype=np.float64,
        )
        for key, camera_name in CAMERA_KEY_TO_NAME.items()
        if key in missing
    }
    memmaps: dict[str, np.memmap] = {}
    temporary_paths: dict[str, Path] = {}
    valid_pixels = {key: 0 for key in missing}
    total_pixels = {key: 0 for key in missing}
    minima = {key: float("inf") for key in missing}
    maxima = {key: float("-inf") for key in missing}
    state_roundtrip_max_abs = 0.0
    for key in missing:
        output_paths[key].parent.mkdir(parents=True, exist_ok=True)
        temporary = _temporary_array_path(output_paths[key])
        if temporary.exists():
            temporary.unlink()
        temporary_paths[key] = temporary
        memmaps[key] = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float16,
            shape=tuple(expected[key]["shape"]),
        )

    opacity_report: dict[str, Any] = {}
    try:
        with temporarily_force_visible_alpha_opaque(env.sim.model) as opacity_report:
            for frame_index in range(episode_length):
                state_error = set_replay_episode_state(env, states[frame_index])
                state_roundtrip_max_abs = max(state_roundtrip_max_abs, state_error)
                if state_error > 1e-9:
                    raise RuntimeError(
                        f"episode_{episode_index:06d}/frame_{frame_index} "
                        f"state round-trip error={state_error}"
                    )
                for camera_key in missing:
                    _, normalized_depth = env.sim.render(
                        height=source_height,
                        width=source_width,
                        camera_name=CAMERA_KEY_TO_NAME[camera_key],
                        depth=True,
                    )
                    normalized_depth = np.asarray(normalized_depth, dtype=np.float32)[::-1]
                    metric_depth = np.asarray(
                        get_real_depth_map(env.sim, normalized_depth),
                        dtype=np.float32,
                    )
                    xyz, valid = metric_depth_to_camera_xyz(
                        metric_depth,
                        intrinsics[camera_key],
                    )
                    normalized, _ = normalize_camera_xyz(xyz)
                    target = resize_nearest_hwc(
                        normalized,
                        target_height,
                        target_width,
                    )
                    chw = np.moveaxis(target, -1, 0).astype(np.float16)
                    memmaps[camera_key][frame_index] = chw
                    valid_pixels[camera_key] += int(valid.sum())
                    total_pixels[camera_key] += int(valid.size)
                    minima[camera_key] = min(minima[camera_key], float(np.min(chw)))
                    maxima[camera_key] = max(maxima[camera_key], float(np.max(chw)))
                if (
                    progress_interval > 0
                    and (frame_index + 1) % progress_interval == 0
                ) or frame_index + 1 == episode_length:
                    print(
                        f"[PointMap-cache] {task_name}/episode_{episode_index:06d} "
                        f"frame={frame_index + 1}/{episode_length} cameras={len(missing)}",
                        flush=True,
                    )
        elapsed = float(time.monotonic() - started)
        for key in missing:
            array = memmaps.pop(key)
            _publish_memmap(temporary_paths[key], output_paths[key], array)
            array_audit = audit_pointmap_cache_array(
                output_paths[key],
                expected_shape=tuple(expected[key]["shape"]),
            )
            if not array_audit["ok"]:
                raise RuntimeError(
                    f"新PointMap数组审计失败：{output_paths[key]} "
                    f"{array_audit['errors']}"
                )
            sidecar = {
                **expected[key],
                "camera_intrinsics": intrinsics[key].astype(float).tolist(),
                "forced_opaque": opacity_report,
                "state_roundtrip_max_abs": float(state_roundtrip_max_abs),
                "source_valid_pixels": int(valid_pixels[key]),
                "source_total_pixels": int(total_pixels[key]),
                "source_valid_fraction": float(
                    valid_pixels[key] / total_pixels[key]
                ),
                "generated_value_range": [minima[key], maxima[key]],
                "array_sha256": array_audit["sha256"],
                "array_audit": array_audit,
                "generation_seconds_episode_pass": elapsed,
                "generated_at": _utc_now(),
                "git": git_state,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "ok": True,
            }
            write_json_atomic(pointmap_sidecar_path(output_paths[key]), sidecar)
            resumed[key] = sidecar
    finally:
        for array in memmaps.values():
            array.flush()
        memmaps.clear()

    return {
        "episode_index": int(episode_index),
        "episode_length": int(episode_length),
        "resumed": False,
        "resumed_camera_count": len(CAMERA_KEY_TO_NAME) - len(missing),
        "regenerated_camera_count": len(missing),
        "rebuild_reasons": rebuild_reasons,
        "generation_seconds": float(time.monotonic() - started),
        "cameras": [resumed[key] for key in CAMERA_KEY_TO_NAME],
        "ok": True,
    }


def _progress_manifest(
    *,
    cache_root: Path,
    contract: dict[str, Any],
    task_name: str,
    root: Path,
    episode_reports: list[dict[str, Any]],
    expected_episode_count: int,
    git_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": "PointMap-P1",
        "scope": "atomic_only",
        "status": "in_progress",
        "result": "in_progress",
        "ok": False,
        "cache_root": str(cache_root),
        "contract": contract,
        "contract_sha256": contract["contract_sha256"],
        "task_name": task_name,
        "lerobot_root": str(root),
        "expected_episode_count": int(expected_episode_count),
        "completed_episode_count": len(episode_reports),
        "completed_array_count": sum(len(item["cameras"]) for item in episode_reports),
        "episodes": episode_reports,
        "git": git_state,
        "updated_at": _utc_now(),
    }


def _final_audit(
    manifest: dict[str, Any],
    *,
    expected_episode_count: int,
    expected_array_count: int,
) -> dict[str, Any]:
    episodes = manifest["episodes"]
    episode_indices = [int(item["episode_index"]) for item in episodes]
    records = [camera for item in episodes for camera in item["cameras"]]
    artifact_audits = [
        validate_manifest_artifact(
            record,
            expected_contract_sha256=manifest["contract_sha256"],
        )
        for record in records
    ]
    expected_indices = list(range(expected_episode_count))
    camera_pairs = {
        (int(item["episode_index"]), str(camera["camera_key"]))
        for item in episodes
        for camera in item["cameras"]
    }
    expected_pairs = {
        (episode_index, camera_key)
        for episode_index in expected_indices
        for camera_key in CAMERA_KEY_TO_NAME
    }
    checks = {
        "atomic_only": manifest.get("scope") == "atomic_only",
        "task_name_present": isinstance(manifest.get("task_name"), str)
        and bool(manifest.get("task_name")),
        "episode_count": len(episodes) == expected_episode_count,
        "episode_indices": episode_indices == expected_indices,
        "array_count": len(records) == expected_array_count,
        "camera_coverage": camera_pairs == expected_pairs,
        "all_sidecars_pass": all(item.get("ok") is True for item in records),
        "all_arrays_reopened": all(item["ok"] for item in artifact_audits),
        "frame_coverage": sum(int(item["frame_count"]) for item in records)
        == 3 * sum(int(item["episode_length"]) for item in episodes),
        "unique_array_paths": len({item["array_path"] for item in records})
        == len(records),
    }
    errors = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": 1,
        "stage": "PointMap-P1-final-audit",
        "scope": "atomic_only",
        "task_name": manifest["task_name"],
        "cache_root": manifest["cache_root"],
        "manifest_path": manifest["manifest_path"],
        "contract_sha256": manifest["contract_sha256"],
        "render_policy": POINTMAP_RENDER_POLICY,
        "episode_count": len(episodes),
        "array_count": len(records),
        "total_source_frames": sum(
            int(item["episode_length"]) for item in episodes
        ),
        "total_pointmap_frames": sum(int(item["frame_count"]) for item in records),
        "total_cache_bytes": sum(
            int(item["array_audit"]["size_bytes"]) for item in records
        ),
        "checks": checks,
        "artifact_audits": artifact_audits,
        "errors": errors,
        "ok": not errors,
        "result": "pass" if not errors else "fail",
        "audited_at": _utc_now(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--dataset-path",
        help="可选的精确任务日期目录；多任务生成时用于绑定训练manifest。",
    )
    parser.add_argument("--task-manifest", required=True)
    parser.add_argument("--task-name", default="CloseFridge")
    parser.add_argument("--transparent-audit", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--source-height", type=int, default=256)
    parser.add_argument("--source-width", type=int, default=256)
    parser.add_argument("--target-height", type=int, default=256)
    parser.add_argument("--target-width", type=int, default=320)
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=0,
        help="0表示全部episode；仅用于本地/集群小规模验证时可设正数。",
    )
    parser.add_argument("--expected-episode-count", type=int, default=106)
    parser.add_argument("--expected-array-count", type=int, default=318)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--audit-output", required=True)
    args = parser.parse_args()

    if args.episodes_per_task < 0:
        raise ValueError("--episodes-per-task不能为负数")
    manifest_path = Path(args.task_manifest).expanduser().resolve()
    task_manifest = load_task_manifest(manifest_path)
    if args.task_name not in task_manifest.tasks:
        raise ValueError(
            f"PointMap任务{args.task_name}不在atomic清单中：{task_manifest.tasks}"
        )
    dataset_path = (
        Path(args.dataset_path).expanduser().resolve()
        if args.dataset_path
        else resolve_task_dataset_directory(args.dataset_root, args.task_name)
    )
    if args.task_name not in dataset_path.parts:
        raise ValueError(
            f"精确dataset path必须包含任务名{args.task_name}：{dataset_path}"
        )
    root, info, all_episodes = load_episode_records(dataset_path)
    episodes = (
        all_episodes
        if args.episodes_per_task == 0
        else all_episodes[: args.episodes_per_task]
    )
    formal_full = args.episodes_per_task == 0
    if formal_full and len(episodes) != args.expected_episode_count:
        raise ValueError(
            f"{args.task_name} episode数量漂移："
            f"{len(episodes)} != {args.expected_episode_count}"
        )
    expected_array_count = (
        args.expected_array_count
        if formal_full
        else len(episodes) * len(CAMERA_KEY_TO_NAME)
    )
    cache_root = Path(args.cache_root).expanduser().resolve()
    output = Path(args.manifest_output).expanduser().resolve()
    audit_output = Path(args.audit_output).expanduser().resolve()
    progress_output = output.with_name(f".{output.stem}.in-progress.json")
    transparent_audit_path = Path(args.transparent_audit).expanduser().resolve()
    policy_evidence = validate_transparent_policy_evidence(
        _read_json(transparent_audit_path),
        report_sha256=file_sha256(transparent_audit_path),
    )
    policy_evidence["report_path"] = str(transparent_audit_path)
    contract = frozen_pointmap_cache_contract(
        source_height=args.source_height,
        source_width=args.source_width,
        target_height=args.target_height,
        target_width=args.target_width,
        policy_evidence=policy_evidence,
    )
    git_state = collect_git_state(REPO_ROOT)
    cache_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[PointMap-cache] task={args.task_name} episodes={len(episodes)} "
        f"cache={cache_root} policy={POINTMAP_RENDER_POLICY}",
        flush=True,
    )
    env = None
    episode_reports: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        env, robosuite_module, runtime = create_replay_environment(root)
        for position, episode in enumerate(episodes, start=1):
            report = _build_episode(
                env,
                robosuite_module,
                task_name=args.task_name,
                root=root,
                info=info,
                episode_index=episode.episode_index,
                episode_length=episode.length,
                cache_root=cache_root,
                contract=contract,
                source_height=args.source_height,
                source_width=args.source_width,
                target_height=args.target_height,
                target_width=args.target_width,
                chunks_size=int(info.get("chunks_size", 1000)),
                git_state=git_state,
                progress_interval=args.progress_interval,
            )
            episode_reports.append(report)
            write_json_atomic(
                progress_output,
                _progress_manifest(
                    cache_root=cache_root,
                    contract=contract,
                    task_name=args.task_name,
                    root=root,
                    episode_reports=episode_reports,
                    expected_episode_count=len(episodes),
                    git_state=git_state,
                ),
            )
            print(
                f"[PointMap-cache] episode={position}/{len(episodes)} "
                f"resumed={report['resumed_camera_count']} "
                f"regenerated={report['regenerated_camera_count']}",
                flush=True,
            )
    finally:
        if env is not None:
            env.close()

    manifest = {
        "schema_version": 1,
        "stage": "PointMap-P1",
        "scope": "atomic_only",
        "task_name": args.task_name,
        "task_manifest": str(manifest_path),
        "task_manifest_sha256": file_sha256(manifest_path),
        "lerobot_root": str(root),
        "cache_root": str(cache_root),
        "contract": contract,
        "contract_sha256": contract["contract_sha256"],
        "render_policy": POINTMAP_RENDER_POLICY,
        "transparent_policy_evidence": policy_evidence,
        "runtime": runtime,
        "episode_count": len(episode_reports),
        "array_count": sum(len(item["cameras"]) for item in episode_reports),
        "resumed_camera_count": sum(
            int(item["resumed_camera_count"]) for item in episode_reports
        ),
        "regenerated_camera_count": sum(
            int(item["regenerated_camera_count"]) for item in episode_reports
        ),
        "generation_seconds": float(time.monotonic() - started),
        "episodes": episode_reports,
        "git": git_state,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "generated_at": _utc_now(),
        "manifest_path": str(output),
        "errors": [],
        "ok": True,
        "result": "pass",
    }
    write_json_atomic(output, manifest)
    audit = _final_audit(
        manifest,
        expected_episode_count=len(episodes),
        expected_array_count=expected_array_count,
    )
    write_json_atomic(audit_output, audit)
    print(f"pointmap_cache_manifest={output}")
    print(f"pointmap_cache_audit={audit_output}")
    if not audit["ok"]:
        raise RuntimeError(f"PointMap P1最终审计失败：{audit['errors']}")
    print(f"[PASS] {args.task_name} resumable forced-opaque PointMap cache")


if __name__ == "__main__":
    main()
