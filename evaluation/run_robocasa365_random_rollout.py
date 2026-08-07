#!/usr/bin/env python3
"""运行一个不加载 X-WAM 的 RoboCasa365 atomic 随机闭环 smoke。"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


os.environ.setdefault("MUJOCO_GL", "egl")


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.robocasa365_benchmark import (
    BenchmarkContractError,
    aggregate_episode_records,
    flat_action_to_gym_dict,
    load_configured_schema,
    load_random_smoke_config,
    pack_online_cameras,
    pack_online_state,
    sample_random_flat_action,
    tile_camera_views,
    validate_gym_action_space,
)
from project_tools.training_run import collect_git_state, write_json_atomic


DEFAULT_CONFIG = REPO_ROOT / "configs" / "evaluation" / "robocasa365_close_fridge_m4_random_smoke.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    if args.task is not None:
        updated["task"] = args.task
    if args.episodes is not None:
        updated["episodes"] = args.episodes
    if args.seed_start is not None:
        updated["seed_start"] = args.seed_start
    if args.max_steps is not None:
        updated["rollout"]["max_steps"] = args.max_steps
    if args.output_root is not None:
        updated["output_root"] = args.output_root
    if args.no_video:
        updated["video"]["enabled"] = False
    return updated


def _validate_runtime_horizon(task: str, expected_horizon: int) -> int:
    from robocasa.utils.dataset_registry import ATOMIC_TASK_DATASETS

    if task not in ATOMIC_TASK_DATASETS:
        raise BenchmarkContractError(f"运行时 RoboCasa atomic registry 缺少任务：{task}")
    actual = int(ATOMIC_TASK_DATASETS[task]["horizon"])
    if actual != int(expected_horizon):
        raise BenchmarkContractError(
            f"运行时 horizon 与版本化清单不一致：{task} expected={expected_horizon}, actual={actual}"
        )
    return actual


def _create_environment(config: dict[str, Any], seed: int) -> Any:
    import gymnasium as gym
    import robocasa  # noqa: F401 -- import 完成 Gym task 注册

    scene = config["scene"]
    return gym.make(
        f"robocasa/{config['task']}",
        split=scene.get("split"),
        seed=seed,
        enable_render=True,
        obj_instance_split=scene["obj_instance_split"],
        layout_and_style_ids=[(scene["layout_id"], scene["style_id"])],
        randomize_cameras=bool(scene["randomize_cameras"]),
        generative_textures=scene.get("generative_textures"),
        disable_env_checker=True,
    )


def _environment_metadata(env: Any) -> dict[str, Any]:
    unwrapped = env.unwrapped
    episode_meta = unwrapped.get_ep_meta()
    return _json_safe({
        "layout_id": getattr(unwrapped, "layout_id", None),
        "style_id": getattr(unwrapped, "style_id", None),
        "language": episode_meta.get("lang", ""),
        "episode_meta": episode_meta,
    })


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _runtime_versions() -> dict[str, Any]:
    import importlib.metadata
    import robocasa
    import robosuite

    distributions = (
        "robocasa",
        "robosuite",
        "mujoco",
        "numpy",
        "gymnasium",
        "imageio",
        "imageio-ffmpeg",
        "pyzmq",
    )
    versions: dict[str, str | None] = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return {
        "distributions": versions,
        "robocasa_module": getattr(robocasa, "__file__", None),
        "robosuite_module": getattr(robosuite, "__file__", None),
    }


def _run_episode(
    *,
    config: dict[str, Any],
    schema: Any,
    episode_index: int,
    seed: int,
    episode_dir: Path,
) -> dict[str, Any]:
    import imageio.v2 as imageio

    episode_id = f"episode_{episode_index:03d}_seed_{seed:06d}"
    started_at = _utc_now()
    started = time.monotonic()
    env = None
    writer = None
    partial_video = episode_dir / "rollout.partial.mp4"
    final_video = episode_dir / "rollout.mp4"
    episode_dir.mkdir(parents=True, exist_ok=True)
    try:
        env = _create_environment(config, seed)
        observation, reset_info = env.reset(seed=seed)
        environment_metadata = _environment_metadata(env)
        rng = np.random.default_rng(seed)
        video_config = config["video"]
        camera_keys = list(video_config["camera_keys"])
        camera_shape = list(video_config["camera_shape"])
        video_enabled = bool(video_config["enabled"])
        if video_enabled:
            writer = imageio.get_writer(partial_video, fps=int(video_config["fps"]))

        success = bool(reset_info.get("success", False))
        terminated = False
        truncated = False
        steps = 0
        end_reason = "success_on_reset" if success else "max_steps"
        state_min = np.full(schema.state.dimension, np.inf, dtype=np.float64)
        state_max = np.full(schema.state.dimension, -np.inf, dtype=np.float64)
        action_min = np.full(schema.action.dimension, np.inf, dtype=np.float64)
        action_max = np.full(schema.action.dimension, -np.inf, dtype=np.float64)
        base_nonzero_steps = 0
        control_modes: set[float] = set()
        gripper_values: set[float] = set()
        components = {component.name: component for component in schema.action.components}

        initial_state = pack_online_state(observation, schema)
        initial_cameras = pack_online_cameras(observation, camera_keys, camera_shape)
        state_min = np.minimum(state_min, initial_state)
        state_max = np.maximum(state_max, initial_state)
        if video_enabled:
            writer.append_data(tile_camera_views(initial_cameras))

        while steps < int(config["rollout"]["max_steps"]) and not success:
            state = pack_online_state(observation, schema)
            cameras = pack_online_cameras(observation, camera_keys, camera_shape)
            state_min = np.minimum(state_min, state)
            state_max = np.maximum(state_max, state)
            if video_enabled and steps > 0 and steps % int(video_config["stride"]) == 0:
                writer.append_data(tile_camera_views(cameras))

            flat_action = sample_random_flat_action(
                rng,
                schema,
                base_motion_mode=str(config["rollout"]["base_motion_mode"]),
            )
            gym_action = flat_action_to_gym_dict(flat_action, schema)
            validate_gym_action_space(gym_action, env.action_space)
            action_min = np.minimum(action_min, flat_action)
            action_max = np.maximum(action_max, flat_action)
            base = components["base_motion"]
            control = components["control_mode"]
            gripper = components["gripper_close"]
            if np.linalg.norm(flat_action[base.start : base.end]) > 1e-8:
                base_nonzero_steps += 1
            control_modes.update(float(value) for value in flat_action[control.start : control.end])
            gripper_values.update(float(value) for value in flat_action[gripper.start : gripper.end])

            observation, _, terminated, truncated, step_info = env.step(gym_action)
            steps += 1
            success = bool(step_info.get("success", False))
            if success and bool(config["rollout"]["stop_on_success"]):
                end_reason = "success"
                break
            if terminated:
                end_reason = "terminated"
                break
            if truncated:
                end_reason = "truncated"
                break

        if video_enabled:
            final_cameras = pack_online_cameras(observation, camera_keys, camera_shape)
            writer.append_data(tile_camera_views(final_cameras))
            writer.close()
            writer = None
            os.replace(partial_video, final_video)

        record = {
            "schema_version": 1,
            "episode_id": episode_id,
            "result": "pass",
            "task": config["task"],
            "scope": "atomic_only",
            "policy": config["policy"],
            "seed": seed,
            "official_horizon": config["official_horizon"],
            "smoke_max_steps": config["rollout"]["max_steps"],
            "steps": steps,
            "success": success,
            "end_reason": end_reason,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "started_at_utc": started_at,
            "duration_seconds": time.monotonic() - started,
            "environment": environment_metadata,
            "observation_diagnostics": {
                "state_dimension": schema.state.dimension,
                "state_min": state_min.tolist(),
                "state_max": state_max.tolist(),
                "camera_keys": camera_keys,
                "camera_shape": camera_shape,
                "depth_mode": config["rollout"]["depth_mode"],
            },
            "action_diagnostics": {
                "action_dimension": schema.action.dimension,
                "action_min": action_min.tolist() if steps else None,
                "action_max": action_max.tolist() if steps else None,
                "base_motion_mode": config["rollout"]["base_motion_mode"],
                "base_nonzero_steps": base_nonzero_steps,
                "control_modes": sorted(control_modes),
                "gripper_values": sorted(gripper_values),
            },
            "video": {
                "enabled": video_enabled,
                "path": str(final_video) if video_enabled else None,
                "stride": video_config["stride"],
                "fps": video_config["fps"],
            },
        }
        write_json_atomic(episode_dir / "episode.json", record)
        return record
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="M4.1 random smoke JSON 配置。")
    parser.add_argument("--task", help="覆盖 Atomic-Seen task。")
    parser.add_argument("--episodes", type=int, help="覆盖 episode 数。")
    parser.add_argument("--seed-start", type=int, help="覆盖首个 seed。")
    parser.add_argument("--max-steps", type=int, help="覆盖 smoke step 上限；不能超过官方 horizon。")
    parser.add_argument("--output-root", help="覆盖输出根目录。")
    parser.add_argument("--no-video", action="store_true", help="关闭视频，仅用于诊断写盘问题。")
    args = parser.parse_args()

    run_id = _run_id()
    try:
        raw_config = json.loads(Path(args.config).expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"无法读取 M4.1 配置：{exc}", file=sys.stderr)
        return 2
    requested_config = _apply_overrides(raw_config, args)
    output_root = Path(
        args.output_root or requested_config.get("output_root", "eval_results/robocasa365_m4_random_smoke")
    ).expanduser().resolve()
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    started_at = _utc_now()
    started = time.monotonic()
    try:
        overridden_path = run_dir / "requested_config.json"
        write_json_atomic(overridden_path, requested_config)
        config = load_random_smoke_config(overridden_path, REPO_ROOT)
        schema = load_configured_schema(config)
        runtime_horizon = _validate_runtime_horizon(config["task"], config["official_horizon"])
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "started_at_utc": started_at,
            "git": collect_git_state(REPO_ROOT),
            "python_executable": sys.executable,
            "config_path": str(Path(args.config).expanduser().resolve()),
            "resolved_config": config,
            "runtime_horizon": runtime_horizon,
            "simulator_runtime": _runtime_versions(),
            "mujoco_gl": os.environ.get("MUJOCO_GL", "<unset>"),
        }
        write_json_atomic(run_dir / "metadata.json", metadata)

        for episode_index in range(int(config["episodes"])):
            seed = int(config["seed_start"]) + episode_index
            episode_id = f"episode_{episode_index:03d}_seed_{seed:06d}"
            episode_dir = run_dir / config["task"] / episode_id
            try:
                record = _run_episode(
                    config=config,
                    schema=schema,
                    episode_index=episode_index,
                    seed=seed,
                    episode_dir=episode_dir,
                )
            except Exception as exc:
                record = {
                    "schema_version": 1,
                    "episode_id": episode_id,
                    "result": "fail",
                    "task": config["task"],
                    "scope": "atomic_only",
                    "seed": seed,
                    "steps": 0,
                    "success": False,
                    "end_reason": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                write_json_atomic(episode_dir / "episode.json", record)
            records.append(record)
            if record["result"] == "fail":
                break

        aggregate = aggregate_episode_records(records, expected_episodes=int(config["episodes"]))
        result = (
            "pass"
            if aggregate["episodes_failed"] == 0 and aggregate["episodes_missing"] == 0
            else "fail"
        )
        summary = {
            "schema_version": 1,
            "run_id": run_id,
            "result": result,
            "task": config["task"],
            "scope": "atomic_only",
            "policy": config["policy"],
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "duration_seconds": time.monotonic() - started,
            "aggregate": aggregate,
            "episode_records": [str(run_dir / config["task"] / record["episode_id"] / "episode.json") for record in records],
        }
        write_json_atomic(run_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if result == "pass" else 1
    except Exception as exc:
        summary = {
            "schema_version": 1,
            "run_id": run_id,
            "result": "fail",
            "scope": "atomic_only",
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "duration_seconds": time.monotonic() - started,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        write_json_atomic(run_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
