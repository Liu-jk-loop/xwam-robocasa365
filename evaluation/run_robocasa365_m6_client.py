#!/usr/bin/env python3
"""Run one lean FastWAM-compatible M6 client task queue."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.robocasa365_benchmark import (  # noqa: E402
    BenchmarkContractError,
    flat_action_to_gym_dict,
    load_configured_schema,
    pack_online_cameras,
    pack_online_state,
    tile_camera_views,
    validate_gym_action_space,
)
from evaluation.robocasa365_m6_topology import (  # noqa: E402
    load_m6_evaluation_topology,
)
from evaluation.robocasa365_protocol import make_request  # noqa: E402
from evaluation.run_robocasa365_policy_rollout import _request_policy  # noqa: E402
from project_tools.training_run import write_json_atomic  # noqa: E402


DEFAULT_TOPOLOGY = (
    REPO_ROOT
    / "configs"
    / "evaluation"
    / "robocasa365_m6_atomic18_8server_16client.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BenchmarkContractError(f"JSON 顶层必须为对象：{path}")
    return payload


def _create_environment(task: str, split: str) -> Any:
    import gymnasium as gym
    import robocasa  # noqa: F401 -- registers gym environments

    # The legacy robosuite import gate compares MUJOCO_EGL_DEVICE_ID with the
    # physical ids listed in CUDA_VISIBLE_DEVICES. The actual EGL context in the
    # validated RoboCasa EDF exposes only one process-local device, index 0.
    # Keep the physical id through the import above, then switch to local 0
    # before gym.make() creates the offscreen context.
    current_egl_device = os.environ.get("MUJOCO_EGL_DEVICE_ID", "0")
    physical_egl_device = os.environ.setdefault(
        "XWAM_ROBOCASA_IMPORT_EGL_DEVICE", current_egl_device
    )
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible_devices and current_egl_device != "0":
        visible_ids = visible_devices.split(",")
        if current_egl_device not in visible_ids:
            raise BenchmarkContractError(
                "MUJOCO_EGL_DEVICE_ID必须属于CUDA_VISIBLE_DEVICES："
                f"egl={current_egl_device!r} visible={visible_ids!r}"
            )
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    print(
        "[EGL] RoboCasa import使用物理设备"
        f" {physical_egl_device}；runtime context切换到逻辑设备0",
        flush=True,
    )

    # Match the validated FastWAM formal evaluation: target split, seeded reset,
    # and no fixed M4 smoke layout/style.
    return gym.make(
        f"robocasa/{task}",
        split=split,
    )


def _open_policy_socket(frontend_port: int) -> tuple[Any, Any, Any]:
    import zmq

    context = zmq.Context()
    # The broker frontend is ROUTER and expects [identity, payload]. DEALER
    # produces exactly those two frames; REQ inserts an empty delimiter and the
    # broker correctly rejects the resulting three-frame message.
    socket = context.socket(zmq.DEALER)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.IDENTITY, f"xwam-m6-client-{os.getpid()}".encode())
    socket.connect(f"tcp://127.0.0.1:{frontend_port}")
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    return context, socket, poller


def _episode_prompt(observation: dict[str, Any], env: Any) -> tuple[str, str | None]:
    prompt = str(observation.get("annotation.human.task_description", "")).strip()
    if not prompt:
        raise BenchmarkContractError("observation 缺少 annotation.human.task_description")
    meta_prompt = str(env.unwrapped.get_ep_meta().get("lang", "")).strip() or None
    if meta_prompt is not None and meta_prompt != prompt:
        print(
            "[WARN] observation/meta prompts differ: "
            f"observation={prompt!r} meta={meta_prompt!r}",
            flush=True,
        )
    return prompt, meta_prompt


def _open_video(path: Path, fps: int, initial_cameras: np.ndarray) -> Any:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.stem}.partial.mp4")
    partial.unlink(missing_ok=True)
    writer = imageio.get_writer(partial, fps=fps, codec="libx264", quality=8)
    writer.append_data(tile_camera_views(initial_cameras))
    return writer


def _finish_video(writer: Any, path: Path) -> None:
    writer.close()
    partial = path.with_name(f".{path.stem}.partial.mp4")
    if not partial.is_file() or partial.stat().st_size <= 0:
        raise BenchmarkContractError(f"视频编码没有生成非空MP4：{partial}")
    os.replace(partial, path)


def _summarize_base_diagnostics(
    base_commands: list[np.ndarray],
    control_modes: list[float],
    base_position_deltas: list[np.ndarray],
) -> dict[str, Any]:
    """Return compact physical-unit diagnostics without retaining step traces."""

    if not (
        len(base_commands) == len(control_modes) == len(base_position_deltas)
    ):
        raise BenchmarkContractError("base诊断序列长度不一致")
    if not base_commands:
        return {
            "steps": 0,
            "base_command_nonzero_steps": 0,
            "base_command_nonzero_fraction": 0.0,
            "base_command_rms_per_dim": [],
            "base_command_min_per_dim": [],
            "base_command_max_per_dim": [],
            "control_mode_counts": {},
            "base_position_delta_nonzero_steps": 0,
            "base_position_delta_nonzero_fraction": 0.0,
            "base_position_delta_rms_per_dim": [],
            "base_position_total_displacement": 0.0,
            "commanded_but_stationary_steps": 0,
        }

    commands = np.asarray(base_commands, dtype=np.float32)
    modes = np.asarray(control_modes, dtype=np.float32)
    deltas = np.asarray(base_position_deltas, dtype=np.float32)
    if commands.ndim != 2 or commands.shape[1] != 4:
        raise BenchmarkContractError(f"base command诊断应为[N,4]，实际={commands.shape}")
    if modes.shape != (commands.shape[0],):
        raise BenchmarkContractError(f"control mode诊断应为[N]，实际={modes.shape}")
    if deltas.ndim != 2 or deltas.shape != (commands.shape[0], 3):
        raise BenchmarkContractError(f"base position delta诊断应为[N,3]，实际={deltas.shape}")
    if not (
        np.isfinite(commands).all()
        and np.isfinite(modes).all()
        and np.isfinite(deltas).all()
    ):
        raise BenchmarkContractError("base诊断包含NaN/Inf")

    command_norms = np.linalg.norm(commands, axis=1)
    delta_norms = np.linalg.norm(deltas, axis=1)
    command_nonzero = command_norms > 1e-6
    delta_nonzero = delta_norms > 1e-8
    mode_counts = {
        "-1": int(np.count_nonzero(modes == -1.0)),
        "+1": int(np.count_nonzero(modes == 1.0)),
        "other": int(np.count_nonzero((modes != -1.0) & (modes != 1.0))),
    }
    return {
        "steps": int(commands.shape[0]),
        "base_command_nonzero_steps": int(np.count_nonzero(command_nonzero)),
        "base_command_nonzero_fraction": float(command_nonzero.mean()),
        "base_command_rms_per_dim": np.sqrt(np.mean(commands**2, axis=0)).tolist(),
        "base_command_min_per_dim": commands.min(axis=0).tolist(),
        "base_command_max_per_dim": commands.max(axis=0).tolist(),
        "control_mode_counts": mode_counts,
        "base_position_delta_nonzero_steps": int(np.count_nonzero(delta_nonzero)),
        "base_position_delta_nonzero_fraction": float(delta_nonzero.mean()),
        "base_position_delta_rms_per_dim": np.sqrt(
            np.mean(deltas**2, axis=0)
        ).tolist(),
        "base_position_total_displacement": float(
            np.linalg.norm(deltas.sum(axis=0))
        ),
        "commanded_but_stationary_steps": int(
            np.count_nonzero(command_nonzero & ~delta_nonzero)
        ),
    }


def _run_episode(
    *,
    env: Any,
    socket: Any,
    poller: Any,
    topology: dict[str, Any],
    schema: Any,
    task: str,
    episode_index: int,
    seed: int,
    video_path: Path,
) -> dict[str, Any]:
    observation, _ = env.reset(seed=seed)
    prompt, meta_prompt = _episode_prompt(observation, env)
    camera_keys = list(topology["video"]["camera_keys"])
    camera_shape = list(topology["video"]["camera_shape"])
    cameras = pack_online_cameras(observation, camera_keys, camera_shape)
    writer = _open_video(video_path, int(topology["video"]["fps"]), cameras)
    started = time.monotonic()
    steps = 0
    replans = 0
    inference_seconds = 0.0
    clipped_actions = 0
    clipped_scalars = 0
    max_abs_raw_action = 0.0
    action_components = {
        component.name: component for component in schema.action.components
    }
    state_components = {
        component.name: component for component in schema.state.components
    }
    base_action_component = action_components["base_motion"]
    control_mode_component = action_components["control_mode"]
    base_state_component = state_components["base_position"]
    base_commands: list[np.ndarray] = []
    control_modes: list[float] = []
    base_position_deltas: list[np.ndarray] = []
    success = False
    terminated = False
    truncated = False
    end_reason = "running"
    try:
        while steps < int(topology["max_steps_per_episode"]):
            state = pack_online_state(observation, schema)
            cameras = pack_online_cameras(observation, camera_keys, camera_shape)
            request = make_request(
                request_id=f"{task}-seed{seed}-step{steps:06d}",
                task=task,
                episode_id=f"episode_{episode_index:03d}_seed{seed}",
                seed=seed,
                step_id=steps,
                prompt=prompt,
                video=cameras,
                state=state,
                cfg=float(topology["cfg"]),
            )
            response = _request_policy(
                socket,
                poller,
                request,
                float(topology["request_timeout_seconds"]),
            )
            actions = np.asarray(response["actions"], dtype=np.float32)
            if actions.shape != (32, 12):
                raise BenchmarkContractError(
                    f"X-WAM必须返回[32,12] action，实际={actions.shape}"
                )
            replans += 1
            inference_seconds += float(response["inference_seconds"])
            execute_count = min(
                int(topology["replan_steps"]),
                int(topology["max_steps_per_episode"]) - steps,
            )
            print(
                f"[REPLAN] task={task} seed={seed} step={steps} "
                f"infer={response['inference_seconds']:.3f}s",
                flush=True,
            )
            for raw_action in actions[:execute_count]:
                base_position_before = pack_online_state(observation, schema)[
                    base_state_component.start : base_state_component.end
                ].copy()
                outside = (raw_action < -1.0) | (raw_action > 1.0)
                outside_count = int(np.count_nonzero(outside))
                if outside_count:
                    clipped_actions += 1
                    clipped_scalars += outside_count
                max_abs_raw_action = max(
                    max_abs_raw_action, float(np.max(np.abs(raw_action)))
                )
                flat_action = np.clip(raw_action, -1.0, 1.0).astype(np.float32)
                gym_action = flat_action_to_gym_dict(flat_action, schema)
                validate_gym_action_space(gym_action, env.action_space)
                base_commands.append(
                    flat_action[
                        base_action_component.start : base_action_component.end
                    ].copy()
                )
                control_modes.append(
                    float(flat_action[control_mode_component.start])
                )
                observation, reward, terminated, truncated, info = env.step(gym_action)
                base_position_after = pack_online_state(observation, schema)[
                    base_state_component.start : base_state_component.end
                ]
                base_position_deltas.append(
                    np.asarray(
                        base_position_after - base_position_before,
                        dtype=np.float32,
                    )
                )
                steps += 1
                cameras = pack_online_cameras(observation, camera_keys, camera_shape)
                writer.append_data(tile_camera_views(cameras))
                success = bool(info.get("success", False) or reward > 0)
                if success:
                    end_reason = "success"
                elif terminated:
                    end_reason = "terminated"
                elif truncated:
                    end_reason = "truncated"
                elif steps >= int(topology["max_steps_per_episode"]):
                    end_reason = "max_steps"
                if end_reason != "running":
                    break
            if end_reason != "running":
                break
        _finish_video(writer, video_path)
        writer = None
    finally:
        if writer is not None:
            writer.close()
            video_path.with_name(f".{video_path.stem}.partial.mp4").unlink(
                missing_ok=True
            )
    return {
        "episode_idx": episode_index,
        "seed": seed,
        "success": success,
        "steps": steps,
        "replans": replans,
        "max_steps": int(topology["max_steps_per_episode"]),
        "replan_steps": int(topology["replan_steps"]),
        "end_reason": end_reason,
        "terminated": terminated,
        "truncated": truncated,
        "instruction": prompt,
        "meta_instruction": meta_prompt,
        "inference_time_s": inference_seconds,
        "mean_inference_time_s": inference_seconds / replans if replans else None,
        "clipped_actions": clipped_actions,
        "clipped_action_scalars": clipped_scalars,
        "max_abs_raw_action": max_abs_raw_action,
        "base_action_diagnostics": _summarize_base_diagnostics(
            base_commands,
            control_modes,
            base_position_deltas,
        ),
        "elapsed_seconds": time.monotonic() - started,
        "video": str(video_path),
    }


def _load_completed_episodes(
    path: Path,
    *,
    task: str,
    topology: dict[str, Any],
    expected_episodes: int,
    comparison_group: str | None = None,
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    payload = _read_json(path)
    expected = {
        "task": task,
        "split": topology["scene"]["split"],
        "seed_start": topology["seed_start"],
        "episodes_expected": expected_episodes,
        "max_steps": topology["max_steps_per_episode"],
        "replan_steps": topology["replan_steps"],
        "model_seed": topology["model_seed"],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise BenchmarkContractError(
                f"已有{task} result与本次配置不一致：{key} "
                f"expected={value!r} actual={payload.get(key)!r}"
            )
    if comparison_group is not None and payload.get("comparison_group") != comparison_group:
        raise BenchmarkContractError(
            f"已有{task} result comparison_group漂移："
            f"expected={comparison_group!r} actual={payload.get('comparison_group')!r}"
        )
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise BenchmarkContractError(f"已有{task} result缺少episodes")
    expected_seeds = [int(topology["seed_start"]) + i for i in range(len(episodes))]
    actual_seeds = [int(row["seed"]) for row in episodes]
    if actual_seeds != expected_seeds or len(episodes) > expected_episodes:
        raise BenchmarkContractError(f"已有{task} episode seed不是合法连续前缀")
    return [dict(row) for row in episodes]


def _write_task_result(
    *,
    path: Path,
    topology: dict[str, Any],
    client: dict[str, Any],
    task_entry: dict[str, Any],
    episodes: list[dict[str, Any]],
    expected_episodes: int,
) -> None:
    successes = sum(bool(row["success"]) for row in episodes)
    total_inference = sum(float(row["inference_time_s"]) for row in episodes)
    total_replans = sum(int(row["replans"]) for row in episodes)
    complete = len(episodes) == expected_episodes
    write_json_atomic(
        path,
        {
            "schema_version": 1,
            "result": "pass" if complete else "running",
            "task": task_entry["name"],
            "client_id": client["client_id"],
            "server_id": client["server_id"],
            "comparison_group": client.get("comparison_group"),
            "split": topology["scene"]["split"],
            "seed_start": topology["seed_start"],
            "model_seed": topology["model_seed"],
            "episodes_expected": expected_episodes,
            "episodes_completed": len(episodes),
            "n_success": successes,
            "success_rate": successes / len(episodes) if episodes else 0.0,
            "mean_inference_time_s": (
                total_inference / total_replans if total_replans else None
            ),
            "max_steps": topology["max_steps_per_episode"],
            "replan_steps": topology["replan_steps"],
            "action_denoise_steps": topology["action_denoise_steps"],
            "video_fps": topology["video"]["fps"],
            "fastwam_reference_success_percent": task_entry[
                "fastwam_reference_success_percent"
            ],
            "episodes": episodes,
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", default=str(DEFAULT_TOPOLOGY))
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--episodes-per-task", type=int)
    args = parser.parse_args()
    if args.client_id not in range(16):
        parser.error("client id 必须位于0..15")
    if args.episodes_per_task is not None and args.episodes_per_task <= 0:
        parser.error("episodes per task 必须为正")
    return args


def main() -> int:
    args = _parse_args()
    topology = load_m6_evaluation_topology(args.topology, REPO_ROOT)
    client = topology["clients"][args.client_id]
    server = topology["servers"][client["server_id"]]
    expected_episodes = int(args.episodes_per_task or topology["episodes_per_task"])
    output_root = Path(args.output_root).expanduser().resolve()
    schema = load_configured_schema(
        {"resolved_panda_omron_schema": str(REPO_ROOT / topology["panda_omron_schema"])}
    )
    context, socket, poller = _open_policy_socket(server["frontend_port"])
    try:
        for task_entry in client["tasks"]:
            task = task_entry["name"]
            task_root = output_root / task
            video_root = task_root / "videos"
            result_path = task_root / "result.json"
            task_root.mkdir(parents=True, exist_ok=True)
            episodes = _load_completed_episodes(
                result_path,
                task=task,
                topology=topology,
                expected_episodes=expected_episodes,
                comparison_group=client.get("comparison_group"),
            )
            if len(episodes) == expected_episodes:
                print(f"[SKIP] task complete: {task}", flush=True)
                continue
            env = _create_environment(task, str(topology["scene"]["split"]))
            try:
                for episode_index in range(len(episodes), expected_episodes):
                    seed = int(topology["seed_start"]) + episode_index
                    video_path = video_root / f"episode_{episode_index:03d}_seed{seed}.mp4"
                    print(f"[START] task={task} seed={seed}", flush=True)
                    row = _run_episode(
                        env=env,
                        socket=socket,
                        poller=poller,
                        topology=topology,
                        schema=schema,
                        task=task,
                        episode_index=episode_index,
                        seed=seed,
                        video_path=video_path,
                    )
                    episodes.append(row)
                    _write_task_result(
                        path=result_path,
                        topology=topology,
                        client=client,
                        task_entry=task_entry,
                        episodes=episodes,
                        expected_episodes=expected_episodes,
                    )
                    print(
                        f"[DONE] task={task} episode={episode_index + 1}/"
                        f"{expected_episodes} success={row['success']} steps={row['steps']}",
                        flush=True,
                    )
            finally:
                env.close()
        print(f"[PASS] client {args.client_id} completed", flush=True)
        return 0
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    raise SystemExit(main())
