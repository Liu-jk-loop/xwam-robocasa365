#!/usr/bin/env python3
"""运行或恢复 M4.3 RoboCasa365 atomic X-WAM 长 horizon 闭环。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


os.environ.setdefault("MUJOCO_GL", "egl")
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.robocasa365_benchmark import (  # noqa: E402
    BenchmarkContractError,
    aggregate_episode_records,
    flat_action_to_gym_dict,
    load_configured_schema,
    load_policy_full_config,
    pack_online_cameras,
    pack_online_state,
    tile_camera_views,
    validate_gym_action_space,
)
from evaluation.robocasa365_protocol import PROTOCOL_NAME, make_request  # noqa: E402
from evaluation.robocasa365_rollout_recovery import (  # noqa: E402
    append_executed_step,
    append_policy_request,
    compare_replayed_state,
    complete_rollout_progress,
    create_rollout_progress,
    pending_action,
    utc_now,
    validate_rollout_progress,
)
from evaluation.run_robocasa365_policy_rollout import (  # noqa: E402
    _create_environment,
    _environment_metadata,
    _request_policy,
    _runtime_versions,
    _run_id,
    _validate_runtime_horizon,
)
from project_tools.training_run import collect_git_state, write_json_atomic  # noqa: E402


DEFAULT_CONFIG = (
    REPO_ROOT / "configs" / "evaluation" / "robocasa365_close_fridge_m4_full.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkContractError(f"无法读取 JSON：{path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BenchmarkContractError(f"JSON 顶层必须为对象：{path}")
    return payload


def _frame_path(frame_dir: Path, step: int) -> Path:
    return frame_dir / f"frame_{int(step):06d}.png"


def _save_frame_atomic(frame_dir: Path, step: int, cameras: np.ndarray) -> Path:
    import imageio.v2 as imageio

    frame_dir.mkdir(parents=True, exist_ok=True)
    output = _frame_path(frame_dir, step)
    if output.is_file() and output.stat().st_size > 0:
        return output
    temporary = output.with_name(f".{output.stem}.partial.png")
    imageio.imwrite(temporary, tile_camera_views(cameras))
    os.replace(temporary, output)
    return output


def _build_video(frame_dir: Path, output: Path, fps: int) -> int:
    import imageio.v2 as imageio

    frames = sorted(frame_dir.glob("frame_*.png"))
    if not frames:
        raise BenchmarkContractError("durable video frame cache 为空")
    temporary = output.with_name(f".{output.stem}.partial.mp4")
    writer = imageio.get_writer(temporary, fps=int(fps))
    try:
        for frame in frames:
            writer.append_data(imageio.imread(frame))
    finally:
        writer.close()
    if not temporary.is_file() or temporary.stat().st_size <= 0:
        raise BenchmarkContractError("视频编码没有生成非空 MP4")
    os.replace(temporary, output)
    return len(frames)


def _validate_resume_git(metadata: dict[str, Any]) -> dict[str, Any]:
    expected = metadata.get("git")
    actual = collect_git_state(REPO_ROOT)
    if not isinstance(expected, dict) or expected.get("dirty") is not False:
        raise BenchmarkContractError("原 run metadata 缺少干净 Git provenance")
    if actual.get("dirty") is not False:
        raise BenchmarkContractError(f"恢复前工作区必须干净：{actual.get('status')}")
    if actual.get("commit") != expected.get("commit"):
        raise BenchmarkContractError(
            "恢复必须使用原 commit："
            f"expected={expected.get('commit')}, actual={actual.get('commit')}"
        )
    return actual


def _metadata_matches_environment(
    expected: dict[str, Any], actual: dict[str, Any]
) -> None:
    for key in ("layout_id", "style_id", "language"):
        if expected.get(key) != actual.get(key):
            raise BenchmarkContractError(
                f"恢复环境 metadata 漂移：{key} expected={expected.get(key)!r}, "
                f"actual={actual.get(key)!r}"
            )


def _should_save_frame(config: dict[str, Any], progress: dict[str, Any]) -> bool:
    step = int(progress["steps"])
    return (
        step % int(config["video"]["stride"]) == 0
        or progress["end_reason"] != "running"
    )


def _replay_progress(
    *,
    env: Any,
    observation: dict[str, Any],
    config: dict[str, Any],
    schema: Any,
    progress: dict[str, Any],
    frame_dir: Path,
) -> tuple[dict[str, Any], int, float]:
    state = pack_online_state(observation, schema)
    max_error = compare_replayed_state(
        state,
        progress["initial_state"],
        atol=float(config["recovery"]["state_replay_atol"]),
        label="reset",
    )
    camera_keys = list(config["video"]["camera_keys"])
    camera_shape = list(config["video"]["camera_shape"])
    if bool(config["video"]["enabled"]):
        cameras = pack_online_cameras(observation, camera_keys, camera_shape)
        _save_frame_atomic(frame_dir, 0, cameras)

    for record in progress["step_records"]:
        action = np.asarray(record["action"], dtype=np.float32)
        gym_action = flat_action_to_gym_dict(action, schema)
        validate_gym_action_space(gym_action, env.action_space)
        observation, _, terminated, truncated, step_info = env.step(gym_action)
        state = pack_online_state(observation, schema)
        error = compare_replayed_state(
            state,
            record["state_after"],
            atol=float(config["recovery"]["state_replay_atol"]),
            label=f"step_{record['step_id']}",
        )
        max_error = max(max_error, error)
        actual_flags = (
            bool(step_info.get("success", False)),
            bool(terminated),
            bool(truncated),
        )
        expected_flags = (
            bool(record["success"]),
            bool(record["terminated"]),
            bool(record["truncated"]),
        )
        if actual_flags != expected_flags:
            raise BenchmarkContractError(
                "确定性回放 terminal flags 漂移："
                f"step={record['step_id']}, expected={expected_flags}, actual={actual_flags}"
            )
        completed_steps = int(record["step_id"]) + 1
        if bool(config["video"]["enabled"]) and (
            completed_steps % int(config["video"]["stride"]) == 0
            or any(actual_flags)
        ):
            cameras = pack_online_cameras(observation, camera_keys, camera_shape)
            _save_frame_atomic(frame_dir, completed_steps, cameras)
    return observation, len(progress["step_records"]), max_error


def _episode_record(
    *,
    config: dict[str, Any],
    schema: Any,
    progress: dict[str, Any],
    episode_dir: Path,
    started_at: str,
    duration_seconds: float,
    resumed: bool,
    replayed_steps: int,
    replay_max_state_error: float,
    video_frame_count: int,
) -> dict[str, Any]:
    states = np.asarray(
        [progress["initial_state"]]
        + [record["state_after"] for record in progress["step_records"]],
        dtype=np.float32,
    )
    actions = np.asarray(
        [record["action"] for record in progress["step_records"]],
        dtype=np.float32,
    )
    components = {component.name: component for component in schema.action.components}
    base = components["base_motion"]
    control = components["control_mode"]
    gripper = components["gripper_close"]
    request_records = []
    for record in progress["request_records"]:
        request_records.append(
            {
                key: value
                for key, value in record.items()
                if key != "actions_to_execute"
            }
        )
    video_enabled = bool(config["video"]["enabled"])
    return {
        "schema_version": 1,
        "episode_id": progress["episode_id"],
        "result": "pass",
        "task": config["task"],
        "scope": "atomic_only",
        "policy": config["policy"],
        "protocol": config["protocol"],
        "seed": progress["seed"],
        "official_horizon": config["official_horizon"],
        "max_steps": config["rollout"]["max_steps"],
        "action_chunk_length": config["rollout"]["action_chunk_length"],
        "steps": progress["steps"],
        "policy_requests": len(progress["request_records"]),
        "success": progress["success"],
        "end_reason": progress["end_reason"],
        "terminated": progress["terminated"],
        "truncated": progress["truncated"],
        "started_at_utc": progress["created_at_utc"],
        "final_invocation_started_at_utc": started_at,
        "final_invocation_duration_seconds": duration_seconds,
        "environment": progress["environment"],
        "recovery": {
            "mode": progress["recovery_mode"],
            "resumed": resumed,
            "replayed_steps": replayed_steps,
            "replay_max_state_error": replay_max_state_error,
            "state_replay_atol": config["recovery"]["state_replay_atol"],
            "progress_path": str(episode_dir / "progress.json"),
        },
        "observation_diagnostics": {
            "state_dimension": schema.state.dimension,
            "state_min": states.min(axis=0).tolist(),
            "state_max": states.max(axis=0).tolist(),
            "camera_keys": config["video"]["camera_keys"],
            "camera_shape": config["video"]["camera_shape"],
            "depth_mode": config["rollout"]["depth_mode"],
        },
        "action_diagnostics": {
            "action_dimension": schema.action.dimension,
            "action_min": actions.min(axis=0).tolist() if len(actions) else None,
            "action_max": actions.max(axis=0).tolist() if len(actions) else None,
            "base_nonzero_steps": int(
                np.count_nonzero(
                    np.linalg.norm(actions[:, base.start : base.end], axis=1) > 1e-8
                )
            )
            if len(actions)
            else 0,
            "control_modes": sorted(
                set(float(value) for value in actions[:, control.start : control.end].flat)
            )
            if len(actions)
            else [],
            "gripper_min": float(actions[:, gripper.start : gripper.end].min())
            if len(actions)
            else None,
            "gripper_max": float(actions[:, gripper.start : gripper.end].max())
            if len(actions)
            else None,
        },
        "request_records": request_records,
        "video": {
            "enabled": video_enabled,
            "path": str(episode_dir / "rollout.mp4") if video_enabled else None,
            "frame_cache": str(episode_dir / "video_frames") if video_enabled else None,
            "frame_count": video_frame_count,
            "stride": config["video"]["stride"],
            "fps": config["video"]["fps"],
        },
    }


def _run_episode(
    *,
    config: dict[str, Any],
    schema: Any,
    socket: Any,
    poller: Any,
    run_id: str,
    episode_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    episode_index = 0
    seed = int(config["seed_start"])
    episode_id = f"episode_{episode_index:03d}_seed_{seed:06d}"
    progress_path = episode_dir / "progress.json"
    frame_dir = episode_dir / "video_frames"
    episode_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    started = time.monotonic()
    env = _create_environment(config, seed)
    try:
        observation, reset_info = env.reset(seed=seed)
        environment = _environment_metadata(env)
        prompt = str(environment["language"])
        if not prompt:
            raise BenchmarkContractError("RoboCasa episode metadata 缺少语言指令")
        state = pack_online_state(observation, schema)
        if resume:
            progress = _read_json(progress_path)
            validate_rollout_progress(progress)
            if (
                progress["run_id"] != run_id
                or progress["task"] != config["task"]
                or progress["episode_id"] != episode_id
                or int(progress["seed"]) != seed
                or int(progress["max_steps"]) != int(config["rollout"]["max_steps"])
            ):
                raise BenchmarkContractError("恢复 progress 与 run/config 不一致")
            _metadata_matches_environment(progress["environment"], environment)
            observation, replayed_steps, replay_max_error = _replay_progress(
                env=env,
                observation=observation,
                config=config,
                schema=schema,
                progress=progress,
                frame_dir=frame_dir,
            )
            print(
                "M4.3 deterministic replay passed: "
                f"steps={replayed_steps} max_state_error={replay_max_error:.9g}",
                flush=True,
            )
        else:
            progress = create_rollout_progress(
                run_id=run_id,
                task=config["task"],
                episode_id=episode_id,
                seed=seed,
                max_steps=int(config["rollout"]["max_steps"]),
                action_chunk_length=int(config["rollout"]["action_chunk_length"]),
                environment=environment,
                initial_state=state,
            )
            progress["success"] = bool(reset_info.get("success", False))
            progress["end_reason"] = (
                "success" if progress["success"] else "running"
            )
            write_json_atomic(progress_path, progress)
            if bool(config["video"]["enabled"]):
                cameras = pack_online_cameras(
                    observation,
                    list(config["video"]["camera_keys"]),
                    list(config["video"]["camera_shape"]),
                )
                _save_frame_atomic(frame_dir, 0, cameras)
            replayed_steps = 0
            replay_max_error = 0.0

        camera_keys = list(config["video"]["camera_keys"])
        camera_shape = list(config["video"]["camera_shape"])
        while progress["end_reason"] == "running":
            pending = pending_action(progress)
            if pending is None:
                state = pack_online_state(observation, schema)
                cameras = pack_online_cameras(observation, camera_keys, camera_shape)
                request_id = (
                    f"{run_id}-{episode_id}-step-{progress['steps']:06d}"
                )
                request = make_request(
                    request_id=request_id,
                    task=config["task"],
                    episode_id=episode_id,
                    seed=seed,
                    step_id=int(progress["steps"]),
                    prompt=prompt,
                    video=cameras,
                    state=state,
                    cfg=float(config["inference"]["cfg"]),
                )
                response = _request_policy(
                    socket,
                    poller,
                    request,
                    float(config["network"]["request_timeout_seconds"]),
                )
                actions = response["actions"]
                execute_count = min(
                    int(config["rollout"]["action_chunk_length"]),
                    int(config["rollout"]["max_steps"]) - int(progress["steps"]),
                )
                if actions.shape[0] < execute_count:
                    raise BenchmarkContractError(
                        f"policy action chunk 太短：required={execute_count}, actual={actions.shape[0]}"
                    )
                checkpoints = {
                    record["checkpoint"] for record in progress["request_records"]
                }
                if checkpoints and response["checkpoint"] not in checkpoints:
                    raise BenchmarkContractError(
                        "恢复后的 policy checkpoint 与已有请求不一致："
                        f"existing={sorted(checkpoints)}, actual={response['checkpoint']}"
                    )
                append_policy_request(
                    progress,
                    request_id=request_id,
                    step_id=int(progress["steps"]),
                    actions_to_execute=actions[:execute_count],
                    returned_action_shape=list(actions.shape),
                    inference_seed=response["inference_seed"],
                    inference_seconds=response["inference_seconds"],
                    round_trip_seconds=response["round_trip_seconds"],
                    checkpoint=response["checkpoint"],
                    returned_action_min=actions.min(axis=0),
                    returned_action_max=actions.max(axis=0),
                )
                write_json_atomic(progress_path, progress)
                pending = pending_action(progress)

            if pending is None:
                raise BenchmarkContractError("policy request 后没有可执行 action")
            request_index, chunk_offset, flat_action = pending
            gym_action = flat_action_to_gym_dict(flat_action, schema)
            validate_gym_action_space(gym_action, env.action_space)
            observation, _, terminated, truncated, step_info = env.step(gym_action)
            state_after = pack_online_state(observation, schema)
            append_executed_step(
                progress,
                request_index=request_index,
                chunk_offset=chunk_offset,
                action=flat_action,
                state_after=state_after,
                success=bool(step_info.get("success", False)),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
            write_json_atomic(progress_path, progress)
            if bool(config["video"]["enabled"]) and _should_save_frame(config, progress):
                cameras = pack_online_cameras(observation, camera_keys, camera_shape)
                _save_frame_atomic(frame_dir, int(progress["steps"]), cameras)
            if pending_action(progress) is None or progress["end_reason"] != "running":
                print(
                    "M4.3 progress saved: "
                    f"steps={progress['steps']}/{progress['max_steps']} "
                    f"requests={len(progress['request_records'])} "
                    f"end_reason={progress['end_reason']}",
                    flush=True,
                )

        if len(progress["request_records"]) < int(
            config["rollout"]["minimum_policy_requests"]
        ):
            raise BenchmarkContractError("长 rollout 没有完成最小 policy request 数")
        complete_rollout_progress(progress)
        write_json_atomic(progress_path, progress)
        if bool(config["video"]["enabled"]):
            video_frame_count = _build_video(
                frame_dir,
                episode_dir / "rollout.mp4",
                int(config["video"]["fps"]),
            )
        else:
            video_frame_count = 0
        record = _episode_record(
            config=config,
            schema=schema,
            progress=progress,
            episode_dir=episode_dir,
            started_at=started_at,
            duration_seconds=time.monotonic() - started,
            resumed=resume,
            replayed_steps=replayed_steps,
            replay_max_state_error=replay_max_error,
            video_frame_count=video_frame_count,
        )
        write_json_atomic(episode_dir / "episode.json", record)
        return record
    finally:
        env.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root")
    parser.add_argument(
        "--resume-run-dir",
        help="恢复已有 run 目录；必须使用原 commit，且不能同时覆盖 config/output。",
    )
    args = parser.parse_args()
    if args.resume_run_dir and args.output_root:
        parser.error("--resume-run-dir 不能与 --output-root 同时使用")
    return args


def main() -> int:
    args = _parse_args()
    resume = args.resume_run_dir is not None
    started = time.monotonic()
    context = None
    socket = None
    if resume:
        run_dir = Path(args.resume_run_dir).expanduser().resolve()
        if not run_dir.is_dir():
            print(f"恢复 run 目录不存在：{run_dir}", file=sys.stderr)
            return 2
        requested_path = run_dir / "requested_config.json"
        metadata = _read_json(run_dir / "metadata.json")
        _validate_resume_git(metadata)
        run_id = str(metadata["run_id"])
    else:
        run_id = _run_id()
        raw_config = _read_json(Path(args.config).expanduser().resolve())
        output_root = Path(
            args.output_root
            or raw_config.get("output_root", "eval_results/robocasa365_m4_full")
        ).expanduser().resolve()
        run_dir = output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        requested_path = run_dir / "requested_config.json"
        if args.output_root:
            raw_config["output_root"] = args.output_root
        write_json_atomic(requested_path, raw_config)

    config = load_policy_full_config(requested_path, REPO_ROOT)
    schema = load_configured_schema(config)
    runtime_horizon = _validate_runtime_horizon(
        config["task"], config["official_horizon"]
    )
    if not resume:
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "started_at_utc": utc_now(),
            "git": collect_git_state(REPO_ROOT),
            "python_executable": sys.executable,
            "config_path": str(Path(args.config).expanduser().resolve()),
            "resolved_config": config,
            "runtime_horizon": runtime_horizon,
            "simulator_runtime": _runtime_versions(),
            "mujoco_gl": os.environ.get("MUJOCO_GL", "<unset>"),
            "broker": (
                f"tcp://{config['network']['broker_address']}:"
                f"{config['network']['broker_frontend_port']}"
            ),
            "protocol": PROTOCOL_NAME,
            "recovery": {
                "enabled": True,
                "mode": config["recovery"]["mode"],
                "state_replay_atol": config["recovery"]["state_replay_atol"],
            },
        }
        if metadata["git"].get("dirty") is not False:
            raise BenchmarkContractError("M4.3 新 run 必须从干净 Git 工作区启动")
        write_json_atomic(run_dir / "metadata.json", metadata)
    elif int(metadata.get("runtime_horizon", -1)) != runtime_horizon:
        raise BenchmarkContractError("恢复时 RoboCasa runtime horizon 已漂移")

    episode_id = f"episode_000_seed_{int(config['seed_start']):06d}"
    episode_dir = run_dir / config["task"] / episode_id
    try:
        import zmq

        context = zmq.Context()
        socket = context.socket(zmq.DEALER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(
            zmq.IDENTITY,
            f"robocasa365-resumable-{run_id}-{os.getpid()}".encode("utf-8"),
        )
        socket.connect(
            f"tcp://{config['network']['broker_address']}:"
            f"{config['network']['broker_frontend_port']}"
        )
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        record = _run_episode(
            config=config,
            schema=schema,
            socket=socket,
            poller=poller,
            run_id=run_id,
            episode_dir=episode_dir,
            resume=resume,
        )
        aggregate = aggregate_episode_records([record], expected_episodes=1)
        summary = {
            "schema_version": 1,
            "run_id": run_id,
            "result": "pass",
            "task": config["task"],
            "scope": "atomic_only",
            "policy": config["policy"],
            "protocol": config["protocol"],
            "resumed": resume,
            "finished_at_utc": utc_now(),
            "invocation_duration_seconds": time.monotonic() - started,
            "aggregate": aggregate,
            "policy_requests": record["policy_requests"],
            "episode_records": [str(episode_dir / "episode.json")],
        }
        write_json_atomic(run_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        error_type = "KeyboardInterrupt"
        error_message = "用户中断；progress 已原子保存，可使用 --resume-run-dir 恢复"
        traceback_text = None
        exit_code = 130
    except Exception as exc:
        error_type = type(exc).__name__
        error_message = str(exc)
        traceback_text = traceback.format_exc()
        exit_code = 1
    finally:
        if socket is not None:
            socket.close()
        if context is not None:
            context.term()

    progress_path = episode_dir / "progress.json"
    progress = _read_json(progress_path) if progress_path.is_file() else None
    interruption = {
        "schema_version": 1,
        "run_id": run_id,
        "result": "interrupted",
        "resumable": progress is not None,
        "error": f"{error_type}: {error_message}",
        "traceback": traceback_text,
        "progress_path": str(progress_path),
        "completed_steps": int(progress["steps"]) if progress else 0,
        "completed_policy_requests": len(progress["request_records"]) if progress else 0,
        "finished_at_utc": utc_now(),
    }
    write_json_atomic(run_dir / "interruption.json", interruption)
    write_json_atomic(run_dir / "summary.json", interruption)
    print(json.dumps(interruption, ensure_ascii=False, indent=2), file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
