"""RoboCasa365 atomic benchmark 的具名 observation/action 与结果合同。"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from data.robocasa365_schema import PandaOmronSchema, load_panda_omron_schema


class BenchmarkContractError(ValueError):
    """评测配置、观测、动作或结果不满足版本化合同。"""


M4_RANDOM_SMOKE_CAMERA_KEYS = [
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
]


def _read_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkContractError(f"无法读取 JSON：{resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BenchmarkContractError(f"JSON 顶层必须是对象：{resolved}")
    return payload


def load_atomic_task_manifest(path: str | Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("scope") != "atomic_only":
        raise BenchmarkContractError("评测任务清单必须为 scope=atomic_only")
    tasks = payload.get("tasks")
    horizons = payload.get("horizons")
    if not isinstance(tasks, list) or not tasks or not all(isinstance(task, str) for task in tasks):
        raise BenchmarkContractError("评测任务清单缺少非空 tasks 字符串列表")
    if len(tasks) != len(set(tasks)):
        raise BenchmarkContractError("评测任务清单包含重复任务")
    if not isinstance(horizons, dict) or set(horizons) != set(tasks):
        raise BenchmarkContractError("Atomic-Seen 每个任务都必须有且只有一个 horizon")
    for task, horizon in horizons.items():
        if not isinstance(horizon, int) or horizon <= 0:
            raise BenchmarkContractError(f"任务 horizon 必须为正整数：{task}={horizon!r}")
    return payload


def load_random_smoke_config(path: str | Path, repo_root: str | Path) -> dict[str, Any]:
    payload, resolved = _load_smoke_config_common(path, repo_root)
    if payload.get("policy") != "random_uniform_named_12d":
        raise BenchmarkContractError("M4.1 只允许 random_uniform_named_12d smoke policy")
    if payload["rollout"].get("base_motion_mode") not in {"sample", "zero"}:
        raise BenchmarkContractError("base_motion_mode 只允许 sample 或 zero")
    return resolved


def load_policy_smoke_config(path: str | Path, repo_root: str | Path) -> dict[str, Any]:
    payload, resolved = _load_smoke_config_common(path, repo_root)
    if payload.get("policy") != "xwam_broker_named_12d":
        raise BenchmarkContractError("M4.2 只允许 xwam_broker_named_12d policy")
    if payload.get("protocol") != "xwam.robocasa365.atomic.v1":
        raise BenchmarkContractError("M4.2 protocol 必须为 xwam.robocasa365.atomic.v1")
    try:
        action_chunk_length = int(payload["rollout"]["action_chunk_length"])
        minimum_policy_requests = int(payload["rollout"]["minimum_policy_requests"])
        broker_port = int(payload["network"]["broker_frontend_port"])
        timeout_seconds = float(payload["network"]["request_timeout_seconds"])
        cfg = float(payload["inference"]["cfg"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BenchmarkContractError("M4.2 配置缺少合法 chunk/network/inference 参数") from exc
    broker_address = payload["network"].get("broker_address")
    if action_chunk_length <= 0 or action_chunk_length > resolved["rollout"]["max_steps"]:
        raise BenchmarkContractError("action_chunk_length 必须为正且不能超过 smoke max_steps")
    if minimum_policy_requests <= 0:
        raise BenchmarkContractError("minimum_policy_requests 必须为正")
    if not isinstance(broker_address, str) or not broker_address:
        raise BenchmarkContractError("broker_address 必须为非空字符串")
    if not 1 <= broker_port <= 65535 or timeout_seconds <= 0:
        raise BenchmarkContractError("broker port/timeout 非法")
    if not math.isfinite(cfg) or cfg < 0:
        raise BenchmarkContractError("inference cfg 必须为有限非负数")
    resolved["protocol"] = payload["protocol"]
    resolved["rollout"] = dict(
        resolved["rollout"],
        action_chunk_length=action_chunk_length,
        minimum_policy_requests=minimum_policy_requests,
    )
    resolved["network"] = dict(
        payload["network"],
        broker_frontend_port=broker_port,
        request_timeout_seconds=timeout_seconds,
    )
    resolved["inference"] = dict(payload["inference"], cfg=cfg)
    return resolved


def load_policy_full_config(path: str | Path, repo_root: str | Path) -> dict[str, Any]:
    resolved = load_policy_smoke_config(path, repo_root)
    if resolved["episodes"] != 1:
        raise BenchmarkContractError("M4.3 首轮可恢复长评测只允许 episodes=1")
    if resolved["rollout"]["max_steps"] != resolved["official_horizon"]:
        raise BenchmarkContractError(
            "M4.3 必须使用官方完整 horizon："
            f"max_steps={resolved['rollout']['max_steps']}, "
            f"official={resolved['official_horizon']}"
        )
    recovery = resolved.get("recovery")
    if not isinstance(recovery, dict):
        raise BenchmarkContractError("M4.3 配置缺少 recovery 对象")
    if recovery.get("enabled") is not True:
        raise BenchmarkContractError("M4.3 必须启用 recovery")
    if recovery.get("mode") != "deterministic_action_replay":
        raise BenchmarkContractError("M4.3 recovery mode 必须为 deterministic_action_replay")
    try:
        state_replay_atol = float(recovery["state_replay_atol"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BenchmarkContractError("M4.3 state_replay_atol 必须为合法数值") from exc
    if not math.isfinite(state_replay_atol) or state_replay_atol <= 0:
        raise BenchmarkContractError("M4.3 state_replay_atol 必须为有限正数")
    if resolved["video"].get("durable_frames") is not True:
        raise BenchmarkContractError("M4.3 必须启用 durable_frames")
    resolved["recovery"] = dict(recovery, state_replay_atol=state_replay_atol)
    return resolved


def _load_smoke_config_common(
    path: str | Path, repo_root: str | Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(repo_root).resolve()
    payload = _read_json(path)
    if payload.get("schema_version") != 1:
        raise BenchmarkContractError(f"不支持的评测配置 schema：{payload.get('schema_version')!r}")
    if payload.get("scope") != "atomic_only":
        raise BenchmarkContractError("M4 评测配置必须为 scope=atomic_only")

    task_manifest_path = root / str(payload.get("task_manifest", ""))
    schema_path = root / str(payload.get("panda_omron_schema", ""))
    task_manifest = load_atomic_task_manifest(task_manifest_path)
    schema = load_panda_omron_schema(schema_path)
    task = payload.get("task")
    if task not in task_manifest["tasks"]:
        raise BenchmarkContractError(f"任务不在 Atomic-Seen 清单：{task!r}")
    if schema.action.dimension != 12 or schema.state.dimension != 16:
        raise BenchmarkContractError("M4.1 要求 PandaOmron 16D state / 12D action schema")

    try:
        episodes = int(payload["episodes"])
        seed_start = int(payload["seed_start"])
        max_steps = int(payload["rollout"]["max_steps"])
        layout_id = int(payload["scene"]["layout_id"])
        style_id = int(payload["scene"]["style_id"])
        video_stride = int(payload["video"]["stride"])
        video_fps = int(payload["video"]["fps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BenchmarkContractError("M4.1 配置缺少合法整数参数") from exc
    if episodes <= 0 or seed_start < 0 or max_steps <= 0:
        raise BenchmarkContractError("episodes/max_steps 必须为正，seed_start 不能为负")
    official_horizon = int(task_manifest["horizons"][task])
    if max_steps > official_horizon:
        raise BenchmarkContractError(
            f"smoke max_steps 不能超过官方 horizon：{max_steps} > {official_horizon}"
        )
    if layout_id <= 0 or style_id <= 0:
        raise BenchmarkContractError("layout_id/style_id 必须为正整数")
    if video_stride <= 0 or video_fps <= 0:
        raise BenchmarkContractError("video stride/fps 必须为正整数")
    if payload["rollout"].get("depth_mode") != "disabled":
        raise BenchmarkContractError("M4.1 随机 RGB smoke 必须为 depth_mode=disabled")
    if payload["rollout"].get("stop_on_success") is not True:
        raise BenchmarkContractError("M4 smoke 必须设置 stop_on_success=true")
    if payload["scene"].get("obj_instance_split") != "target":
        raise BenchmarkContractError("M4.1 必须固定 obj_instance_split=target")
    if payload["scene"].get("split") is not None:
        raise BenchmarkContractError("M4.1 必须使用 split=null 并显式固定 target object split")
    if payload["scene"].get("randomize_cameras") is not False:
        raise BenchmarkContractError("M4.1 必须关闭 camera randomization")
    if payload["scene"].get("generative_textures") is not None:
        raise BenchmarkContractError("M4.1 必须关闭 generative textures")

    camera_keys = payload["video"].get("camera_keys")
    camera_shape = payload["video"].get("camera_shape")
    if camera_keys != M4_RANDOM_SMOKE_CAMERA_KEYS:
        raise BenchmarkContractError(
            f"M4.1 camera 顺序必须固定为：{M4_RANDOM_SMOKE_CAMERA_KEYS}"
        )
    if camera_shape != [256, 256, 3]:
        raise BenchmarkContractError("M4.1 camera_shape 必须为 [256,256,3]")

    resolved = dict(payload)
    resolved["episodes"] = episodes
    resolved["seed_start"] = seed_start
    resolved["rollout"] = dict(payload["rollout"], max_steps=max_steps)
    resolved["scene"] = dict(payload["scene"], layout_id=layout_id, style_id=style_id)
    resolved["video"] = dict(payload["video"], stride=video_stride, fps=video_fps)
    resolved["official_horizon"] = official_horizon
    resolved["resolved_task_manifest"] = str(task_manifest_path.resolve())
    resolved["resolved_panda_omron_schema"] = str(schema_path.resolve())
    return payload, resolved


def load_configured_schema(config: dict[str, Any]) -> PandaOmronSchema:
    return load_panda_omron_schema(config["resolved_panda_omron_schema"])


def pack_online_state(observation: dict[str, Any], schema: PandaOmronSchema) -> np.ndarray:
    parts: list[np.ndarray] = []
    for component in schema.state.components:
        key = f"state.{component.name}"
        if key not in observation:
            raise BenchmarkContractError(f"online observation 缺少 state key：{key}")
        part = np.asarray(observation[key], dtype=np.float32).reshape(-1)
        if part.shape != (component.size,):
            raise BenchmarkContractError(
                f"online state 维度错误：{key} expected={component.size}, actual={part.shape}"
            )
        if not np.isfinite(part).all():
            raise BenchmarkContractError(f"online state 包含 NaN/Inf：{key}")
        parts.append(part)
    packed = np.concatenate(parts).astype(np.float32, copy=False)
    if packed.shape != (schema.state.dimension,):
        raise BenchmarkContractError(
            f"online state 总维度错误：expected={schema.state.dimension}, actual={packed.shape}"
        )
    return packed


def pack_online_cameras(
    observation: dict[str, Any], camera_keys: list[str], camera_shape: list[int]
) -> np.ndarray:
    cameras: list[np.ndarray] = []
    expected_shape = tuple(int(value) for value in camera_shape)
    for key in camera_keys:
        if key not in observation:
            raise BenchmarkContractError(f"online observation 缺少 camera key：{key}")
        camera = np.asarray(observation[key])
        if camera.shape != expected_shape or camera.dtype != np.uint8:
            raise BenchmarkContractError(
                f"camera 合同错误：{key} expected={expected_shape}/uint8, "
                f"actual={camera.shape}/{camera.dtype}"
            )
        cameras.append(camera)
    return np.stack(cameras, axis=0)


def tile_camera_views(cameras: np.ndarray) -> np.ndarray:
    array = np.asarray(cameras)
    if array.ndim != 4 or array.shape[0] != 3 or array.shape[-1] != 3:
        raise BenchmarkContractError(f"三相机 tensor 应为 [3,H,W,3]，实际={array.shape}")
    return np.concatenate([array[index] for index in range(3)], axis=1)


def flat_action_to_gym_dict(action: Any, schema: PandaOmronSchema) -> dict[str, np.ndarray]:
    flat = np.asarray(action, dtype=np.float32).reshape(-1)
    if flat.shape != (schema.action.dimension,) or not np.isfinite(flat).all():
        raise BenchmarkContractError(
            f"policy action 必须为有限 [{schema.action.dimension}]，实际={flat.shape}"
        )
    output: dict[str, np.ndarray] = {}
    for component in schema.action.components:
        output[f"action.{component.name}"] = flat[component.start : component.end].copy()
    if sum(value.size for value in output.values()) != schema.action.dimension:
        raise BenchmarkContractError("Gym 动作打包丢失了 PandaOmron action 维度")
    return output


def sample_random_flat_action(
    rng: np.random.Generator,
    schema: PandaOmronSchema,
    *,
    base_motion_mode: str,
) -> np.ndarray:
    if base_motion_mode not in {"sample", "zero"}:
        raise BenchmarkContractError(f"未知 base_motion_mode：{base_motion_mode}")
    action = rng.uniform(-1.0, 1.0, size=schema.action.dimension).astype(np.float32)
    by_name = {component.name: component for component in schema.action.components}
    base = by_name["base_motion"]
    control = by_name["control_mode"]
    gripper = by_name["gripper_close"]
    if base_motion_mode == "zero":
        action[base.start : base.end] = 0.0
    action[control.start : control.end] = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32))
    action[gripper.start : gripper.end] = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32))
    return action


def validate_gym_action_space(action: dict[str, np.ndarray], action_space: Any) -> None:
    spaces = getattr(action_space, "spaces", None)
    if not isinstance(spaces, dict):
        raise BenchmarkContractError("RoboCasa Gym action_space 必须为 spaces.Dict")
    if set(action) != set(spaces):
        raise BenchmarkContractError(
            f"Gym action key 不一致：expected={sorted(spaces)}, actual={sorted(action)}"
        )
    for key, value in action.items():
        expected_shape = tuple(spaces[key].shape)
        if value.shape != expected_shape:
            raise BenchmarkContractError(
                f"Gym action shape 错误：{key} expected={expected_shape}, actual={value.shape}"
            )
        if not np.isfinite(value).all():
            raise BenchmarkContractError(f"Gym action 包含 NaN/Inf：{key}")
        low = np.asarray(spaces[key].low, dtype=np.float32)
        high = np.asarray(spaces[key].high, dtype=np.float32)
        if np.any(value < low - 1e-6) or np.any(value > high + 1e-6):
            raise BenchmarkContractError(
                f"Gym action 越界：{key} range=[{float(value.min())},{float(value.max())}]"
            )


def aggregate_episode_records(
    records: list[dict[str, Any]], *, expected_episodes: int | None = None
) -> dict[str, Any]:
    if not records:
        raise BenchmarkContractError("不能聚合空 episode 记录")
    episode_ids: set[str] = set()
    for record in records:
        episode_id = record.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id or episode_id in episode_ids:
            raise BenchmarkContractError(f"episode_id 缺失或重复：{episode_id!r}")
        episode_ids.add(episode_id)
        if record.get("result") not in {"pass", "fail"}:
            raise BenchmarkContractError(f"episode result 非法：{record.get('result')!r}")
    expected = len(records) if expected_episodes is None else int(expected_episodes)
    if expected < len(records) or expected <= 0:
        raise BenchmarkContractError(
            f"expected_episodes 必须不小于现有记录数且为正：{expected} < {len(records)}"
        )
    passed = [record for record in records if record["result"] == "pass"]
    failed_records = len(records) - len(passed)
    missing_records = expected - len(records)
    successes = sum(bool(record.get("success")) for record in passed)
    end_reasons = Counter(str(record.get("end_reason", "unknown")) for record in records)
    if missing_records:
        end_reasons["missing"] += missing_records
    return {
        "episodes_expected": expected,
        "episodes_observed": len(records),
        "episodes_completed": len(passed),
        "episodes_failed": failed_records,
        "episodes_missing": missing_records,
        "successes": successes,
        "success_rate": successes / len(passed) if passed else 0.0,
        "total_steps": sum(int(record.get("steps", 0)) for record in passed),
        "base_nonzero_steps": sum(int(record.get("action_diagnostics", {}).get("base_nonzero_steps", 0)) for record in passed),
        "end_reason_counts": dict(sorted(end_reasons.items())),
    }
