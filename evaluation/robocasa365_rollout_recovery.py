"""M4.3 atomic 长 horizon rollout 的可恢复进度合同。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from evaluation.robocasa365_benchmark import BenchmarkContractError
from evaluation.robocasa365_protocol import PROTOCOL_NAME


RECOVERY_SCHEMA_VERSION = 1
RECOVERY_MODE = "deterministic_action_replay"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_rollout_progress(
    *,
    run_id: str,
    task: str,
    episode_id: str,
    seed: int,
    max_steps: int,
    action_chunk_length: int,
    environment: dict[str, Any],
    initial_state: Any,
) -> dict[str, Any]:
    state = _finite_vector(initial_state, 16, "initial_state")
    return {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "recovery_mode": RECOVERY_MODE,
        "result": "running",
        "protocol": PROTOCOL_NAME,
        "scope": "atomic_only",
        "run_id": run_id,
        "task": task,
        "episode_id": episode_id,
        "seed": int(seed),
        "max_steps": int(max_steps),
        "action_chunk_length": int(action_chunk_length),
        "environment": environment,
        "initial_state": state.tolist(),
        "steps": 0,
        "success": False,
        "terminated": False,
        "truncated": False,
        "end_reason": "running",
        "request_records": [],
        "step_records": [],
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
    }


def append_policy_request(
    progress: dict[str, Any],
    *,
    request_id: str,
    step_id: int,
    actions_to_execute: Any,
    returned_action_shape: list[int],
    inference_seed: int,
    inference_seconds: float,
    round_trip_seconds: float,
    checkpoint: str,
    returned_action_min: Any,
    returned_action_max: Any,
) -> int:
    validate_rollout_progress(progress)
    if pending_action(progress) is not None:
        raise BenchmarkContractError("上一 policy chunk 尚未执行完，不能追加新请求")
    if int(step_id) != int(progress["steps"]):
        raise BenchmarkContractError(
            f"policy request step 与进度不一致：request={step_id}, progress={progress['steps']}"
        )
    actions = _finite_matrix(actions_to_execute, 12, "actions_to_execute")
    if not 1 <= actions.shape[0] <= int(progress["action_chunk_length"]):
        raise BenchmarkContractError(
            "持久化 action chunk 长度非法："
            f"actual={actions.shape[0]}, limit={progress['action_chunk_length']}"
        )
    if returned_action_shape != [32, 12]:
        raise BenchmarkContractError(
            f"M4.3 要求模型返回 [32,12]，实际={returned_action_shape}"
        )
    if any(record["request_id"] == request_id for record in progress["request_records"]):
        raise BenchmarkContractError(f"重复 request_id：{request_id}")

    record = {
        "request_id": request_id,
        "step_id": int(step_id),
        "returned_action_shape": list(returned_action_shape),
        "returned_action_steps": int(returned_action_shape[0]),
        "planned_action_steps": int(actions.shape[0]),
        "executed_action_steps": 0,
        "actions_to_execute": actions.tolist(),
        "inference_seed": int(inference_seed),
        "inference_seconds": _finite_nonnegative(inference_seconds, "inference_seconds"),
        "round_trip_seconds": _finite_nonnegative(
            round_trip_seconds, "round_trip_seconds"
        ),
        "checkpoint": str(checkpoint),
        "returned_action_min": _finite_vector(
            returned_action_min, 12, "returned_action_min"
        ).tolist(),
        "returned_action_max": _finite_vector(
            returned_action_max, 12, "returned_action_max"
        ).tolist(),
    }
    progress["request_records"].append(record)
    progress["updated_at_utc"] = utc_now()
    validate_rollout_progress(progress)
    return len(progress["request_records"]) - 1


def pending_action(
    progress: dict[str, Any],
) -> tuple[int, int, np.ndarray] | None:
    requests = progress.get("request_records", [])
    if not requests:
        return None
    index = len(requests) - 1
    request = requests[index]
    executed = int(request["executed_action_steps"])
    planned = int(request["planned_action_steps"])
    if executed >= planned:
        return None
    actions = _finite_matrix(request["actions_to_execute"], 12, "actions_to_execute")
    return index, executed, actions[executed].copy()


def append_executed_step(
    progress: dict[str, Any],
    *,
    request_index: int,
    chunk_offset: int,
    action: Any,
    state_after: Any,
    success: bool,
    terminated: bool,
    truncated: bool,
) -> None:
    validate_rollout_progress(progress)
    pending = pending_action(progress)
    if pending is None:
        raise BenchmarkContractError("没有待执行 action，不能追加 step")
    expected_request_index, expected_offset, expected_action = pending
    if (int(request_index), int(chunk_offset)) != (
        expected_request_index,
        expected_offset,
    ):
        raise BenchmarkContractError(
            "step 对应的 request/chunk offset 不连续："
            f"expected={(expected_request_index, expected_offset)}, "
            f"actual={(request_index, chunk_offset)}"
        )
    flat_action = _finite_vector(action, 12, "action")
    if not np.array_equal(flat_action, expected_action):
        raise BenchmarkContractError("执行 action 与已持久化 policy chunk 不一致")
    state = _finite_vector(state_after, 16, "state_after")
    step_id = int(progress["steps"])
    progress["step_records"].append(
        {
            "step_id": step_id,
            "request_index": int(request_index),
            "chunk_offset": int(chunk_offset),
            "action": flat_action.tolist(),
            "state_after": state.tolist(),
            "success": bool(success),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
    )
    progress["request_records"][request_index]["executed_action_steps"] += 1
    progress["steps"] = step_id + 1
    progress["success"] = bool(success)
    progress["terminated"] = bool(terminated)
    progress["truncated"] = bool(truncated)
    if success:
        progress["end_reason"] = "success"
    elif terminated:
        progress["end_reason"] = "terminated"
    elif truncated:
        progress["end_reason"] = "truncated"
    elif progress["steps"] >= progress["max_steps"]:
        progress["end_reason"] = "max_steps"
    else:
        progress["end_reason"] = "running"
    progress["updated_at_utc"] = utc_now()
    validate_rollout_progress(progress)


def complete_rollout_progress(progress: dict[str, Any]) -> None:
    validate_rollout_progress(progress)
    if progress["end_reason"] == "running":
        raise BenchmarkContractError("rollout 尚未到达 terminal/success/max_steps")
    progress["result"] = "pass"
    progress["completed_at_utc"] = utc_now()
    progress["updated_at_utc"] = progress["completed_at_utc"]
    validate_rollout_progress(progress, allow_complete=True)


def validate_rollout_progress(
    progress: dict[str, Any], *, allow_complete: bool = True
) -> dict[str, Any]:
    if not isinstance(progress, dict):
        raise BenchmarkContractError("rollout progress 顶层必须是对象")
    if progress.get("schema_version") != RECOVERY_SCHEMA_VERSION:
        raise BenchmarkContractError("不支持的 rollout progress schema")
    if progress.get("recovery_mode") != RECOVERY_MODE:
        raise BenchmarkContractError("M4.3 只支持 deterministic_action_replay")
    if progress.get("protocol") != PROTOCOL_NAME or progress.get("scope") != "atomic_only":
        raise BenchmarkContractError("rollout progress protocol/scope 不合法")
    if progress.get("result") not in ({"running", "pass"} if allow_complete else {"running"}):
        raise BenchmarkContractError(f"rollout progress result 非法：{progress.get('result')!r}")
    for key in ("run_id", "task", "episode_id"):
        if not isinstance(progress.get(key), str) or not progress[key]:
            raise BenchmarkContractError(f"rollout progress.{key} 必须为非空字符串")
    seed = int(progress.get("seed", -1))
    max_steps = int(progress.get("max_steps", 0))
    chunk_length = int(progress.get("action_chunk_length", 0))
    if seed < 0 or max_steps <= 0 or chunk_length <= 0 or chunk_length > max_steps:
        raise BenchmarkContractError("rollout progress seed/max_steps/chunk 非法")
    _finite_vector(progress.get("initial_state"), 16, "initial_state")
    requests = progress.get("request_records")
    steps = progress.get("step_records")
    if not isinstance(requests, list) or not isinstance(steps, list):
        raise BenchmarkContractError("rollout progress request/step records 必须为列表")
    if int(progress.get("steps", -1)) != len(steps) or len(steps) > max_steps:
        raise BenchmarkContractError("rollout progress steps 与 step_records 不一致")

    request_ids: set[str] = set()
    total_executed = 0
    expected_request_step = 0
    for index, request in enumerate(requests):
        request_id = request.get("request_id")
        if not isinstance(request_id, str) or not request_id or request_id in request_ids:
            raise BenchmarkContractError(f"rollout request_id 缺失或重复：{request_id!r}")
        request_ids.add(request_id)
        if request.get("returned_action_shape") != [32, 12]:
            raise BenchmarkContractError("rollout request 返回 shape 不是 [32,12]")
        if int(request.get("step_id", -1)) != expected_request_step:
            raise BenchmarkContractError(
                "rollout request step_id 不连续："
                f"expected={expected_request_step}, actual={request.get('step_id')}"
            )
        planned = int(request.get("planned_action_steps", 0))
        executed = int(request.get("executed_action_steps", -1))
        actions = _finite_matrix(request.get("actions_to_execute"), 12, "actions_to_execute")
        if planned != actions.shape[0] or not 1 <= planned <= chunk_length:
            raise BenchmarkContractError("rollout request planned actions 非法")
        if not 0 <= executed <= planned:
            raise BenchmarkContractError("rollout request executed actions 非法")
        if index < len(requests) - 1 and executed != planned:
            raise BenchmarkContractError("只有最后一个 policy request 可以存在未执行 action")
        _finite_vector(request.get("returned_action_min"), 12, "returned_action_min")
        _finite_vector(request.get("returned_action_max"), 12, "returned_action_max")
        _finite_nonnegative(request.get("inference_seconds"), "inference_seconds")
        _finite_nonnegative(request.get("round_trip_seconds"), "round_trip_seconds")
        total_executed += executed
        expected_request_step += executed
    if total_executed != len(steps):
        raise BenchmarkContractError("request executed action 总数与 step_records 不一致")

    expected_pairs = [
        (request_index, offset)
        for request_index, request in enumerate(requests)
        for offset in range(int(request["executed_action_steps"]))
    ]
    for expected_step, step in enumerate(steps):
        if int(step.get("step_id", -1)) != expected_step:
            raise BenchmarkContractError("step_records 的 step_id 不连续")
        request_index = int(step.get("request_index", -1))
        offset = int(step.get("chunk_offset", -1))
        if not 0 <= request_index < len(requests):
            raise BenchmarkContractError("step_records request_index 越界")
        if (request_index, offset) != expected_pairs[expected_step]:
            raise BenchmarkContractError("step_records request/chunk 顺序不连续")
        request = requests[request_index]
        if not 0 <= offset < int(request["executed_action_steps"]):
            raise BenchmarkContractError("step_records chunk_offset 越界")
        action = _finite_vector(step.get("action"), 12, "step.action")
        expected_action = np.asarray(request["actions_to_execute"][offset], dtype=np.float32)
        if not np.array_equal(action, expected_action):
            raise BenchmarkContractError("step action 与 request action 不一致")
        _finite_vector(step.get("state_after"), 16, "step.state_after")
    return progress


def compare_replayed_state(
    actual: Any, expected: Any, *, atol: float, label: str
) -> float:
    actual_array = _finite_vector(actual, 16, f"{label}.actual")
    expected_array = _finite_vector(expected, 16, f"{label}.expected")
    error = float(np.max(np.abs(actual_array - expected_array)))
    if error > float(atol):
        raise BenchmarkContractError(
            f"确定性回放 state 漂移：{label} max_abs_error={error:.9g} > atol={atol}"
        )
    return error


def _finite_vector(value: Any, dimension: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (dimension,) or not np.isfinite(array).all():
        raise BenchmarkContractError(
            f"{label} 必须为有限 [{dimension}]，实际={array.shape}"
        )
    return array


def _finite_matrix(value: Any, dimension: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if (
        array.ndim != 2
        or array.shape[0] <= 0
        or array.shape[1] != dimension
        or not np.isfinite(array).all()
    ):
        raise BenchmarkContractError(
            f"{label} 必须为非空有限 [T,{dimension}]，实际={array.shape}"
        )
    return array


def _finite_nonnegative(value: Any, label: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0:
        raise BenchmarkContractError(f"{label} 必须为有限非负数：{value!r}")
    return number
