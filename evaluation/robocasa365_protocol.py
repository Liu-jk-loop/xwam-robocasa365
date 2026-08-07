"""M4.2 RoboCasa365 atomic policy 请求/响应协议。"""

from __future__ import annotations

import hashlib
import io
import json
from typing import Any

import numpy as np


PROTOCOL_NAME = "xwam.robocasa365.atomic.v1"
REQUEST_KIND = "policy_request"
RESPONSE_KIND = "policy_response"
ERROR_KIND = "policy_error"
MAX_WIRE_BYTES = 4 * 1024 * 1024


class PolicyProtocolError(ValueError):
    """网络消息不满足版本化的 atomic-only policy 合同。"""


def deterministic_inference_seed(task: str, episode_id: str, step_id: int) -> int:
    key = f"{PROTOCOL_NAME}|{task}|{episode_id}|{int(step_id)}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "big")


def make_request(
    *,
    request_id: str,
    task: str,
    episode_id: str,
    seed: int,
    step_id: int,
    prompt: str,
    video: Any,
    state: Any,
    cfg: float,
) -> dict[str, Any]:
    request = {
        "protocol": PROTOCOL_NAME,
        "kind": REQUEST_KIND,
        "request_id": request_id,
        "scope": "atomic_only",
        "task": task,
        "episode_id": episode_id,
        "seed": int(seed),
        "step_id": int(step_id),
        "prompt": prompt,
        "cfg": float(cfg),
        "video": np.asarray(video),
        "state": np.asarray(state, dtype=np.float32),
    }
    return validate_request(request)


def validate_request(payload: dict[str, Any]) -> dict[str, Any]:
    _validate_envelope(payload, REQUEST_KIND)
    for key in ("request_id", "task", "episode_id", "prompt"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise PolicyProtocolError(f"request.{key} 必须为非空字符串")
    if payload.get("scope") != "atomic_only":
        raise PolicyProtocolError("policy request 必须为 scope=atomic_only")
    try:
        seed = int(payload["seed"])
        step_id = int(payload["step_id"])
        cfg = float(payload["cfg"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PolicyProtocolError("request seed/step_id/cfg 必须为合法数值") from exc
    if seed < 0 or step_id < 0 or not np.isfinite(cfg):
        raise PolicyProtocolError("request seed/step_id 不能为负，cfg 必须有限")

    video = np.asarray(payload.get("video"))
    if video.shape != (3, 256, 256, 3) or video.dtype != np.uint8:
        raise PolicyProtocolError(
            "request.video 必须为 [3,256,256,3] uint8，"
            f"实际={video.shape}/{video.dtype}"
        )
    state = np.asarray(payload.get("state"), dtype=np.float32)
    if state.shape != (16,) or not np.isfinite(state).all():
        raise PolicyProtocolError(f"request.state 必须为有限 [16]，实际={state.shape}")

    validated = dict(payload)
    validated.update(seed=seed, step_id=step_id, cfg=cfg, video=video, state=state)
    return validated


def make_success_response(
    request: dict[str, Any],
    *,
    actions: Any,
    inference_seed: int,
    inference_seconds: float,
    checkpoint: str,
) -> dict[str, Any]:
    response = {
        "protocol": PROTOCOL_NAME,
        "kind": RESPONSE_KIND,
        "request_id": request["request_id"],
        "scope": "atomic_only",
        "task": request["task"],
        "episode_id": request["episode_id"],
        "step_id": int(request["step_id"]),
        "actions": np.asarray(actions, dtype=np.float32),
        "inference_seed": int(inference_seed),
        "inference_seconds": float(inference_seconds),
        "checkpoint": str(checkpoint),
    }
    return validate_response(response, expected_request=request)


def make_error_response(
    request: dict[str, Any] | None,
    *,
    error_type: str,
    error_message: str,
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_NAME,
        "kind": ERROR_KIND,
        "request_id": str(request.get("request_id", "unknown")) if request else "unknown",
        "scope": "atomic_only",
        "task": str(request.get("task", "unknown")) if request else "unknown",
        "episode_id": str(request.get("episode_id", "unknown")) if request else "unknown",
        "step_id": int(request.get("step_id", -1)) if request else -1,
        "error_type": str(error_type),
        "error_message": str(error_message),
    }


def validate_response(
    payload: dict[str, Any], *, expected_request: dict[str, Any] | None = None
) -> dict[str, Any]:
    if payload.get("protocol") != PROTOCOL_NAME:
        raise PolicyProtocolError(f"response protocol 不匹配：{payload.get('protocol')!r}")
    if payload.get("kind") == ERROR_KIND:
        raise PolicyProtocolError(
            f"policy server 返回 {payload.get('error_type', 'error')}: "
            f"{payload.get('error_message', '')}"
        )
    _validate_envelope(payload, RESPONSE_KIND)
    if payload.get("scope") != "atomic_only":
        raise PolicyProtocolError("policy response 必须为 scope=atomic_only")
    for key in ("request_id", "task", "episode_id", "checkpoint"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise PolicyProtocolError(f"response.{key} 必须为非空字符串")
    try:
        step_id = int(payload["step_id"])
        inference_seed = int(payload["inference_seed"])
        inference_seconds = float(payload["inference_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PolicyProtocolError("response step/seed/latency 必须为合法数值") from exc
    if step_id < 0 or inference_seed < 0 or not np.isfinite(inference_seconds) or inference_seconds < 0:
        raise PolicyProtocolError("response step/seed/latency 数值非法")

    actions = np.asarray(payload.get("actions"), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] <= 0 or actions.shape[1] != 12:
        raise PolicyProtocolError(f"response.actions 必须为非空 [Ta,12]，实际={actions.shape}")
    if not np.isfinite(actions).all():
        raise PolicyProtocolError("response.actions 包含 NaN/Inf")

    if expected_request is not None:
        for key in ("request_id", "task", "episode_id"):
            if payload.get(key) != expected_request.get(key):
                raise PolicyProtocolError(
                    f"response.{key} 不对应请求：expected={expected_request.get(key)!r}, "
                    f"actual={payload.get(key)!r}"
                )
        if step_id != int(expected_request["step_id"]):
            raise PolicyProtocolError(
                f"response.step_id 不对应请求：expected={expected_request['step_id']}, actual={step_id}"
            )

    validated = dict(payload)
    validated.update(
        step_id=step_id,
        inference_seed=inference_seed,
        inference_seconds=inference_seconds,
        actions=actions,
    )
    return validated


def encode_message(payload: dict[str, Any]) -> bytes:
    kind = payload.get("kind")
    arrays: dict[str, np.ndarray] = {}
    if kind == REQUEST_KIND:
        validated = validate_request(payload)
        arrays = {"video": validated.pop("video"), "state": validated.pop("state")}
    elif kind == RESPONSE_KIND:
        validated = validate_response(payload)
        arrays = {"actions": validated.pop("actions")}
    elif kind == ERROR_KIND:
        validated = _validate_error_payload(payload)
    else:
        raise PolicyProtocolError(f"未知 message kind：{kind!r}")

    metadata_bytes = json.dumps(
        validated, ensure_ascii=False, sort_keys=True, allow_nan=False
    ).encode("utf-8")
    buffer = io.BytesIO()
    np.savez(
        buffer,
        metadata=np.frombuffer(metadata_bytes, dtype=np.uint8),
        **arrays,
    )
    wire = buffer.getvalue()
    if len(wire) > MAX_WIRE_BYTES:
        raise PolicyProtocolError(f"消息超过 {MAX_WIRE_BYTES} bytes：{len(wire)}")
    return wire


def decode_message(wire: bytes) -> dict[str, Any]:
    if not isinstance(wire, (bytes, bytearray, memoryview)):
        raise PolicyProtocolError("wire message 必须为 bytes")
    raw = bytes(wire)
    if not raw or len(raw) > MAX_WIRE_BYTES:
        raise PolicyProtocolError(f"wire message 大小非法：{len(raw)}")
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if "metadata" not in archive.files:
                raise PolicyProtocolError("wire message 缺少 metadata")
            metadata_bytes = np.asarray(archive["metadata"], dtype=np.uint8).tobytes()
            payload = json.loads(metadata_bytes.decode("utf-8"))
            if not isinstance(payload, dict):
                raise PolicyProtocolError("wire metadata 顶层必须是对象")
            if payload.get("kind") == REQUEST_KIND:
                if set(archive.files) != {"metadata", "video", "state"}:
                    raise PolicyProtocolError(f"request wire arrays 非法：{sorted(archive.files)}")
                payload["video"] = archive["video"].copy()
                payload["state"] = archive["state"].copy()
                return validate_request(payload)
            if payload.get("kind") == RESPONSE_KIND:
                if set(archive.files) != {"metadata", "actions"}:
                    raise PolicyProtocolError(f"response wire arrays 非法：{sorted(archive.files)}")
                payload["actions"] = archive["actions"].copy()
                return validate_response(payload)
            if payload.get("kind") == ERROR_KIND:
                if set(archive.files) != {"metadata"}:
                    raise PolicyProtocolError(f"error wire arrays 非法：{sorted(archive.files)}")
                return _validate_error_payload(payload)
            raise PolicyProtocolError(f"未知 wire message kind：{payload.get('kind')!r}")
    except PolicyProtocolError:
        raise
    except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PolicyProtocolError(f"无法解码 wire message：{exc}") from exc


def _validate_envelope(payload: dict[str, Any], expected_kind: str) -> None:
    if not isinstance(payload, dict):
        raise PolicyProtocolError("protocol payload 必须是 dict")
    if payload.get("protocol") != PROTOCOL_NAME:
        raise PolicyProtocolError(f"protocol 不匹配：{payload.get('protocol')!r}")
    if payload.get("kind") != expected_kind:
        raise PolicyProtocolError(
            f"message kind 不匹配：expected={expected_kind!r}, actual={payload.get('kind')!r}"
        )


def _validate_error_payload(payload: dict[str, Any]) -> dict[str, Any]:
    _validate_envelope(payload, ERROR_KIND)
    if payload.get("scope") != "atomic_only":
        raise PolicyProtocolError("policy error 必须为 scope=atomic_only")
    for key in ("request_id", "task", "episode_id", "error_type", "error_message"):
        if not isinstance(payload.get(key), str):
            raise PolicyProtocolError(f"error.{key} 必须为字符串")
    try:
        step_id = int(payload["step_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PolicyProtocolError("error.step_id 必须为整数") from exc
    validated = dict(payload)
    validated["step_id"] = step_id
    return validated
