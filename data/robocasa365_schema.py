"""Schema validation and NumPy codecs for RoboCasa365 PandaOmron tensors."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from data.robocasa365_contract import DatasetContractError, resolve_lerobot_root


SUPPORTED_NORMALIZATION = {"quantile", "identity_unit", "categorical_sign"}


@dataclass(frozen=True)
class Component:
    name: str
    start: int
    end: int
    normalization: str
    metadata: dict[str, Any]

    @property
    def size(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class TensorSchema:
    name: str
    original_key: str
    dimension: int
    components: tuple[Component, ...]


@dataclass(frozen=True)
class PandaOmronSchema:
    schema_id: str
    schema_version: int
    robot_type: str
    state: TensorSchema
    action: TensorSchema
    videos: tuple[dict[str, Any], ...]
    checkpoint_mapping: dict[str, Any]
    payload: dict[str, Any]

    @property
    def canonical_sha256(self) -> str:
        canonical = json.dumps(
            self.payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Statistics:
    q01: np.ndarray
    q99: np.ndarray
    minimum: np.ndarray | None
    maximum: np.ndarray | None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetContractError(f"无法读取 JSON：{path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DatasetContractError(f"JSON 顶层必须是对象：{path}")
    return payload


def _parse_tensor_schema(name: str, payload: Any) -> TensorSchema:
    if not isinstance(payload, dict):
        raise DatasetContractError(f"schema.{name} 必须是对象")
    original_key = payload.get("original_key")
    dimension = payload.get("dimension")
    raw_components = payload.get("components")
    if not isinstance(original_key, str) or not original_key:
        raise DatasetContractError(f"schema.{name}.original_key 必须是非空字符串")
    try:
        dimension = int(dimension)
    except (TypeError, ValueError) as exc:
        raise DatasetContractError(f"schema.{name}.dimension 必须是整数") from exc
    if not isinstance(raw_components, list) or not raw_components:
        raise DatasetContractError(f"schema.{name}.components 必须是非空数组")

    components: list[Component] = []
    expected_start = 0
    seen_names: set[str] = set()
    for raw in raw_components:
        if not isinstance(raw, dict):
            raise DatasetContractError(f"schema.{name}.components 元素必须是对象")
        component_name = raw.get("name")
        normalization = raw.get("normalization")
        try:
            start, end = int(raw.get("start")), int(raw.get("end"))
        except (TypeError, ValueError) as exc:
            raise DatasetContractError(
                f"schema.{name} 组件切片必须是整数：{raw}"
            ) from exc
        if not isinstance(component_name, str) or not component_name:
            raise DatasetContractError(f"schema.{name} 组件缺少名称：{raw}")
        if component_name in seen_names:
            raise DatasetContractError(f"schema.{name} 组件名称重复：{component_name}")
        if start != expected_start or end <= start:
            raise DatasetContractError(
                f"schema.{name}.{component_name} 切片必须连续且非空，期望 start={expected_start}，实际={start}:{end}"
            )
        if normalization not in SUPPORTED_NORMALIZATION:
            raise DatasetContractError(
                f"schema.{name}.{component_name} normalization 不支持：{normalization!r}"
            )
        seen_names.add(component_name)
        expected_start = end
        components.append(
            Component(
                name=component_name,
                start=start,
                end=end,
                normalization=normalization,
                metadata={
                    key: value
                    for key, value in raw.items()
                    if key not in {"name", "start", "end", "normalization"}
                },
            )
        )
    if expected_start != dimension:
        raise DatasetContractError(
            f"schema.{name} 组件只覆盖到 {expected_start}，dimension={dimension}"
        )
    return TensorSchema(
        name=name,
        original_key=original_key,
        dimension=dimension,
        components=tuple(components),
    )


def load_panda_omron_schema(path: str | Path) -> PandaOmronSchema:
    schema_path = Path(path).expanduser().resolve()
    payload = _read_json(schema_path)
    if payload.get("scope") != "atomic_only":
        raise DatasetContractError(f"M2 schema 必须为 scope=atomic_only：{schema_path}")
    videos = payload.get("videos")
    if (
        not isinstance(videos, list)
        or len(videos) != 3
        or not all(isinstance(item, dict) for item in videos)
    ):
        raise DatasetContractError(f"M2 schema 必须配置三路 video：{schema_path}")
    video_keys = [item.get("original_key") for item in videos]
    if len(set(video_keys)) != 3 or not all(
        isinstance(key, str) and key for key in video_keys
    ):
        raise DatasetContractError(
            f"M2 schema video original_key 必须是三个不重复字符串：{schema_path}"
        )
    checkpoint_mapping = payload.get("checkpoint_mapping")
    if not isinstance(checkpoint_mapping, dict):
        raise DatasetContractError(f"M2 schema 缺少 checkpoint_mapping：{schema_path}")
    return PandaOmronSchema(
        schema_id=str(payload.get("schema_id", schema_path.stem)),
        schema_version=int(payload.get("schema_version", 0)),
        robot_type=str(payload.get("robot_type", "")),
        state=_parse_tensor_schema("state", payload.get("state")),
        action=_parse_tensor_schema("action", payload.get("action")),
        videos=tuple(dict(item) for item in videos),
        checkpoint_mapping=dict(checkpoint_mapping),
        payload=payload,
    )


def _canonical_modality_contract(payload: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for tensor_name in ("state", "action"):
        raw = payload.get(tensor_name)
        if not isinstance(raw, dict):
            raise DatasetContractError(f"modality.json 缺少 {tensor_name} 对象")
        output[tensor_name] = [
            {
                "name": name,
                "original_key": spec.get("original_key"),
                "start": spec.get("start"),
                "end": spec.get("end"),
            }
            for name, spec in raw.items()
            if isinstance(spec, dict)
        ]
    raw_videos = payload.get("video")
    if not isinstance(raw_videos, dict):
        raise DatasetContractError("modality.json 缺少 video 对象")
    output["video"] = sorted(
        [
            {"name": name, "original_key": spec.get("original_key")}
            for name, spec in raw_videos.items()
            if isinstance(spec, dict)
        ],
        key=lambda item: str(item["original_key"]),
    )
    return output


def expected_modality_contract(schema: PandaOmronSchema) -> dict[str, Any]:
    return {
        "state": [
            {
                "name": component.name,
                "original_key": schema.state.original_key,
                "start": component.start,
                "end": component.end,
            }
            for component in schema.state.components
        ],
        "action": [
            {
                "name": component.name,
                "original_key": schema.action.original_key,
                "start": component.start,
                "end": component.end,
            }
            for component in schema.action.components
        ],
        "video": sorted(
            [
                {"name": item["name"], "original_key": item["original_key"]}
                for item in schema.videos
            ],
            key=lambda item: item["original_key"],
        ),
    }


def validate_dataset_modality(
    dataset_path: str | Path, schema: PandaOmronSchema
) -> dict[str, Any]:
    root = resolve_lerobot_root(dataset_path)
    path = root / "meta" / "modality.json"
    raw_bytes = path.read_bytes()
    payload = _read_json(path)
    actual = _canonical_modality_contract(payload)
    expected = expected_modality_contract(schema)
    if actual != expected:
        raise DatasetContractError(
            "真实 modality.json 与版本化 PandaOmron schema 不一致。"
            f"\nexpected={json.dumps(expected, ensure_ascii=False)}"
            f"\nactual={json.dumps(actual, ensure_ascii=False)}"
        )
    return {
        "path": str(path),
        "file_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "contract": actual,
    }


def _stat_vector(
    block: dict[str, Any], key: str, dimension: int, path: Path
) -> np.ndarray:
    value = block.get(key)
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DatasetContractError(f"stats {key} 无法转换为数值：{path}") from exc
    if array.shape != (dimension,) or not np.isfinite(array).all():
        raise DatasetContractError(
            f"stats {key} 应为有限向量 [{dimension}]，实际为 {array.shape}：{path}"
        )
    return array


def load_statistics_file(
    stats_path: str | Path, tensor_schema: TensorSchema
) -> Statistics:
    path = Path(stats_path).expanduser().resolve()
    payload = _read_json(path)
    block = payload.get(tensor_schema.original_key)
    if not isinstance(block, dict):
        raise DatasetContractError(
            f"stats.json 缺少 {tensor_schema.original_key!r}：{path}"
        )
    q01 = _stat_vector(block, "q01", tensor_schema.dimension, path)
    q99 = _stat_vector(block, "q99", tensor_schema.dimension, path)
    if np.any(q99 < q01):
        raise DatasetContractError(f"stats q99 存在小于 q01 的维度：{path}")
    minimum = (
        _stat_vector(block, "min", tensor_schema.dimension, path)
        if "min" in block
        else None
    )
    maximum = (
        _stat_vector(block, "max", tensor_schema.dimension, path)
        if "max" in block
        else None
    )
    return Statistics(q01=q01, q99=q99, minimum=minimum, maximum=maximum)


def load_statistics(
    dataset_path: str | Path, tensor_schema: TensorSchema
) -> Statistics:
    root = resolve_lerobot_root(dataset_path)
    return load_statistics_file(root / "meta" / "stats.json", tensor_schema)


class PandaOmronTensorCodec:
    """Normalize state/action by named components and preserve the full 12D action."""

    def __init__(
        self,
        schema: PandaOmronSchema,
        state_stats: Statistics,
        action_stats: Statistics,
    ):
        self.schema = schema
        self.state_stats = state_stats
        self.action_stats = action_stats

    @classmethod
    def from_dataset(
        cls,
        dataset_path: str | Path,
        schema_path: str | Path,
        *,
        statistics_path: str | Path | None = None,
    ) -> "PandaOmronTensorCodec":
        schema = load_panda_omron_schema(schema_path)
        validate_dataset_modality(dataset_path, schema)
        if statistics_path is None:
            state_stats = load_statistics(dataset_path, schema.state)
            action_stats = load_statistics(dataset_path, schema.action)
        else:
            state_stats = load_statistics_file(statistics_path, schema.state)
            action_stats = load_statistics_file(statistics_path, schema.action)
        return cls(
            schema=schema,
            state_stats=state_stats,
            action_stats=action_stats,
        )

    @staticmethod
    def _validate_values(values: Any, tensor_schema: TensorSchema) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim < 1 or array.shape[-1] != tensor_schema.dimension:
            raise ValueError(
                f"{tensor_schema.name} 最后一维应为 {tensor_schema.dimension}，实际为 {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{tensor_schema.name} 包含 NaN/Inf")
        return array

    @staticmethod
    def _normalize(
        values: Any,
        tensor_schema: TensorSchema,
        stats: Statistics,
        *,
        clip: bool,
    ) -> np.ndarray:
        raw = PandaOmronTensorCodec._validate_values(values, tensor_schema)
        output = np.empty_like(raw, dtype=np.float32)
        for component in tensor_schema.components:
            sl = slice(component.start, component.end)
            block = raw[..., sl]
            if component.normalization == "quantile":
                lower, upper = stats.q01[sl], stats.q99[sl]
                span = upper - lower
                stable = span > 1e-8
                normalized = np.zeros_like(block, dtype=np.float32)
                if np.any(stable):
                    normalized[..., stable] = (
                        2.0 * (block[..., stable] - lower[stable]) / span[stable] - 1.0
                    )
                output[..., sl] = normalized
            else:
                output[..., sl] = block
        return np.clip(output, -1.0, 1.0) if clip else output

    @staticmethod
    def _denormalize(
        values: Any, tensor_schema: TensorSchema, stats: Statistics
    ) -> np.ndarray:
        normalized = PandaOmronTensorCodec._validate_values(values, tensor_schema)
        output = np.empty_like(normalized, dtype=np.float32)
        for component in tensor_schema.components:
            sl = slice(component.start, component.end)
            block = normalized[..., sl]
            if component.normalization == "quantile":
                lower, upper = stats.q01[sl], stats.q99[sl]
                span = upper - lower
                stable = span > 1e-8
                raw = np.broadcast_to(lower, block.shape).astype(np.float32).copy()
                if np.any(stable):
                    raw[..., stable] = (block[..., stable] + 1.0) * 0.5 * span[
                        stable
                    ] + lower[stable]
                output[..., sl] = raw
            else:
                output[..., sl] = block
        return output

    def encode_state(self, values: Any, *, clip: bool = True) -> np.ndarray:
        return self._normalize(values, self.schema.state, self.state_stats, clip=clip)

    def decode_state(self, values: Any) -> np.ndarray:
        return self._denormalize(values, self.schema.state, self.state_stats)

    def encode_action(self, values: Any, *, clip: bool = True) -> np.ndarray:
        return self._normalize(values, self.schema.action, self.action_stats, clip=clip)

    def decode_action(
        self, values: Any, *, discretize_control_mode: bool = False
    ) -> np.ndarray:
        output = self._denormalize(values, self.schema.action, self.action_stats)
        if discretize_control_mode:
            component = next(
                item
                for item in self.schema.action.components
                if item.name == "control_mode"
            )
            sl = slice(component.start, component.end)
            output[..., sl] = np.where(output[..., sl] >= 0.0, 1.0, -1.0)
        return output

    def component_summary(
        self, values: Any, tensor_name: str
    ) -> dict[str, dict[str, Any]]:
        tensor_schema = (
            self.schema.state if tensor_name == "state" else self.schema.action
        )
        array = self._validate_values(values, tensor_schema)
        return {
            component.name: {
                "slice": [component.start, component.end],
                "normalization": component.normalization,
                "min": float(array[..., component.start : component.end].min()),
                "max": float(array[..., component.start : component.end].max()),
            }
            for component in tensor_schema.components
        }
