"""Dependency-free contract for adapting public X-WAM checkpoints to PandaOmron."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, Sequence


ACTION_REMAP_KEYS = (
    "model.action_encoder.0.weight",
    "model.action_decoder.2.weight",
    "model.action_decoder.2.bias",
)
PROPRIO_REINITIALIZE_KEYS = (
    "model.proprio_encoder.0.weight",
    "model.proprio_encoder.0.bias",
    "model.proprio_decoder.2.weight",
    "model.proprio_decoder.2.bias",
)
RGB_ONLY_DISCARD_PREFIXES = (
    "model.extra_blocks.",
    "model.extra_heads.",
)


def resolve_xwam_checkpoint(path: str | Path) -> Path:
    """Resolve the public checkpoint root, DeepSpeed directory, or state file."""

    requested = Path(path).expanduser().resolve()
    candidates = (
        requested,
        requested / "checkpoint" / "mp_rank_00_model_states.pt",
        requested / "pretrained" / "checkpoints" / "last.ckpt" / "checkpoint" / "mp_rank_00_model_states.pt",
        requested / "checkpoints" / "last.ckpt" / "checkpoint" / "mp_rank_00_model_states.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "未找到 mp_rank_00_model_states.pt；检查 checkpoint 根目录或直接传入文件："
        + str(requested)
    )


def _shape_tuple(shape: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(dim) for dim in shape)


def _validate_action_boundary_shapes(
    key: str,
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    source_slice: tuple[int, int],
    target_slice: tuple[int, int],
) -> str | None:
    source_width = source_slice[1] - source_slice[0]
    target_width = target_slice[1] - target_slice[0]
    if source_width <= 0 or source_width != target_width:
        return f"动作映射切片宽度无效：source={source_slice}, target={target_slice}"

    if key == "model.action_encoder.0.weight":
        valid = (
            len(source_shape) == len(target_shape) == 2
            and source_shape[0] == target_shape[0]
            and source_shape[1] == 14
            and target_shape[1] == 12
            and source_slice[1] <= source_shape[1]
            and target_slice[1] <= target_shape[1]
        )
    elif key == "model.action_decoder.2.weight":
        valid = (
            len(source_shape) == len(target_shape) == 2
            and source_shape[1] == target_shape[1]
            and source_shape[0] == 14
            and target_shape[0] == 12
            and source_slice[1] <= source_shape[0]
            and target_slice[1] <= target_shape[0]
        )
    else:
        valid = (
            len(source_shape) == len(target_shape) == 1
            and source_shape[0] == 14
            and target_shape[0] == 12
            and source_slice[1] <= source_shape[0]
            and target_slice[1] <= target_shape[0]
        )
    if valid:
        return None
    return f"动作边界 shape 与声明映射不兼容：source={source_shape}, target={target_shape}"


def plan_xwam_checkpoint_adaptation(
    source_shapes: Mapping[str, Sequence[int]],
    target_shapes: Mapping[str, Sequence[int]],
    *,
    source_arm_slice: Sequence[int] = (0, 7),
    target_arm_slice: Sequence[int] = (5, 12),
    discard_prefixes: Sequence[str] = RGB_ONLY_DISCARD_PREFIXES,
) -> dict[str, Any]:
    """Classify every source/target key; undeclared differences are blocking errors."""

    source_slice = tuple(int(value) for value in source_arm_slice)
    target_slice = tuple(int(value) for value in target_arm_slice)
    if len(source_slice) != 2 or len(target_slice) != 2:
        raise ValueError("source_arm_slice/target_arm_slice 必须各包含两个整数")

    source = {str(key): _shape_tuple(shape) for key, shape in source_shapes.items()}
    target = {str(key): _shape_tuple(shape) for key, shape in target_shapes.items()}
    exact_load: list[str] = []
    action_remap: list[str] = []
    reinitialize: list[str] = []
    discard_source: list[str] = []
    missing_source: list[str] = []
    unexpected_source: list[str] = []
    shape_errors: list[str] = []

    for key, target_shape in target.items():
        if key in PROPRIO_REINITIALIZE_KEYS:
            reinitialize.append(key)
            continue
        if key in ACTION_REMAP_KEYS:
            if key not in source:
                missing_source.append(key)
                continue
            error = _validate_action_boundary_shapes(
                key, source[key], target_shape, source_slice, target_slice
            )
            if error:
                shape_errors.append(f"{key}: {error}")
            else:
                action_remap.append(key)
            continue
        if key not in source:
            missing_source.append(key)
        elif source[key] != target_shape:
            shape_errors.append(
                f"{key}: 未声明的 shape mismatch：source={source[key]}, target={target_shape}"
            )
        else:
            exact_load.append(key)

    target_keys = set(target)
    ignored_target_keys = set(PROPRIO_REINITIALIZE_KEYS)
    for key in source:
        if key in target_keys or key in ignored_target_keys:
            continue
        if any(key.startswith(prefix) for prefix in discard_prefixes):
            discard_source.append(key)
        else:
            unexpected_source.append(key)

    errors = [
        *[f"目标参数缺少 source：{key}" for key in missing_source],
        *[f"source 含未声明参数：{key}" for key in unexpected_source],
        *shape_errors,
    ]
    initialized = sorted(reinitialize) + [
        f"model.action_encoder.0.weight[:,0:{target_slice[0]}]",
        f"model.action_decoder.2.weight[0:{target_slice[0]},:]",
        f"model.action_decoder.2.bias[0:{target_slice[0]}]",
    ]
    return {
        "source_arm_slice": list(source_slice),
        "target_arm_slice": list(target_slice),
        "exact_load": sorted(exact_load),
        "action_remap": sorted(action_remap),
        "reinitialize": sorted(reinitialize),
        "discard_source": sorted(discard_source),
        "missing_source": sorted(missing_source),
        "unexpected_source": sorted(unexpected_source),
        "shape_errors": sorted(shape_errors),
        "errors": errors,
        "loaded": sorted(exact_load),
        "remapped": sorted(action_remap),
        "initialized": initialized,
        "missing": sorted(missing_source),
        "unexpected": sorted(unexpected_source),
        "ok": not errors,
    }


def remap_action_boundary(
    key: str,
    source: Any,
    target: Any,
    *,
    source_arm_slice: Sequence[int] = (0, 7),
    target_arm_slice: Sequence[int] = (5, 12),
) -> Any:
    """Copy the legacy 7D arm boundary into PandaOmron action[5:12]."""

    if key not in ACTION_REMAP_KEYS:
        raise KeyError(f"不是已声明的动作边界参数：{key}")
    source_start, source_end = (int(value) for value in source_arm_slice)
    target_start, target_end = (int(value) for value in target_arm_slice)
    if isinstance(target, list):
        output = copy.deepcopy(target)
        if key == "model.action_encoder.0.weight":
            for row_index, row in enumerate(output):
                row[target_start:target_end] = source[row_index][source_start:source_end]
        else:
            output[target_start:target_end] = copy.deepcopy(source[source_start:source_end])
        return output
    output = target.clone() if hasattr(target, "clone") else target.copy()
    if key == "model.action_encoder.0.weight":
        output[:, target_start:target_end] = source[:, source_start:source_end]
    else:
        output[target_start:target_end] = source[source_start:source_end]
    return output
