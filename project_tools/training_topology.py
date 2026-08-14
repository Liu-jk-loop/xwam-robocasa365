"""Dependency-free GPU topology validation for X-WAM training entry points."""

from __future__ import annotations

from typing import Any


def resolve_optimizer_backend(config: Any) -> str:
    """Keep CPUAdam restricted to runs that explicitly enable optimizer offload."""
    if bool(config.get("deepspeed_offload_optimizer", False)):
        return "deepspeed_cpu_adam"
    return "torch_adamw"


def resolve_cpu_adam_options(config: Any) -> dict[str, bool]:
    """Resolve CPUAdam state precision while keeping FP32 as the safe default."""
    fp32_optimizer_states = config.get("deepspeed_fp32_optimizer_states", True)
    if not isinstance(fp32_optimizer_states, bool):
        raise ValueError("deepspeed_fp32_optimizer_states 必须是布尔值")
    return {"fp32_optimizer_states": fp32_optimizer_states}


def resolve_deepspeed_options(config: Any) -> dict[str, Any]:
    """Resolve memory-sensitive DeepSpeed options without importing Torch/Lightning."""
    stage = int(config.get("deepspeed_stage", 2))
    if stage not in {1, 2, 3}:
        raise ValueError(f"DeepSpeed stage 只允许 1、2 或 3，当前为 {stage}")

    bucket_size = int(config.get("deepspeed_bucket_size", 500_000_000))
    if bucket_size <= 0:
        raise ValueError(
            f"DeepSpeed bucket size 必须为正整数，当前为 {bucket_size}"
        )

    offload_device = str(
        config.get("deepspeed_offload_optimizer_device", "cpu")
    ).lower()
    if offload_device not in {"cpu", "nvme"}:
        raise ValueError(
            "DeepSpeed optimizer offload device 只允许 cpu 或 nvme，"
            f"当前为 {offload_device}"
        )

    exclude_frozen_parameters = config.get(
        "deepspeed_exclude_frozen_parameters", False
    )
    if not isinstance(exclude_frozen_parameters, bool):
        raise ValueError("deepspeed_exclude_frozen_parameters 必须是布尔值")

    return {
        "stage": stage,
        "offload_optimizer": bool(
            config.get("deepspeed_offload_optimizer", False)
        ),
        "offload_optimizer_device": offload_device,
        "pin_memory": bool(config.get("deepspeed_pin_memory", False)),
        "overlap_comm": bool(config.get("deepspeed_overlap_comm", True)),
        "allgather_bucket_size": bucket_size,
        "reduce_bucket_size": bucket_size,
        "exclude_frozen_parameters": exclude_frozen_parameters,
    }


def resolve_distributed_timeout_minutes(config: Any) -> int:
    """Resolve the process-group timeout without importing Torch/Lightning."""
    value = config.get("distributed_timeout_minutes", 30)
    if isinstance(value, bool):
        raise ValueError("distributed_timeout_minutes 必须是正整数")
    try:
        timeout_minutes = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("distributed_timeout_minutes 必须是正整数") from exc
    if timeout_minutes <= 0:
        raise ValueError("distributed_timeout_minutes 必须是正整数")
    return timeout_minutes


def resolve_training_topology(
    *,
    visible_devices: int,
    requested_devices: Any = "auto",
    world_size: int | str | None = None,
) -> dict[str, int | str]:
    visible = int(visible_devices)
    devices_per_node = visible if requested_devices == "auto" else int(requested_devices)
    if devices_per_node <= 0 or devices_per_node > visible:
        raise ValueError(
            f"GPU 配置无效：requested={requested_devices}, visible={visible}"
        )
    resolved_world_size = devices_per_node if world_size is None else int(world_size)
    if resolved_world_size <= 0 or resolved_world_size % devices_per_node != 0:
        raise ValueError(
            "WORLD_SIZE 必须是每节点设备数的正整数倍："
            f"world_size={resolved_world_size}, devices_per_node={devices_per_node}"
        )
    return {
        "trainer_devices": requested_devices,
        "devices_per_node": devices_per_node,
        "visible_devices": visible,
        "world_size": resolved_world_size,
        "num_nodes": resolved_world_size // devices_per_node,
    }
