"""Dependency-free GPU topology validation for X-WAM training entry points."""

from __future__ import annotations

from typing import Any


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
