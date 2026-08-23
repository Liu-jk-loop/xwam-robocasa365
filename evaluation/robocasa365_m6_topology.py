"""Validation and lookup helpers for versioned M6 evaluation topologies."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation.robocasa365_benchmark import BenchmarkContractError, load_atomic_task_manifest


def _read_json(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkContractError(f"无法读取 M6 evaluation topology：{resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BenchmarkContractError("M6 evaluation topology 顶层必须为对象")
    return resolved, payload


def load_m6_evaluation_topology(
    path: str | Path, repo_root: str | Path
) -> dict[str, Any]:
    resolved_path, payload = _read_json(path)
    root = Path(repo_root).resolve()
    schema_version = int(payload.get("schema_version", -1))
    if schema_version not in {1, 2} or payload.get("scope") != "atomic_only":
        raise BenchmarkContractError(
            "M6 topology 必须为 schema_version=1|2/scope=atomic_only"
        )
    task_manifest_path = root / str(payload.get("task_manifest", ""))
    task_manifest = load_atomic_task_manifest(task_manifest_path)
    expected_tasks = set(task_manifest["tasks"])

    try:
        model_seed = int(payload["model_seed"])
        seed_start = int(payload["seed_start"])
        episodes = int(payload["episodes_per_task"])
        video_steps = int(payload["video_denoise_steps"])
        action_steps = int(payload["action_denoise_steps"])
        replan_steps = int(payload["replan_steps"])
        max_steps = int(payload["max_steps_per_episode"])
        timeout = float(payload["request_timeout_seconds"])
        cfg = float(payload["cfg"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BenchmarkContractError("M6 topology seed/episode/inference 参数非法") from exc
    if model_seed < 0 or seed_start < 0 or episodes <= 0 or max_steps <= 0:
        raise BenchmarkContractError("M6 topology seed 不能为负且 episodes 必须为正")
    if not 0 < action_steps <= video_steps or not 0 < replan_steps <= 32:
        raise BenchmarkContractError("M6 topology 要求 0 < action_steps <= video_steps 且 replan<=32")
    if timeout <= 0 or cfg < 0:
        raise BenchmarkContractError("M6 topology timeout/cfg 非法")

    resource_profile = str(
        payload.get("resource_profile", "4gpu_8server_16client")
    )
    if resource_profile not in {
        "4gpu_8server_16client",
        "4gpu_6server_9client",
    }:
        raise BenchmarkContractError(
            f"未知M6 resource_profile：{resource_profile!r}"
        )
    if schema_version == 2 and resource_profile != "4gpu_8server_16client":
        raise BenchmarkContractError(
            "schema v2 checkpoint comparison只支持4gpu_8server_16client"
        )
    compact_atomic9 = resource_profile == "4gpu_6server_9client"
    expected_server_count = 6 if compact_atomic9 else 8
    expected_client_count = 9 if compact_atomic9 else 16
    expected_gpu_counts = (
        Counter({0: 2, 1: 2, 2: 1, 3: 1})
        if compact_atomic9
        else Counter({0: 2, 1: 2, 2: 2, 3: 2})
    )
    expected_server_client_counts = (
        Counter({0: 2, 1: 1, 2: 2, 3: 1, 4: 2, 5: 1})
        if compact_atomic9
        else Counter({index: 2 for index in range(8)})
    )

    comparison_groups: dict[str, dict[str, Any]] = {}
    if schema_version == 2:
        raw_groups = payload.get("comparison_groups")
        if not isinstance(raw_groups, dict) or len(raw_groups) != 2:
            raise BenchmarkContractError("schema v2 topology必须包含两个comparison group")
        for group_name, raw_group in raw_groups.items():
            if not isinstance(group_name, str) or not group_name:
                raise BenchmarkContractError("comparison group名称必须为非空字符串")
            if not isinstance(raw_group, dict):
                raise BenchmarkContractError(f"comparison group非法：{group_name}")
            try:
                checkpoint_step = int(raw_group["checkpoint_step"])
                gpus = [int(value) for value in raw_group["gpus"]]
                server_ids = [int(value) for value in raw_group["server_ids"]]
                client_ids = [int(value) for value in raw_group["client_ids"]]
            except (KeyError, TypeError, ValueError) as exc:
                raise BenchmarkContractError(
                    f"comparison group字段非法：{group_name}"
                ) from exc
            if checkpoint_step <= 0:
                raise BenchmarkContractError("comparison checkpoint step必须为正")
            if len(gpus) != 2 or len(set(gpus)) != 2:
                raise BenchmarkContractError("每个comparison group必须独占两张GPU")
            if len(server_ids) != 4 or len(set(server_ids)) != 4:
                raise BenchmarkContractError("每个comparison group必须包含四个server")
            if len(client_ids) != 8 or len(set(client_ids)) != 8:
                raise BenchmarkContractError("每个comparison group必须包含八个client")
            comparison_groups[group_name] = {
                **raw_group,
                "checkpoint_step": checkpoint_step,
                "gpus": gpus,
                "server_ids": server_ids,
                "client_ids": client_ids,
            }
        all_group_gpus = [
            gpu for group in comparison_groups.values() for gpu in group["gpus"]
        ]
        all_group_servers = [
            server_id
            for group in comparison_groups.values()
            for server_id in group["server_ids"]
        ]
        all_group_clients = [
            client_id
            for group in comparison_groups.values()
            for client_id in group["client_ids"]
        ]
        if sorted(all_group_gpus) != list(range(4)):
            raise BenchmarkContractError("两个comparison group必须恰好覆盖GPU 0..3")
        if sorted(all_group_servers) != list(range(8)):
            raise BenchmarkContractError("两个comparison group必须恰好覆盖server 0..7")
        if sorted(all_group_clients) != list(range(16)):
            raise BenchmarkContractError("两个comparison group必须恰好覆盖client 0..15")

    raw_servers = payload.get("servers")
    raw_clients = payload.get("clients")
    if not isinstance(raw_servers, list) or len(raw_servers) != expected_server_count:
        raise BenchmarkContractError(
            f"M6 topology 必须恰好包含 {expected_server_count} 个 server"
        )
    if not isinstance(raw_clients, list) or len(raw_clients) != expected_client_count:
        raise BenchmarkContractError(
            f"M6 topology 必须恰好包含 {expected_client_count} 个 client"
        )

    servers: dict[int, dict[str, Any]] = {}
    ports: set[int] = set()
    gpu_counts: Counter[int] = Counter()
    for raw in raw_servers:
        if not isinstance(raw, dict):
            raise BenchmarkContractError("server entry 必须为对象")
        try:
            server_id = int(raw["server_id"])
            gpu = int(raw["gpu"])
            frontend = int(raw["frontend_port"])
            backend = int(raw["backend_port"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BenchmarkContractError(f"server entry 非法：{raw}") from exc
        if server_id in servers or server_id not in range(expected_server_count):
            raise BenchmarkContractError(
                f"server_id 必须唯一且位于0..{expected_server_count - 1}：{server_id}"
            )
        if gpu not in range(4):
            raise BenchmarkContractError(f"server GPU 必须位于0..3：{gpu}")
        if not 1 <= frontend <= 65535 or not 1 <= backend <= 65535 or frontend == backend:
            raise BenchmarkContractError(f"server port 非法：{raw}")
        if frontend in ports or backend in ports:
            raise BenchmarkContractError(f"server port 重复：{raw}")
        comparison_group = raw.get("comparison_group")
        if schema_version == 2:
            if comparison_group not in comparison_groups:
                raise BenchmarkContractError(
                    f"server comparison group非法：{raw}"
                )
            group = comparison_groups[str(comparison_group)]
            if server_id not in group["server_ids"] or gpu not in group["gpus"]:
                raise BenchmarkContractError(
                    f"server不属于声明的comparison group资源：{raw}"
                )
        elif comparison_group is not None:
            raise BenchmarkContractError("schema v1 server不能声明comparison group")
        ports.update((frontend, backend))
        gpu_counts[gpu] += 1
        servers[server_id] = {
            "server_id": server_id,
            "gpu": gpu,
            "frontend_port": frontend,
            "backend_port": backend,
            "comparison_group": comparison_group,
        }
    if (
        set(servers) != set(range(expected_server_count))
        or gpu_counts != expected_gpu_counts
    ):
        raise BenchmarkContractError(
            f"M6 topology server/GPU映射不符合{resource_profile}"
        )

    clients: dict[int, dict[str, Any]] = {}
    assigned_tasks: list[str] = []
    server_client_counts: Counter[int] = Counter()
    for raw in raw_clients:
        if not isinstance(raw, dict):
            raise BenchmarkContractError("client entry 必须为对象")
        try:
            client_id = int(raw["client_id"])
            server_id = int(raw["server_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BenchmarkContractError(f"client entry 非法：{raw}") from exc
        if client_id in clients or client_id not in range(expected_client_count):
            raise BenchmarkContractError(
                f"client_id 必须唯一且位于0..{expected_client_count - 1}：{client_id}"
            )
        if server_id not in servers:
            raise BenchmarkContractError(f"client 引用了未知 server：{server_id}")
        comparison_group = raw.get("comparison_group")
        if schema_version == 2:
            if comparison_group not in comparison_groups:
                raise BenchmarkContractError(
                    f"client comparison group非法：{raw}"
                )
            group = comparison_groups[str(comparison_group)]
            if (
                client_id not in group["client_ids"]
                or servers[server_id]["comparison_group"] != comparison_group
            ):
                raise BenchmarkContractError(
                    f"client与server不属于同一comparison group：{raw}"
                )
        elif comparison_group is not None:
            raise BenchmarkContractError("schema v1 client不能声明comparison group")
        raw_tasks = raw.get("tasks")
        valid_task_counts = {1} if compact_atomic9 else {1, 2}
        if not isinstance(raw_tasks, list) or len(raw_tasks) not in valid_task_counts:
            raise BenchmarkContractError(
                "compact Atomic9每个client必须执行1个任务"
                if compact_atomic9
                else "每个 client 必须串行执行 1 或 2 个任务"
            )
        task_entries = []
        for task_entry in raw_tasks:
            if not isinstance(task_entry, dict) or task_entry.get("name") not in expected_tasks:
                raise BenchmarkContractError(f"client task 不属于 Atomic-Seen 18：{task_entry}")
            name = str(task_entry["name"])
            reference = float(task_entry["fastwam_reference_success_percent"])
            if reference < 0 or reference > 100:
                raise BenchmarkContractError(f"FastWAM reference success 非法：{task_entry}")
            historical_xwam = task_entry.get("historical_xwam_success_percent")
            if historical_xwam is not None:
                historical_xwam = float(historical_xwam)
                if historical_xwam < 0 or historical_xwam > 100:
                    raise BenchmarkContractError(
                        f"historical X-WAM success非法：{task_entry}"
                    )
            task_entries.append(
                {
                    "name": name,
                    "fastwam_reference_success_percent": reference,
                    "historical_xwam_success_percent": historical_xwam,
                }
            )
            assigned_tasks.append(name)
        server_client_counts[server_id] += 1
        clients[client_id] = {
            "client_id": client_id,
            "server_id": server_id,
            "tasks": task_entries,
            "comparison_group": comparison_group,
        }
    if (
        set(clients) != set(range(expected_client_count))
        or server_client_counts != expected_server_client_counts
    ):
        raise BenchmarkContractError(
            f"M6 topology client/server映射不符合{resource_profile}"
        )
    if schema_version == 1:
        if (
            len(assigned_tasks) != len(expected_tasks)
            or set(assigned_tasks) != expected_tasks
            or len(set(assigned_tasks)) != len(expected_tasks)
        ):
            raise BenchmarkContractError("M6 topology 必须不重不漏地覆盖任务清单")
    else:
        for group_name in comparison_groups:
            group_tasks = [
                task["name"]
                for client in clients.values()
                if client["comparison_group"] == group_name
                for task in client["tasks"]
            ]
            if (
                len(group_tasks) != len(expected_tasks)
                or set(group_tasks) != expected_tasks
                or len(set(group_tasks)) != len(expected_tasks)
            ):
                raise BenchmarkContractError(
                    f"comparison group {group_name}必须各自不重不漏覆盖任务清单"
                )

    resolved = dict(payload)
    resolved.update(
        model_seed=model_seed,
        seed_start=seed_start,
        episodes_per_task=episodes,
        video_denoise_steps=video_steps,
        action_denoise_steps=action_steps,
        replan_steps=replan_steps,
        max_steps_per_episode=max_steps,
        request_timeout_seconds=timeout,
        cfg=cfg,
        servers=servers,
        clients=clients,
        comparison_groups=comparison_groups,
        resource_profile=resource_profile,
        server_count=expected_server_count,
        client_count=expected_client_count,
        resolved_path=str(resolved_path),
        resolved_task_manifest=str(task_manifest_path.resolve()),
        task_horizons={task: int(task_manifest["horizons"][task]) for task in expected_tasks},
    )
    return resolved


def tasks_for_server(topology: dict[str, Any], server_id: int) -> list[str]:
    return [
        task["name"]
        for client in topology["clients"].values()
        if int(client["server_id"]) == int(server_id)
        for task in client["tasks"]
    ]
