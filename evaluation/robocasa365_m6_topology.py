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
    if payload.get("schema_version") != 1 or payload.get("scope") != "atomic_only":
        raise BenchmarkContractError("M6 topology 必须为 schema_version=1/scope=atomic_only")
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
        raise BenchmarkContractError(f"未知M6 resource_profile：{resource_profile!r}")
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

    servers: dict[int, dict[str, int]] = {}
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
        ports.update((frontend, backend))
        gpu_counts[gpu] += 1
        servers[server_id] = {
            "server_id": server_id,
            "gpu": gpu,
            "frontend_port": frontend,
            "backend_port": backend,
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
            task_entries.append({"name": name, "fastwam_reference_success_percent": reference})
            assigned_tasks.append(name)
        server_client_counts[server_id] += 1
        clients[client_id] = {
            "client_id": client_id,
            "server_id": server_id,
            "tasks": task_entries,
        }
    if (
        set(clients) != set(range(expected_client_count))
        or server_client_counts != expected_server_client_counts
    ):
        raise BenchmarkContractError(
            f"M6 topology client/server映射不符合{resource_profile}"
        )
    if (
        len(assigned_tasks) != len(expected_tasks)
        or set(assigned_tasks) != expected_tasks
        or len(set(assigned_tasks)) != len(expected_tasks)
    ):
        raise BenchmarkContractError("M6 topology 必须不重不漏地覆盖任务清单")

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
