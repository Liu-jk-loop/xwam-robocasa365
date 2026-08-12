#!/usr/bin/env python3
"""Aggregate and validate all M6 client summaries."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.robocasa365_m6_topology import (  # noqa: E402
    load_m6_evaluation_topology,
)
from project_tools.training_run import write_json_atomic  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须为对象：{path}")
    return payload


def aggregate(
    *,
    topology_path: str | Path,
    output_root: str | Path,
    episodes_per_task: int,
    eval_id: str | None = None,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    topology = load_m6_evaluation_topology(topology_path, REPO_ROOT)
    root = Path(output_root).expanduser().resolve()
    errors: list[str] = []
    task_rows: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    for client_id in range(16):
        summary_path = root / f"client_{client_id:02d}" / "client_summary.json"
        if not summary_path.is_file():
            errors.append(f"缺少 client summary：{summary_path}")
            continue
        summary = _read_json(summary_path)
        if summary.get("result") != "pass":
            errors.append(f"client {client_id} 尚未完成：{summary.get('result')}")
        expected_client = topology["clients"][client_id]
        expected_task_names = [row["name"] for row in expected_client["tasks"]]
        actual_task_names = [str(row.get("task")) for row in summary.get("tasks", [])]
        if summary.get("client_id") != client_id:
            errors.append(f"client summary id不匹配：expected={client_id}")
        if summary.get("server_id") != expected_client["server_id"]:
            errors.append(f"client {client_id} server映射漂移")
        if actual_task_names != expected_task_names:
            errors.append(
                f"client {client_id} task顺序漂移："
                f"expected={expected_task_names}, actual={actual_task_names}"
            )
        if summary.get("model_seed") != topology["model_seed"]:
            errors.append(f"client {client_id} model seed漂移")
        if summary.get("seed_start") != topology["seed_start"]:
            errors.append(f"client {client_id} task seed漂移")
        for row in summary.get("tasks", []):
            task = str(row.get("task"))
            if task in seen_tasks:
                errors.append(f"task summary 重复：{task}")
                continue
            seen_tasks.add(task)
            completed = int(row.get("episodes_completed", -1))
            if completed != episodes_per_task:
                errors.append(
                    f"{task} episode不完整：{completed}/{episodes_per_task}"
                )
            task_rows.append(dict(row))
    expected_tasks = set(topology["task_horizons"])
    if seen_tasks != expected_tasks:
        errors.append(
            f"task覆盖不完整：missing={sorted(expected_tasks - seen_tasks)}, "
            f"extra={sorted(seen_tasks - expected_tasks)}"
        )
    total_episodes = sum(int(row.get("episodes_completed", 0)) for row in task_rows)
    total_successes = sum(int(row.get("successes", 0)) for row in task_rows)
    ok = not errors and len(task_rows) == 18
    return {
        "schema_version": 1,
        "result": "pass" if ok else "fail",
        "ok": ok,
        "scope": "atomic_only",
        "eval_id": eval_id,
        "checkpoint": (
            str(Path(checkpoint).expanduser().resolve())
            if checkpoint is not None
            else None
        ),
        "topology": topology["name"],
        "model_seed": topology["model_seed"],
        "seed_start": topology["seed_start"],
        "episodes_per_task": episodes_per_task,
        "replan_steps": topology["replan_steps"],
        "video_denoise_steps": topology["video_denoise_steps"],
        "action_denoise_steps": topology["action_denoise_steps"],
        "tasks": sorted(task_rows, key=lambda row: str(row["task"])),
        "overall": {
            "tasks": len(task_rows),
            "episodes": total_episodes,
            "successes": total_successes,
            "success_rate": total_successes / total_episodes if total_episodes else 0.0,
        },
        "errors": errors,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--episodes-per-task", type=int, required=True)
    parser.add_argument("--eval-id")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.episodes_per_task <= 0:
        parser.error("episodes per task 必须为正")
    report = aggregate(
        topology_path=args.topology,
        output_root=args.output_root,
        episodes_per_task=args.episodes_per_task,
        eval_id=args.eval_id,
        checkpoint=args.checkpoint,
    )
    write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
