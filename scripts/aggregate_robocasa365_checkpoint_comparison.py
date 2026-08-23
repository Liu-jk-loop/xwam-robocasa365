#!/usr/bin/env python3
"""Aggregate two Atomic9 checkpoint groups and compute matched-seed deltas."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return payload


def _parse_group_checkpoints(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        group, separator, path = value.partition("=")
        if not separator or not group or not path:
            raise ValueError("--group-checkpoint必须为GROUP=/absolute/checkpoint")
        if group in result:
            raise ValueError(f"重复的checkpoint group：{group}")
        result[group] = str(Path(path).expanduser().resolve())
    return result


def _entries_for_group(
    topology: dict[str, Any], group: str
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return [
        (client, task)
        for client in topology["clients"].values()
        if client.get("comparison_group") == group
        for task in client["tasks"]
    ]


def _aggregate_group(
    *,
    topology: dict[str, Any],
    group: str,
    root: Path,
    episodes_per_task: int,
    checkpoint: str,
) -> dict[str, Any]:
    errors: list[str] = []
    task_rows: list[dict[str, Any]] = []
    for client, task_entry in _entries_for_group(topology, group):
        task = str(task_entry["name"])
        result_path = root / group / task / "result.json"
        if not result_path.is_file():
            errors.append(f"缺少task result：{result_path}")
            continue
        result = _read_json(result_path)
        expected_fields = {
            "task": task,
            "result": "pass",
            "client_id": client["client_id"],
            "server_id": client["server_id"],
            "comparison_group": group,
            "split": topology["scene"]["split"],
            "model_seed": topology["model_seed"],
            "seed_start": topology["seed_start"],
            "episodes_expected": episodes_per_task,
            "episodes_completed": episodes_per_task,
            "max_steps": topology["max_steps_per_episode"],
            "replan_steps": topology["replan_steps"],
            "action_denoise_steps": topology["action_denoise_steps"],
            "video_fps": topology["video"]["fps"],
        }
        for key, expected in expected_fields.items():
            if result.get(key) != expected:
                errors.append(
                    f"{group}/{task} {key}漂移："
                    f"expected={expected!r}, actual={result.get(key)!r}"
                )
        episodes = result.get("episodes")
        if not isinstance(episodes, list) or len(episodes) != episodes_per_task:
            errors.append(f"{group}/{task} episode列表不完整")
        else:
            expected_seeds = [
                int(topology["seed_start"]) + index
                for index in range(episodes_per_task)
            ]
            actual_seeds = [int(row.get("seed", -1)) for row in episodes]
            if actual_seeds != expected_seeds:
                errors.append(f"{group}/{task} episode seed集合漂移")
        successes = int(result.get("n_success", -1))
        success_rate = float(result.get("success_rate", -1.0))
        expected_rate = successes / episodes_per_task
        if successes < 0 or successes > episodes_per_task:
            errors.append(f"{group}/{task} n_success非法：{successes}")
        if not math.isclose(
            success_rate, expected_rate, rel_tol=0.0, abs_tol=1e-12
        ):
            errors.append(
                f"{group}/{task} success_rate与计数不一致："
                f"expected={expected_rate}, actual={success_rate}"
            )
        task_rows.append(
            {
                "task": task,
                "episodes": int(result.get("episodes_completed", 0)),
                "successes": successes,
                "success_rate": success_rate,
                "mean_inference_time_s": result.get("mean_inference_time_s"),
                "historical_xwam_success_percent": task_entry.get(
                    "historical_xwam_success_percent"
                ),
                "fastwam_reference_success_percent": task_entry[
                    "fastwam_reference_success_percent"
                ],
                "result_path": str(result_path),
            }
        )
    expected_task_count = len(topology["task_horizons"])
    total_episodes = sum(row["episodes"] for row in task_rows)
    total_successes = sum(row["successes"] for row in task_rows)
    macro_rate = (
        sum(row["success_rate"] for row in task_rows) / len(task_rows)
        if task_rows
        else 0.0
    )
    ok = not errors and len(task_rows) == expected_task_count
    return {
        "result": "pass" if ok else "fail",
        "ok": ok,
        "checkpoint_step": topology["comparison_groups"][group]["checkpoint_step"],
        "checkpoint": checkpoint,
        "tasks": sorted(task_rows, key=lambda row: row["task"]),
        "overall": {
            "tasks": len(task_rows),
            "episodes": total_episodes,
            "successes": total_successes,
            "micro_success_rate": (
                total_successes / total_episodes if total_episodes else 0.0
            ),
            "macro_success_rate": macro_rate,
        },
        "errors": errors,
    }


def aggregate_comparison(
    *,
    topology_path: str | Path,
    output_root: str | Path,
    episodes_per_task: int,
    group_checkpoints: dict[str, str],
    eval_id: str | None = None,
) -> dict[str, Any]:
    topology = load_m6_evaluation_topology(topology_path, REPO_ROOT)
    comparison_groups = topology.get("comparison_groups", {})
    if len(comparison_groups) != 2:
        raise ValueError("checkpoint comparison只接受两个group的schema v2 topology")
    if set(group_checkpoints) != set(comparison_groups):
        raise ValueError(
            "checkpoint group不完整："
            f"expected={sorted(comparison_groups)}, actual={sorted(group_checkpoints)}"
        )
    root = Path(output_root).expanduser().resolve()
    groups = {
        group: _aggregate_group(
            topology=topology,
            group=group,
            root=root,
            episodes_per_task=episodes_per_task,
            checkpoint=group_checkpoints[group],
        )
        for group in comparison_groups
    }
    ordered_groups = sorted(
        groups,
        key=lambda name: int(groups[name]["checkpoint_step"]),
    )
    early_name, late_name = ordered_groups
    early_tasks = {row["task"]: row for row in groups[early_name]["tasks"]}
    late_tasks = {row["task"]: row for row in groups[late_name]["tasks"]}
    task_deltas = []
    for task in sorted(set(early_tasks) & set(late_tasks)):
        early_rate = float(early_tasks[task]["success_rate"])
        late_rate = float(late_tasks[task]["success_rate"])
        task_deltas.append(
            {
                "task": task,
                "early_group": early_name,
                "late_group": late_name,
                "early_success_rate": early_rate,
                "late_success_rate": late_rate,
                "success_rate_delta": late_rate - early_rate,
            }
        )
    errors = [
        f"{group}: {error}"
        for group, report in groups.items()
        for error in report["errors"]
    ]
    if set(early_tasks) != set(late_tasks):
        errors.append("两个checkpoint group的任务集合不一致")
    ok = not errors and all(report["ok"] for report in groups.values())
    return {
        "schema_version": 1,
        "result": "pass" if ok else "fail",
        "ok": ok,
        "scope": "atomic_only",
        "eval_id": eval_id,
        "topology": topology["name"],
        "model_seed": topology["model_seed"],
        "seed_start": topology["seed_start"],
        "episodes_per_task": episodes_per_task,
        "groups": groups,
        "comparison": {
            "early_group": early_name,
            "late_group": late_name,
            "macro_success_rate_delta": groups[late_name]["overall"][
                "macro_success_rate"
            ]
            - groups[early_name]["overall"]["macro_success_rate"],
            "tasks": task_deltas,
        },
        "errors": errors,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _write_csv(path: str | Path, report: dict[str, Any]) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    fieldnames = [
        "group",
        "checkpoint_step",
        "task",
        "episodes",
        "successes",
        "success_rate",
        "mean_inference_time_s",
        "historical_xwam_success_percent",
        "fastwam_reference_success_percent",
    ]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group, group_report in report["groups"].items():
            for row in group_report["tasks"]:
                writer.writerow(
                    {
                        **{key: row.get(key) for key in fieldnames},
                        "group": group,
                        "checkpoint_step": group_report["checkpoint_step"],
                    }
                )
    temporary.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--episodes-per-task", type=int, required=True)
    parser.add_argument("--group-checkpoint", action="append", required=True)
    parser.add_argument("--eval-id")
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output", required=True)
    args = parser.parse_args()
    if args.episodes_per_task <= 0:
        parser.error("episodes per task必须为正")
    try:
        checkpoints = _parse_group_checkpoints(args.group_checkpoint)
        report = aggregate_comparison(
            topology_path=args.topology,
            output_root=args.output_root,
            episodes_per_task=args.episodes_per_task,
            group_checkpoints=checkpoints,
            eval_id=args.eval_id,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        report = {
            "schema_version": 1,
            "result": "fail",
            "ok": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
            "groups": {},
        }
    write_json_atomic(args.output, report)
    if report["groups"]:
        _write_csv(args.csv_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
