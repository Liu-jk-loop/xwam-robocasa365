#!/usr/bin/env python3
"""Aggregate and validate the eighteen lean M6 task results."""

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
        raise ValueError(f"JSON 顶层必须为对象：{path}")
    return payload


def _expected_task_entries(topology: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        task
        for client in topology["clients"].values()
        for task in client["tasks"]
    ]


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
    expected_entries = _expected_task_entries(topology)
    for task_entry in expected_entries:
        task = str(task_entry["name"])
        result_path = root / task / "result.json"
        if not result_path.is_file():
            errors.append(f"缺少 task result：{result_path}")
            continue
        result = _read_json(result_path)
        expected_fields = {
            "task": task,
            "result": "pass",
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
            actual = result.get(key)
            if actual != expected:
                errors.append(
                    f"{task} {key}漂移：expected={expected!r}, actual={actual!r}"
                )
        episodes = result.get("episodes")
        if not isinstance(episodes, list) or len(episodes) != episodes_per_task:
            errors.append(f"{task} episode列表不完整")
        successes = int(result.get("n_success", -1))
        if successes < 0 or successes > episodes_per_task:
            errors.append(f"{task} n_success非法：{successes}")
        success_rate = float(result.get("success_rate", -1.0))
        expected_rate = successes / episodes_per_task
        if not math.isclose(success_rate, expected_rate, rel_tol=0.0, abs_tol=1e-12):
            errors.append(
                f"{task} success_rate与计数不一致："
                f"expected={expected_rate}, actual={success_rate}"
            )
        task_rows.append(
            {
                "task": task,
                "episodes": int(result.get("episodes_completed", 0)),
                "successes": successes,
                "success_rate": success_rate,
                "mean_inference_time_s": result.get("mean_inference_time_s"),
                "fastwam_reference_success_percent": task_entry[
                    "fastwam_reference_success_percent"
                ],
                "result_path": str(result_path),
            }
        )
    total_episodes = sum(row["episodes"] for row in task_rows)
    total_successes = sum(row["successes"] for row in task_rows)
    macro_rate = (
        sum(row["success_rate"] for row in task_rows) / len(task_rows)
        if task_rows
        else 0.0
    )
    ok = not errors and len(task_rows) == 18
    return {
        "schema_version": 2,
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
        "split": topology["scene"]["split"],
        "max_steps_per_episode": topology["max_steps_per_episode"],
        "replan_steps": topology["replan_steps"],
        "video_denoise_steps": topology["video_denoise_steps"],
        "action_denoise_steps": topology["action_denoise_steps"],
        "video_fps": topology["video"]["fps"],
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
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _write_csv(path: str | Path, report: dict[str, Any]) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    fieldnames = [
        "task",
        "episodes",
        "successes",
        "success_rate",
        "mean_inference_time_s",
        "fastwam_reference_success_percent",
    ]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["tasks"]:
            writer.writerow({key: row.get(key) for key in fieldnames})
    temporary.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--episodes-per-task", type=int, required=True)
    parser.add_argument("--eval-id")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output", required=True)
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
    _write_csv(args.csv_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
