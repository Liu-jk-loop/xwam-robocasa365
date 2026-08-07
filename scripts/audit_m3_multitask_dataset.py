#!/usr/bin/env python3
"""解析三个 Atomic-Seen 任务的日期目录并生成 M3.2 训练 manifest。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.robocasa365_multitask import build_m3_multitask_manifest_report
from project_tools.training_run import write_json_atomic


DEFAULT_TASKS = (
    "PickPlaceCounterToCabinet",
    "OpenCabinet",
    "TurnOnMicrowave",
)


def _parse_explicit_paths(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        task_name, separator, path = value.partition("=")
        if not separator or not task_name or not path:
            raise ValueError(f"--task-path 必须为 TaskName=/absolute/path，实际为 {value!r}")
        if task_name in result:
            raise ValueError(f"--task-path 重复指定任务：{task_name}")
        result[task_name] = path
    return result


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="pretrain/atomic 根目录。")
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="覆盖默认任务；必须重复传入三次。",
    )
    parser.add_argument(
        "--task-path",
        action="append",
        default=[],
        help="存在多个日期目录时显式指定 TaskName=/path。",
    )
    parser.add_argument(
        "--task-manifest",
        default="configs/tasks/robocasa365_atomic_seen.json",
    )
    parser.add_argument(
        "--output",
        default="logs/cluster/robocasa365_m3_three_task_manifest.json",
    )
    args = parser.parse_args()

    try:
        explicit_paths = _parse_explicit_paths(args.task_path)
        tasks = tuple(args.task) if args.task else DEFAULT_TASKS
        report = build_m3_multitask_manifest_report(
            dataset_root=args.dataset_root,
            task_names=tasks,
            atomic_task_manifest=args.task_manifest,
            explicit_paths=explicit_paths,
        )
    except (OSError, ValueError) as exc:
        report = {
            "schema_version": 1,
            "name": "robocasa365_m3_three_task_smoke",
            "scope": "atomic_only",
            "sampling": "balanced_round_robin",
            "tasks": [],
            "errors": [f"{type(exc).__name__}: {exc}"],
            "ok": False,
            "result": "fail",
        }
    report["git_commit"] = _git_commit()
    output = write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Dataset manifest: {output.resolve()}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
