#!/usr/bin/env python3
"""审计 RoboCasa365 RGB-D 离线回放所需的 states/MJCF/episode 元数据。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.robocasa365_contract import load_task_manifest  # noqa: E402
from data.robocasa365_multitask import resolve_task_dataset_directory  # noqa: E402
from project_tools.robocasa365_depth_preflight import (  # noqa: E402
    inspect_depth_replay_inputs,
)
from project_tools.training_run import write_json_atomic  # noqa: E402


def _explicit_paths(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        task, separator, path = value.partition("=")
        if not separator or not task or not path:
            raise ValueError(f"--task-path 必须为 TaskName=/absolute/path，实际为 {value!r}")
        if task in parsed:
            raise ValueError(f"--task-path 重复指定任务：{task}")
        parsed[task] = path
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="单任务数据目录或其 lerobot 子目录。")
    source.add_argument("--dataset-root", help="包含各任务目录的 pretrain/atomic 根目录。")
    parser.add_argument("--task-name", help="与 --dataset 配套的任务名。")
    parser.add_argument(
        "--task-manifest",
        default="configs/tasks/robocasa365_atomic9_fastwam_overlap.json",
        help="--dataset-root 模式需要审计的 atomic 任务清单。",
    )
    parser.add_argument(
        "--task-path",
        action="append",
        default=[],
        help="多日期任务显式指定 TaskName=/absolute/path。",
    )
    parser.add_argument(
        "--episodes-to-decode",
        type=int,
        default=3,
        help="每任务实际解压检查的 episode 数；0 表示全部。全部 episode 始终检查文件是否存在。",
    )
    parser.add_argument("--output", required=True, help="JSON 审计报告路径。")
    args = parser.parse_args()

    errors: list[str] = []
    tasks: list[dict[str, object]] = []
    try:
        explicit = _explicit_paths(args.task_path)
        if args.dataset:
            if not args.task_name:
                raise ValueError("--dataset 模式必须提供 --task-name")
            if explicit:
                raise ValueError("--dataset 模式不接受 --task-path")
            selected = [(args.task_name, Path(args.dataset).expanduser().resolve())]
        else:
            if args.task_name:
                raise ValueError("--task-name 只用于 --dataset 模式")
            manifest = load_task_manifest(args.task_manifest)
            unknown = sorted(set(explicit) - set(manifest.tasks))
            if unknown:
                raise ValueError(f"--task-path 包含 manifest 外任务：{unknown}")
            selected = []
            for task_name in manifest.tasks:
                path = (
                    Path(explicit[task_name]).expanduser().resolve()
                    if task_name in explicit
                    else resolve_task_dataset_directory(args.dataset_root, task_name)
                )
                selected.append((task_name, path))

        for task_name, path in selected:
            try:
                tasks.append(
                    inspect_depth_replay_inputs(
                        path,
                        task_name=task_name,
                        episodes_to_decode=args.episodes_to_decode,
                    )
                )
            except (OSError, ValueError) as exc:
                errors.append(f"{task_name}: {type(exc).__name__}: {exc}")
    except (OSError, ValueError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")

    failed_tasks = [str(item.get("task_name")) for item in tasks if item.get("ok") is not True]
    ok = bool(tasks) and not errors and not failed_tasks
    report = {
        "schema_version": 2,
        "phase": "RGBD-P0-P1-structural-preflight",
        "scope": "atomic_only",
        "task_manifest": args.task_manifest if args.dataset_root else None,
        "expected_tasks": len(tasks) + len(errors),
        "passed_tasks": sum(item.get("ok") is True for item in tasks),
        "failed_tasks": failed_tasks,
        "tasks": tasks,
        "errors": errors,
        "ok": ok,
        "result": "pass" if ok else "fail",
    }
    output = write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"RGB-D replay input audit: {output.resolve()}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
