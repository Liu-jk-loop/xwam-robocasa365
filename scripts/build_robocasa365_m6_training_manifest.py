#!/usr/bin/env python3
"""审计 Atomic-Seen 18 的 pretrain 数据并生成 M6 正式训练 manifest。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.robocasa365_multitask import (  # noqa: E402
    build_atomic_training_manifest_report,
    training_manifest_digest,
)
from project_tools.training_run import write_json_atomic  # noqa: E402


def _parse_explicit_paths(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        task_name, separator, path = value.partition("=")
        if not separator or not task_name or not path:
            raise ValueError(
                f"--task-path 必须为 TaskName=/absolute/path，实际为 {value!r}"
            )
        if task_name in result:
            raise ValueError(f"--task-path 重复指定任务：{task_name}")
        result[task_name] = path
    return result


def _git_state() -> dict[str, object]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        status = run("status", "--porcelain")
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(status),
            "status": status.splitlines(),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", required=True, help="pretrain/atomic 根目录。"
    )
    parser.add_argument(
        "--task-path",
        action="append",
        default=[],
        help="多日期任务显式指定 TaskName=/absolute/path。",
    )
    parser.add_argument(
        "--task-manifest",
        default="configs/tasks/robocasa365_atomic_seen.json",
    )
    parser.add_argument("--sequence-length", type=int, default=9)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument(
        "--output",
        default="logs/cluster/robocasa365_m6_atomic_seen18_manifest.json",
    )
    args = parser.parse_args()

    try:
        report = build_atomic_training_manifest_report(
            dataset_root=args.dataset_root,
            atomic_task_manifest=args.task_manifest,
            explicit_paths=_parse_explicit_paths(args.task_path),
            sequence_length=args.sequence_length,
            frame_skip=args.frame_skip,
        )
    except (OSError, ValueError) as exc:
        report = {
            "schema_version": 1,
            "name": "robocasa365_m6_atomic_seen18_pretrain",
            "scope": "atomic_only",
            "split": "pretrain",
            "sampling": "natural_proportional",
            "tasks": [],
            "errors": [f"{type(exc).__name__}: {exc}"],
            "ok": False,
            "result": "fail",
        }
    report["git"] = _git_state()
    report["manifest_digest"] = training_manifest_digest(report)
    output = write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"M6 training manifest: {output.resolve()}")
    return 0 if report.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
