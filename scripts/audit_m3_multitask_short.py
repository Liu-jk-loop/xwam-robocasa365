#!/usr/bin/env python3
"""审计 M3.2 三个 atomic 任务的 12-step 训练日志。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from project_tools.multitask_training import build_multitask_short_report
from project_tools.training_run import write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="tee 保存的完整训练日志。")
    parser.add_argument(
        "--manifest",
        default="logs/cluster/robocasa365_m3_three_task_manifest.json",
        help="已通过数据审计的三任务 manifest。",
    )
    parser.add_argument(
        "--output",
        default="logs/cluster/robocasa365_m3_three_task_short_audit.json",
    )
    parser.add_argument(
        "--max-task-count-spread",
        type=int,
        default=2,
        help="12-step 分布式随机采样中任务计数最大值与最小值的允许差，默认 2。",
    )
    args = parser.parse_args()

    try:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        if manifest.get("ok") is not True:
            raise ValueError("三任务 manifest 未通过数据审计")
        task_names = [entry["task_name"] for entry in manifest.get("tasks", [])]
        report = build_multitask_short_report(
            Path(args.log).read_text(encoding="utf-8", errors="replace"),
            task_names=task_names,
            expected_steps=12,
            max_task_count_spread=args.max_task_count_spread,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        report = {
            "schema_version": 1,
            "result": "fail",
            "ok": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
    report["log_path"] = str(Path(args.log).expanduser())
    report["manifest_path"] = str(Path(args.manifest).expanduser())
    output = write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Training audit: {output.resolve()}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
