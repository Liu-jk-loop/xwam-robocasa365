#!/usr/bin/env python3
"""审计 M3.2 单 clip RGB-only 过拟合日志并写出机器可读报告。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from project_tools.training_curve import build_overfit_curve_report
from project_tools.training_run import write_json_atomic


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
    parser.add_argument("--log", required=True, help="M3.2 完整 tee 日志路径。")
    parser.add_argument("--expected-steps", type=int, default=50)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--minimum-relative-drop", type=float, default=0.10)
    parser.add_argument(
        "--output",
        default="logs/cluster/close_fridge_m3_overfit_curve_audit.json",
    )
    args = parser.parse_args()

    log_path = Path(args.log).expanduser().resolve()
    if not log_path.is_file():
        parser.error(f"日志不存在：{log_path}")
    report = build_overfit_curve_report(
        log_path.read_text(encoding="utf-8", errors="replace"),
        expected_steps=args.expected_steps,
        window_size=args.window_size,
        minimum_relative_drop=args.minimum_relative_drop,
    )
    report["git_commit"] = _git_commit()
    report["log_path"] = str(log_path)
    output = write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Audit report: {output.resolve()}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
