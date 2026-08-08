#!/usr/bin/env python3
"""联合审计 H100 M6 门禁的初始2步和恢复至4步运行证据。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from project_tools.h100_training import build_m6_gate_report  # noqa: E402
from project_tools.training_run import write_json_atomic  # noqa: E402


def _artifact_arguments(parser: argparse.ArgumentParser, prefix: str) -> None:
    for name in ("metadata", "result", "events", "optimizer-dir", "log"):
        parser.add_argument(f"--{prefix}-{name}", required=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _artifact_arguments(parser, "initial")
    _artifact_arguments(parser, "resumed")
    parser.add_argument(
        "--output",
        default="logs/cluster/robocasa365_m6_h100_gate_audit.json",
    )
    args = parser.parse_args()
    report = build_m6_gate_report(
        initial={
            "metadata_path": args.initial_metadata,
            "result_path": args.initial_result,
            "events_path": args.initial_events,
            "optimizer_dir": args.initial_optimizer_dir,
            "log_path": args.initial_log,
        },
        resumed={
            "metadata_path": args.resumed_metadata,
            "result_path": args.resumed_result,
            "events_path": args.resumed_events,
            "optimizer_dir": args.resumed_optimizer_dir,
            "log_path": args.resumed_log,
        },
    )
    output = write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"M6 H100 gate audit: {output.resolve()}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
