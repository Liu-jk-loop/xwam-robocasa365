#!/usr/bin/env python3
"""审计 M6 RGB-only/RGB-D 正式训练的完整运行证据。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from project_tools.h100_training import build_m6_formal_report  # noqa: E402
from project_tools.training_run import write_json_atomic  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--optimizer-dir", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--expected-wandb-run-id")
    parser.add_argument("--expected-rolling-checkpoint-root")
    parser.add_argument("--expected-durable-checkpoint-root")
    parser.add_argument("--expected-world-size", type=int, default=4)
    parser.add_argument(
        "--output",
        default="logs/cluster/robocasa365_m6_formal_audit.json",
    )
    args = parser.parse_args()
    report = build_m6_formal_report(
        metadata_path=args.metadata,
        result_path=args.result,
        events_path=args.events,
        optimizer_dir=args.optimizer_dir,
        log_path=args.log,
        expected_wandb_run_id=args.expected_wandb_run_id,
        expected_rolling_checkpoint_root=args.expected_rolling_checkpoint_root,
        expected_durable_checkpoint_root=args.expected_durable_checkpoint_root,
        expected_world_size=args.expected_world_size,
    )
    output = write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"M6 formal audit: {output.resolve()}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
