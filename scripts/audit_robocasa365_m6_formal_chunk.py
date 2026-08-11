#!/usr/bin/env python3
"""Audit one completed restartable M6 formal-training chunk."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from project_tools.h100_training import build_m6_formal_chunk_report  # noqa: E402
from project_tools.training_run import write_json_atomic  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--optimizer-dir", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--expected-resume-checkpoint")
    parser.add_argument("--expected-wandb-run-id", required=True)
    parser.add_argument("--expected-rolling-checkpoint-root", required=True)
    parser.add_argument("--expected-durable-checkpoint-root", required=True)
    parser.add_argument("--expected-world-size", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_m6_formal_chunk_report(
        metadata_path=args.metadata,
        result_path=args.result,
        events_path=args.events,
        optimizer_dir=args.optimizer_dir,
        log_path=args.log,
        expected_step=args.expected_step,
        expected_resume_checkpoint=args.expected_resume_checkpoint,
        expected_wandb_run_id=args.expected_wandb_run_id,
        expected_rolling_checkpoint_root=args.expected_rolling_checkpoint_root,
        expected_durable_checkpoint_root=args.expected_durable_checkpoint_root,
        expected_world_size=args.expected_world_size,
    )
    output = write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"M6 formal chunk audit: {output.resolve()}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
