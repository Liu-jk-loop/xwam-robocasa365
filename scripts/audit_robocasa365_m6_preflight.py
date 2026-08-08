#!/usr/bin/env python3
"""校验 M6 18任务 manifest/global stats，并输出精确5-epoch步数。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from project_tools.h100_training import build_m6_preflight_report  # noqa: E402
from project_tools.training_run import write_json_atomic  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="logs/cluster/robocasa365_m6_atomic_seen18_manifest.json",
    )
    parser.add_argument(
        "--stats",
        default="logs/cluster/robocasa365_m6_atomic_seen18_global_stats.json",
    )
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--output",
        default="logs/cluster/robocasa365_m6_h100_preflight.json",
    )
    args = parser.parse_args()
    report = build_m6_preflight_report(
        args.manifest,
        args.stats,
        global_batch_size=args.global_batch_size,
        num_train_epochs=args.epochs,
    )
    output = write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"M6 preflight: {output.resolve()}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
