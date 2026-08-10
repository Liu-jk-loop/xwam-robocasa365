#!/usr/bin/env python3
"""Resolve a validated Clariden step-2 initial checkpoint for resume."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from project_tools.clariden_training import (  # noqa: E402
    resolve_clariden_initial_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint = resolve_clariden_initial_checkpoint(
        result_path=args.result,
        record_path=args.output,
    )
    print(f"Initial checkpoint: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
