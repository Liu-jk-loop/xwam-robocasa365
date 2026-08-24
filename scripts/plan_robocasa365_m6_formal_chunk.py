#!/usr/bin/env python3
"""Plan the next restartable M6 formal-training chunk from complete checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from project_tools.h100_training import (  # noqa: E402
    quarantine_m6_incomplete_checkpoints,
    resolve_m6_formal_chunk,
)
from project_tools.training_run import write_json_atomic  # noqa: E402


def _write_env_atomic(path: Path, plan: dict) -> Path:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    values = {
        "XWAM_PLAN_ALREADY_COMPLETE": int(plan["already_complete"]),
        "XWAM_PLAN_COMPLETED_STEP": int(plan["completed_step"]),
        "XWAM_PLAN_TARGET_STEP": int(plan["target_step"]),
        "XWAM_PLAN_FINAL_CHUNK": int(plan["final_chunk"]),
        "XWAM_PLAN_RESUME_CHECKPOINT": plan["resume_checkpoint"] or "",
    }
    content = "".join(
        f"{key}={shlex.quote(str(value))}\n" for key, value in values.items()
    )
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{resolved.name}.", dir=str(resolved.parent), text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", action="append", required=True)
    parser.add_argument("--total-steps", type=int, default=16390)
    parser.add_argument("--chunk-steps", type=int, default=1000)
    parser.add_argument("--expected-world-size", type=int, default=4)
    parser.add_argument("--bootstrap-checkpoint-root", action="append")
    parser.add_argument("--bootstrap-step", type=int)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-env", required=True)
    parser.add_argument("--quarantine-incomplete-root", action="append", required=True)
    args = parser.parse_args()
    plan = resolve_m6_formal_chunk(
        args.checkpoint_root,
        total_steps=args.total_steps,
        chunk_steps=args.chunk_steps,
        expected_world_size=args.expected_world_size,
        bootstrap_checkpoint_root=args.bootstrap_checkpoint_root,
        bootstrap_step=args.bootstrap_step,
    )
    plan = quarantine_m6_incomplete_checkpoints(plan, args.quarantine_incomplete_root)
    json_path = write_json_atomic(args.output_json, plan)
    env_path = _write_env_atomic(Path(args.output_env), plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"M6 formal chunk plan: {json_path.resolve()}")
    print(f"M6 formal chunk environment: {env_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
