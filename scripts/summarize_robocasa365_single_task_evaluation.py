#!/usr/bin/env python3
"""Validate one lean RoboCasa365 task result and write a compact summary."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from project_tools.training_run import write_json_atomic  # noqa: E402


def _read_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON顶层必须为对象：{resolved}")
    return payload


def summarize(
    result: dict[str, Any],
    *,
    task: str,
    episodes: int,
    eval_id: str,
    checkpoint: str | Path,
) -> dict[str, Any]:
    errors: list[str] = []
    expected = {
        "result": "pass",
        "task": task,
        "episodes_expected": episodes,
        "episodes_completed": episodes,
        "model_seed": 42,
        "seed_start": 42,
        "max_steps": 1000,
        "replan_steps": 20,
        "action_denoise_steps": 10,
        "video_fps": 20,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            errors.append(
                f"{key}漂移：expected={value!r}, actual={result.get(key)!r}"
            )
    successes = int(result.get("n_success", -1))
    success_rate = float(result.get("success_rate", -1.0))
    if not 0 <= successes <= episodes:
        errors.append(f"n_success非法：{successes}")
    expected_rate = successes / episodes
    if not math.isclose(success_rate, expected_rate, rel_tol=0.0, abs_tol=1e-12):
        errors.append(
            f"success_rate与计数不一致：expected={expected_rate}, actual={success_rate}"
        )
    episode_rows = result.get("episodes")
    if not isinstance(episode_rows, list) or len(episode_rows) != episodes:
        errors.append("episode列表不完整")
    videos = sorted(
        str(row.get("video"))
        for row in episode_rows or []
        if isinstance(row, dict) and row.get("video")
    )
    ok = not errors
    return {
        "schema_version": 1,
        "result": "pass" if ok else "fail",
        "ok": ok,
        "eval_id": eval_id,
        "task": task,
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "episodes": episodes,
        "successes": successes,
        "success_rate": success_rate,
        "success_percent": success_rate * 100.0,
        "mean_inference_time_s": result.get("mean_inference_time_s"),
        "videos": videos,
        "errors": errors,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--eval-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("episodes必须为正")
    report = summarize(
        _read_json(args.result),
        task=args.task,
        episodes=args.episodes,
        eval_id=args.eval_id,
        checkpoint=args.checkpoint,
    )
    write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
