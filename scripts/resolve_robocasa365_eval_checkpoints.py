#!/usr/bin/env python3
"""Resolve exact complete multi-rank checkpoints for grouped evaluation."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from project_tools.h100_training import _checkpoint_layout  # noqa: E402
from project_tools.training_run import write_json_atomic  # noqa: E402


def _parse_group_steps(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        group, separator, raw_step = value.partition("=")
        if (
            not separator
            or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", group)
            or not raw_step.isdigit()
            or int(raw_step) <= 0
        ):
            raise ValueError("--group-step必须为GROUP=正整数")
        if group in result:
            raise ValueError(f"重复的checkpoint group：{group}")
        result[group] = int(raw_step)
    return result


def resolve_exact_checkpoints(
    checkpoint_roots: list[str | Path],
    group_steps: dict[str, int],
    *,
    expected_world_size: int,
) -> dict[str, Any]:
    roots = [Path(value).expanduser().resolve() for value in checkpoint_roots]
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("checkpoint roots不能为空或重复")
    if not group_steps:
        raise ValueError("至少需要一个group step")
    pattern = re.compile(r"^(?:epoch=\d+-step=|final-step=)(\d+)\.ckpt$")
    matches: dict[int, list[tuple[int, int, Path]]] = {
        step: [] for step in group_steps.values()
    }
    incomplete: list[dict[str, Any]] = []
    for root_index, root in enumerate(roots):
        if not root.is_dir():
            continue
        for candidate in sorted(root.iterdir()):
            match = pattern.fullmatch(candidate.name)
            if match is None or not candidate.is_dir():
                continue
            step = int(match.group(1))
            if step not in matches:
                continue
            layout = _checkpoint_layout(
                candidate, expected_world_size=expected_world_size
            )
            if all(layout["checks"].values()):
                final_priority = int(candidate.name.startswith("final-step="))
                matches[step].append((final_priority, root_index, candidate.resolve()))
            else:
                incomplete.append(
                    {
                        "step": step,
                        "path": str(candidate.resolve()),
                        "checks": layout["checks"],
                    }
                )
    resolved: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for group, step in group_steps.items():
        candidates = matches[step]
        if not candidates:
            errors.append(f"{group}: 没有找到完整step {step} checkpoint")
            continue
        _, root_index, selected = max(
            candidates, key=lambda item: (item[0], item[1], str(item[2]))
        )
        resolved[group] = {
            "step": step,
            "path": str(selected),
            "checkpoint_root": str(roots[root_index]),
        }
    return {
        "schema_version": 1,
        "result": "pass" if not errors else "fail",
        "ok": not errors,
        "expected_world_size": int(expected_world_size),
        "checkpoint_roots": [str(root) for root in roots],
        "groups": resolved,
        "incomplete_candidates": incomplete,
        "errors": errors,
    }


def _write_env(path: str | Path, report: dict[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for group, record in sorted(report["groups"].items()):
        key = f"XWAM_EVAL_CHECKPOINT_{group.upper()}"
        lines.append(f"{key}={shlex.quote(str(record['path']))}\n")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.writelines(lines)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", action="append", required=True)
    parser.add_argument("--group-step", action="append", required=True)
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-env", required=True)
    args = parser.parse_args()
    if args.expected_world_size <= 0:
        parser.error("expected world size必须为正")
    try:
        group_steps = _parse_group_steps(args.group_step)
        report = resolve_exact_checkpoints(
            args.checkpoint_root,
            group_steps,
            expected_world_size=args.expected_world_size,
        )
    except (OSError, ValueError) as exc:
        report = {
            "schema_version": 1,
            "result": "fail",
            "ok": False,
            "groups": {},
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
    write_json_atomic(args.output, report)
    if report["ok"]:
        _write_env(args.output_env, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
