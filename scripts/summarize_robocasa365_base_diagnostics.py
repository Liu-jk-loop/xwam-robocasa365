#!/usr/bin/env python3
"""Summarize compact base-action diagnostics from one M6 task result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON顶层必须为对象：{path}")
    return payload


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    episodes = result.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("task result没有已完成episode")
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        diagnostics = episode.get("base_action_diagnostics")
        if not isinstance(diagnostics, dict):
            raise ValueError(
                f"episode seed={episode.get('seed')}缺少base_action_diagnostics；"
                "请使用加入诊断后的代码重新跑新episode"
            )
        rows.append(
            {
                "seed": int(episode["seed"]),
                "steps": int(diagnostics["steps"]),
                "base_command_nonzero_fraction": float(
                    diagnostics["base_command_nonzero_fraction"]
                ),
                "base_command_rms_per_dim": diagnostics[
                    "base_command_rms_per_dim"
                ],
                "control_mode_counts": diagnostics["control_mode_counts"],
                "base_position_delta_nonzero_fraction": float(
                    diagnostics["base_position_delta_nonzero_fraction"]
                ),
                "base_position_delta_rms_per_dim": diagnostics[
                    "base_position_delta_rms_per_dim"
                ],
                "base_position_total_displacement": float(
                    diagnostics["base_position_total_displacement"]
                ),
                "commanded_but_stationary_steps": int(
                    diagnostics["commanded_but_stationary_steps"]
                ),
            }
        )

    total_steps = sum(row["steps"] for row in rows)
    if total_steps <= 0:
        verdict = "no_environment_steps"
    else:
        command_fraction = sum(
            row["base_command_nonzero_fraction"] * row["steps"] for row in rows
        ) / total_steps
        movement_fraction = sum(
            row["base_position_delta_nonzero_fraction"] * row["steps"]
            for row in rows
        ) / total_steps
        if command_fraction < 0.01:
            verdict = "policy_base_output_near_zero"
        elif movement_fraction < 0.01:
            verdict = "base_command_present_but_environment_stationary"
        else:
            verdict = "base_command_and_environment_motion_present"

    return {
        "schema_version": 1,
        "task": result.get("task"),
        "episodes": rows,
        "verdict": verdict,
        "interpretation": {
            "policy_base_output_near_zero": (
                "优先排查base/control新初始化层、监督比例、类别不平衡和学习率。"
            ),
            "base_command_present_but_environment_stationary": (
                "优先排查control_mode语义及X-WAM Gym动作执行链路。"
            ),
            "base_command_and_environment_motion_present": (
                "底盘链路基本有效；下一步比较X-WAM与FastWAM动作幅度、方向和条件响应。"
            ),
            "no_environment_steps": "episode没有执行环境步骤，结果不可判定。",
        }[verdict],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = summarize(_read_json(Path(args.result).expanduser().resolve()))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_text(text + "\n", encoding="utf-8")
        temporary.replace(output)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
