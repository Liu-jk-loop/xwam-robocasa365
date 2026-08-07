"""Dependency-free parsing and validation for M3 overfit training logs."""

from __future__ import annotations

import math
import re
from statistics import fmean
from typing import Any


METRICS_PATTERN = re.compile(r"\[METRICS\]\s+Step:\s+(\d+)\s+-\s+(.*)$")
REQUIRED_METRICS = (
    "train/video_loss",
    "train/action_loss",
    "train/proprio_loss",
    "train/action_proprio_supervision_ratio",
    "train/depth_loss",
    "train/loss",
)
TREND_METRICS = (
    "train/video_loss",
    "train/action_loss",
    "train/proprio_loss",
    "train/loss",
)


def parse_console_metrics(log_text: str) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    seen_steps: set[int] = set()
    for line_number, line in enumerate(log_text.splitlines(), start=1):
        match = METRICS_PATTERN.search(line)
        if match is None:
            continue
        step = int(match.group(1))
        if step in seen_steps:
            errors.append(f"step {step} 在日志中重复出现")
            continue
        seen_steps.add(step)
        metrics: dict[str, float] = {}
        for field in match.group(2).split(", "):
            key, separator, raw_value = field.partition(": ")
            if not separator or not key.startswith("train/"):
                continue
            try:
                metrics[key] = float(raw_value)
            except ValueError:
                errors.append(
                    f"第 {line_number} 行的 {key} 不是有限浮点文本：{raw_value}"
                )
        records.append({"step": step, "metrics": metrics})
    return records, errors


def build_overfit_curve_report(
    log_text: str,
    *,
    expected_steps: int = 50,
    window_size: int = 10,
    minimum_relative_drop: float = 0.10,
) -> dict[str, Any]:
    expected_steps = int(expected_steps)
    window_size = int(window_size)
    minimum_relative_drop = float(minimum_relative_drop)
    if expected_steps <= 0:
        raise ValueError("expected_steps 必须为正整数")
    if window_size <= 0 or 2 * window_size > expected_steps:
        raise ValueError("window_size 必须为正数且两倍不能超过 expected_steps")
    if not 0 <= minimum_relative_drop < 1:
        raise ValueError("minimum_relative_drop 必须位于 [0, 1)")

    records, errors = parse_console_metrics(log_text)
    records.sort(key=lambda item: item["step"])
    actual_steps = [record["step"] for record in records]
    expected_step_ids = list(range(expected_steps))
    checks: dict[str, bool] = {
        "complete_step_sequence": actual_steps == expected_step_ids,
        "trainer_reached_max_steps": f"`max_steps={expected_steps}` reached" in log_text,
        "run_result_pass": "Run result:" in log_text and "(pass)" in log_text,
    }

    missing_metrics: dict[int, list[str]] = {}
    non_finite: dict[int, list[str]] = {}
    for record in records:
        metrics = record["metrics"]
        missing = [key for key in REQUIRED_METRICS if key not in metrics]
        invalid = [
            key for key in REQUIRED_METRICS if key in metrics and not math.isfinite(metrics[key])
        ]
        if missing:
            missing_metrics[record["step"]] = missing
        if invalid:
            non_finite[record["step"]] = invalid
    checks["required_metrics_present"] = not missing_metrics
    checks["all_metrics_finite"] = not non_finite

    summaries: dict[str, dict[str, float | bool]] = {}
    can_summarize = (
        checks["complete_step_sequence"]
        and checks["required_metrics_present"]
        and checks["all_metrics_finite"]
    )
    if can_summarize:
        for metric_name in TREND_METRICS:
            values = [record["metrics"][metric_name] for record in records]
            first_mean = fmean(values[:window_size])
            last_mean = fmean(values[-window_size:])
            relative_drop = (first_mean - last_mean) / max(abs(first_mean), 1e-12)
            summaries[metric_name] = {
                "first_window_mean": first_mean,
                "last_window_mean": last_mean,
                "relative_drop": relative_drop,
                "meets_minimum_drop": relative_drop >= minimum_relative_drop,
            }
        supervision = [
            record["metrics"]["train/action_proprio_supervision_ratio"]
            for record in records
        ]
        depth = [record["metrics"]["train/depth_loss"] for record in records]
        checks["full_action_proprio_supervision"] = all(
            abs(value - 1.0) <= 1e-6 for value in supervision
        )
        checks["depth_loss_zero"] = all(abs(value) <= 1e-8 for value in depth)
        checks["minimum_loss_drop"] = all(
            bool(summary["meets_minimum_drop"])
            for summary in summaries.values()
        )
    else:
        checks["full_action_proprio_supervision"] = False
        checks["depth_loss_zero"] = False
        checks["minimum_loss_drop"] = False

    if missing_metrics:
        errors.append(f"部分 step 缺少指标：{missing_metrics}")
    if non_finite:
        errors.append(f"部分 step 存在非有限指标：{non_finite}")
    failed_checks = [name for name, passed in checks.items() if not passed]
    if failed_checks:
        errors.append(f"未通过检查：{failed_checks}")
    ok = not errors
    return {
        "schema_version": 1,
        "result": "pass" if ok else "fail",
        "ok": ok,
        "expected_steps": expected_steps,
        "window_size": window_size,
        "minimum_relative_drop": minimum_relative_drop,
        "actual_steps": actual_steps,
        "checks": checks,
        "summaries": summaries,
        "errors": errors,
    }
