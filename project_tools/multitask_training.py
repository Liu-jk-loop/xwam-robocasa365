"""Dependency-free parsing and validation for the M3.2 three-task smoke log."""

from __future__ import annotations

import math
import re
from typing import Any, Sequence


METRICS_PATTERN = re.compile(r"\[METRICS\]\s+Step:\s+(\d+)\s+-\s+(.*)$")
REQUIRED_METRICS = (
    "train/video_loss",
    "train/action_loss",
    "train/proprio_loss",
    "train/action_proprio_supervision_ratio",
    "train/task_index",
    "train/depth_loss",
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
                errors.append(f"第 {line_number} 行的 {key} 不是浮点数：{raw_value}")
        records.append({"step": step, "metrics": metrics})
    return records, errors


def build_multitask_short_report(
    log_text: str,
    *,
    task_names: Sequence[str],
    expected_steps: int = 12,
    max_task_count_spread: int = 2,
) -> dict[str, Any]:
    """Validate wiring and numerical sanity without imposing a convergence target."""
    names = [str(name) for name in task_names]
    expected_steps = int(expected_steps)
    max_task_count_spread = int(max_task_count_spread)
    if expected_steps <= 0:
        raise ValueError("expected_steps 必须为正整数")
    if not names or expected_steps % len(names) != 0:
        raise ValueError("expected_steps 必须能被非空任务数整除")
    if max_task_count_spread < 0:
        raise ValueError("max_task_count_spread 不能为负数")

    records, errors = parse_console_metrics(log_text)
    records.sort(key=lambda item: item["step"])
    actual_steps = [record["step"] for record in records]
    expected_step_ids = list(range(expected_steps))
    checks: dict[str, bool] = {
        "three_unique_atomic_tasks": len(names) == 3 and len(set(names)) == 3,
        "clean_action_ratio_preserved": bool(
            re.search(r"['\"]?clean_action_ratio['\"]?\s*:\s*0\.5(?:0*)\b", log_text)
        ),
        "checkpointing_disabled": bool(
            re.search(r"['\"]?enable_checkpointing['\"]?\s*:\s*False\b", log_text)
        ),
        "complete_step_sequence": actual_steps == expected_step_ids,
        "trainer_reached_max_steps": f"`max_steps={expected_steps}` reached" in log_text,
        "run_result_pass": "Run result:" in log_text and "(pass)" in log_text,
        "dataset_provenance_present": "Training dataset provenance:" in log_text
        and all(name in log_text for name in names),
    }

    missing_metrics: dict[int, list[str]] = {}
    non_finite: dict[int, list[str]] = {}
    for record in records:
        metrics = record["metrics"]
        missing = [key for key in REQUIRED_METRICS if key not in metrics]
        invalid = [
            key
            for key in REQUIRED_METRICS
            if key in metrics and not math.isfinite(metrics[key])
        ]
        if missing:
            missing_metrics[record["step"]] = missing
        if invalid:
            non_finite[record["step"]] = invalid
    checks["required_metrics_present"] = not missing_metrics
    checks["all_metrics_finite"] = not non_finite

    observed_task_indices: list[int] = []
    task_counts: list[int] = []
    supervised_steps: list[int] = []
    unsupervised_steps: list[int] = []
    can_validate_metrics = (
        checks["complete_step_sequence"]
        and checks["required_metrics_present"]
        and checks["all_metrics_finite"]
    )
    if can_validate_metrics:
        observed_task_indices = [
            int(round(record["metrics"]["train/task_index"])) for record in records
        ]
        raw_task_indices = [
            record["metrics"]["train/task_index"] for record in records
        ]
        checks["valid_task_indices"] = all(
            abs(raw - rounded) <= 1e-6 and 0 <= rounded < len(names)
            for raw, rounded in zip(raw_task_indices, observed_task_indices)
        )
        task_counts = [
            observed_task_indices.count(index) for index in range(len(names))
        ]
        checks["all_tasks_observed"] = all(count > 0 for count in task_counts)
        checks["task_count_spread_within_tolerance"] = (
            max(task_counts) - min(task_counts) <= max_task_count_spread
        )

        supervision_values = [
            record["metrics"]["train/action_proprio_supervision_ratio"]
            for record in records
        ]
        checks["binary_supervision_ratio"] = all(
            abs(value) <= 1e-6 or abs(value - 1.0) <= 1e-6
            for value in supervision_values
        )
        supervised_steps = [
            record["step"]
            for record, value in zip(records, supervision_values)
            if abs(value - 1.0) <= 1e-6
        ]
        unsupervised_steps = [
            record["step"]
            for record, value in zip(records, supervision_values)
            if abs(value) <= 1e-6
        ]
        checks["both_supervision_branches_observed"] = bool(
            supervised_steps and unsupervised_steps
        )
        checks["supervision_matches_losses"] = all(
            (
                metrics["train/action_loss"] > 0
                and metrics["train/proprio_loss"] > 0
            )
            if abs(metrics["train/action_proprio_supervision_ratio"] - 1.0) <= 1e-6
            else (
                abs(metrics["train/action_loss"]) <= 1e-8
                and abs(metrics["train/proprio_loss"]) <= 1e-8
            )
            for metrics in (record["metrics"] for record in records)
        )
        checks["depth_loss_zero"] = all(
            abs(record["metrics"]["train/depth_loss"]) <= 1e-8
            for record in records
        )
    else:
        checks.update(
            {
                "valid_task_indices": False,
                "all_tasks_observed": False,
                "task_count_spread_within_tolerance": False,
                "binary_supervision_ratio": False,
                "both_supervision_branches_observed": False,
                "supervision_matches_losses": False,
                "depth_loss_zero": False,
            }
        )

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
        "max_task_count_spread": max_task_count_spread,
        "task_names": names,
        "observed_task_indices": observed_task_indices,
        "task_counts": {
            name: task_counts[index] for index, name in enumerate(names)
        }
        if task_counts
        else {},
        "supervised_steps": supervised_steps,
        "unsupervised_steps": unsupervised_steps,
        "actual_steps": actual_steps,
        "checks": checks,
        "errors": errors,
    }
