"""Dependency-light evidence audit for the Clariden 4xGH200 resume gate."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from project_tools.multitask_training import REQUIRED_METRICS, parse_console_metrics


def _load_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是对象：{resolved}")
    return payload


def _load_events(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    events = []
    for line_number, line in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        event = json.loads(line)
        if not isinstance(event, dict):
            raise ValueError(f"JSONL 第 {line_number} 行不是对象：{resolved}")
        events.append(event)
    return events


def _checkpoint_layout(path: str | Path) -> dict[str, Any]:
    if not str(path).strip():
        return {
            "path": None,
            "model_states": [],
            "optimizer_states": [],
            "checks": {
                "checkpoint_directory": False,
                "model_state_present": False,
                "four_optimizer_shards": False,
            },
        }
    root = Path(path).expanduser().resolve()
    model_states = sorted(root.glob("**/mp_rank_*_model_states.pt"))
    optimizer_states = sorted(root.glob("**/zero_pp_rank_*_optim_states.pt"))
    checks = {
        "checkpoint_directory": root.is_dir(),
        "model_state_present": bool(model_states)
        and all(item.stat().st_size > 0 for item in model_states),
        "four_optimizer_shards": len(optimizer_states) == 4
        and all(item.stat().st_size > 0 for item in optimizer_states),
    }
    return {
        "path": str(root),
        "model_states": [str(item) for item in model_states],
        "optimizer_states": [str(item) for item in optimizer_states],
        "checks": checks,
    }


def _optimizer_audit(directory: str | Path) -> dict[str, Any]:
    root = Path(directory).expanduser().resolve()
    reports = [
        _load_json(path) for path in sorted(root.glob("optimizer_state_rank_*.json"))
    ]
    ranks = sorted(int(report.get("global_rank", -1)) for report in reports)
    checks = {
        "four_rank_reports": len(reports) == 4 and ranks == [0, 1, 2, 3],
        "world_size_four": bool(reports)
        and all(int(report.get("world_size", -1)) == 4 for report in reports),
        "all_pass": bool(reports)
        and all(report.get("result") == "pass" for report in reports),
        "actual_fp32_only": bool(reports)
        and all(report.get("fp32_only") is True for report in reports),
        "nonempty_state": bool(reports)
        and all(int(report.get("floating_state_tensors", 0)) > 0 for report in reports),
    }
    return {
        "path": str(root),
        "reports": reports,
        "checks": checks,
    }


def _metrics_audit(
    log_path: str | Path, *, expected_steps: list[int]
) -> dict[str, Any]:
    path = Path(log_path).expanduser().resolve()
    records, errors = parse_console_metrics(
        path.read_text(encoding="utf-8", errors="replace")
    )
    required = tuple(
        metric for metric in REQUIRED_METRICS if metric != "train/task_index"
    )
    missing = {
        int(record["step"]): [
            metric for metric in required if metric not in record["metrics"]
        ]
        for record in records
        if any(metric not in record["metrics"] for metric in required)
    }
    non_finite = {
        int(record["step"]): [
            metric
            for metric in required
            if metric in record["metrics"]
            and not math.isfinite(float(record["metrics"][metric]))
        ]
        for record in records
        if any(
            metric in record["metrics"]
            and not math.isfinite(float(record["metrics"][metric]))
            for metric in required
        )
    }
    ratios = [
        float(record["metrics"]["train/action_proprio_supervision_ratio"])
        for record in records
        if "train/action_proprio_supervision_ratio" in record["metrics"]
    ]
    steps = [int(record["step"]) for record in records]
    checks = {
        "expected_steps": steps == expected_steps,
        "metrics_present": bool(records) and not missing,
        "metrics_finite": bool(records) and not non_finite and not errors,
        "depth_loss_zero": bool(records)
        and all(
            abs(float(record["metrics"].get("train/depth_loss", math.inf))) <= 1e-8
            for record in records
        ),
        "all_batches_supervised": bool(ratios)
        and all(abs(value - 1.0) <= 1e-8 for value in ratios),
    }
    return {
        "path": str(path),
        "steps": steps,
        "checks": checks,
        "parser_errors": errors,
        "missing": missing,
        "non_finite": non_finite,
    }


def _run_contract(
    *,
    metadata_path: str | Path,
    result_path: str | Path,
    events_path: str | Path,
    optimizer_dir: str | Path,
    log_path: str | Path,
    expected_global_step: int,
    expected_metric_steps: list[int],
    expect_resume: bool,
) -> dict[str, Any]:
    metadata = _load_json(metadata_path)
    result = _load_json(result_path)
    events = _load_events(events_path)
    optimizer = _optimizer_audit(optimizer_dir)
    metrics = _metrics_audit(log_path, expected_steps=expected_metric_steps)
    topology = metadata.get("topology") or {}
    environment = metadata.get("environment") or {}
    deepspeed = metadata.get("deepspeed") or {}
    optimizer_config = metadata.get("optimizer") or {}
    training = metadata.get("training") or {}
    dataset = metadata.get("dataset") or {}
    git = metadata.get("git") or {}
    complete_steps = {
        int(event.get("global_step", -1))
        for event in events
        if event.get("event") == "checkpoint_save_complete"
    }
    completed_checkpoint_paths = [
        str(event.get("filepath"))
        for event in events
        if event.get("event") == "checkpoint_save_complete"
        and int(event.get("global_step", -1)) == expected_global_step
        and event.get("filepath")
    ]
    checkpoint_path = (
        completed_checkpoint_paths[-1] if completed_checkpoint_paths else ""
    )
    checkpoint = _checkpoint_layout(checkpoint_path)
    gpu_names = environment.get("gpu_names") or []
    resume_checkpoint = result.get("resume_checkpoint")
    resume_module_load = result.get("resume_module_load") or {}
    checks = {
        "run_pass": result.get("result") == "pass" and result.get("error") is None,
        "expected_global_step": int(result.get("global_step", -1))
        == expected_global_step,
        "trainer_limit_reached": int(result.get("trainer_max_steps", -1))
        == expected_global_step,
        "four_gpu_single_node": int(topology.get("world_size", -1)) == 4
        and int(topology.get("num_nodes", -1)) == 1
        and int(topology.get("visible_devices", -1)) == 4,
        "four_gh200": len(gpu_names) == 4
        and all("GH200" in str(name).upper() for name in gpu_names),
        "zero2_cpu_offload_fp32": int(deepspeed.get("stage", -1)) == 2
        and deepspeed.get("offload_optimizer") is True
        and deepspeed.get("exclude_frozen_parameters") is True
        and optimizer_config.get("backend") == "deepspeed_cpu_adam"
        and optimizer_config.get("fp32_optimizer_states") is True,
        "fixed_four_step_schedule": int(training.get("num_training_steps", -1)) == 4,
        "fixed_atomic_subset": dataset.get("task_name") == "CloseFridge"
        and dataset.get("use_depth") is False
        and dataset.get("subset_indices") == list(range(8))
        and dataset.get("shuffle") is False,
        "clean_git": bool(git.get("commit")) and git.get("dirty") is False,
        "resume_mode": bool(resume_checkpoint) is expect_resume,
        "checkpoint_complete": expected_global_step in complete_steps,
        "checkpoint_layout": all(checkpoint["checks"].values()),
        "optimizer_state": all(optimizer["checks"].values()),
        "finite_rgb_only_metrics": all(metrics["checks"].values()),
    }
    if expect_resume:
        checks["excluded_frozen_resume_strict"] = (
            resume_module_load.get("mode") == "excluded_frozen_parameters"
            and int(resume_module_load.get("missing_frozen_count", 0)) > 0
            and int(resume_module_load.get("unexpected_count", -1)) == 0
        )
    return {
        "run_id": metadata.get("run_id"),
        "git_commit": git.get("commit"),
        "dataset_path": dataset.get("path"),
        "schedule": training,
        "resume_checkpoint": resume_checkpoint,
        "last_checkpoint": result.get("last_checkpoint"),
        "completed_checkpoint": checkpoint.get("path"),
        "checks": checks,
        "checkpoint": checkpoint,
        "optimizer": optimizer,
        "metrics": metrics,
        "resume_module_load": resume_module_load or None,
    }


def build_clariden_4gpu_resume_report(
    *,
    initial: dict[str, str],
    resumed: dict[str, str],
    commit_compatibility: dict[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        initial_report = _run_contract(
            **initial,
            expected_global_step=2,
            expected_metric_steps=[0, 1],
            expect_resume=False,
        )
        resumed_report = _run_contract(
            **resumed,
            expected_global_step=4,
            expected_metric_steps=[2, 3],
            expect_resume=True,
        )
        same_commit = bool(initial_report["git_commit"]) and (
            initial_report["git_commit"] == resumed_report["git_commit"]
        )
        compatible_commit_delta = bool(commit_compatibility) and (
            commit_compatibility.get("ok") is True
            and commit_compatibility.get("initial_commit")
            == initial_report["git_commit"]
            and commit_compatibility.get("resumed_commit")
            == resumed_report["git_commit"]
        )
        cross_checks = {
            "different_run_ids": bool(initial_report["run_id"])
            and bool(resumed_report["run_id"])
            and initial_report["run_id"] != resumed_report["run_id"],
            "compatible_training_source": same_commit or compatible_commit_delta,
            "same_dataset": bool(initial_report["dataset_path"])
            and initial_report["dataset_path"] == resumed_report["dataset_path"],
            "same_scheduler_horizon": initial_report["schedule"].get(
                "num_training_steps"
            )
            == resumed_report["schedule"].get("num_training_steps")
            == 4,
            "resume_uses_initial_checkpoint": bool(
                resumed_report["resume_checkpoint"]
            )
            and str(
                Path(resumed_report["resume_checkpoint"]).expanduser().resolve()
            )
            == initial_report["completed_checkpoint"],
            "all_run_checks": all(initial_report["checks"].values())
            and all(resumed_report["checks"].values()),
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        initial_report = None
        resumed_report = None
        cross_checks = {"artifact_loading": False}
        errors.append(f"{type(exc).__name__}: {exc}")
    failed = [name for name, passed in cross_checks.items() if not passed]
    if failed:
        errors.append(f"未通过检查：{failed}")
    ok = not errors
    return {
        "schema_version": 1,
        "stage": "Clariden-GH200-4GPU-resume-gate",
        "result": "pass" if ok else "fail",
        "ok": ok,
        "initial": initial_report,
        "resumed": resumed_report,
        "commit_compatibility": commit_compatibility,
        "checks": cross_checks,
        "errors": errors,
    }
