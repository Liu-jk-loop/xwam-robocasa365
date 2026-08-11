"""Dependency-light schedule and configuration contract for M6 H100 training."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

from project_tools.multitask_training import REQUIRED_METRICS, parse_console_metrics


def _manifest_digest(payload: dict[str, Any]) -> str:
    import hashlib

    canonical = dict(payload)
    canonical.pop("manifest_digest", None)
    encoded = json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_epoch_schedule(
    *,
    total_samples: int,
    global_batch_size: int,
    num_train_epochs: int,
    trainer_max_steps: int | None = None,
) -> dict[str, int]:
    samples = int(total_samples)
    global_batch = int(global_batch_size)
    epochs = int(num_train_epochs)
    if samples <= 0 or global_batch <= 0 or epochs <= 0:
        raise ValueError(
            "total_samples/global_batch_size/num_train_epochs 必须为正整数"
        )
    steps_per_epoch = samples // global_batch
    if steps_per_epoch <= 0:
        raise ValueError(
            f"样本数不足一个完整 global batch：samples={samples}, gbs={global_batch}"
        )
    total_steps = steps_per_epoch * epochs
    invocation_steps = (
        total_steps if trainer_max_steps is None else int(trainer_max_steps)
    )
    if invocation_steps <= 0 or invocation_steps > total_steps:
        raise ValueError(
            "trainer_max_steps 必须位于正式训练计划内："
            f"invocation={invocation_steps}, total={total_steps}"
        )
    return {
        "total_samples": samples,
        "global_batch_size": global_batch,
        "num_train_epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "num_training_steps": total_steps,
        "trainer_max_steps": invocation_steps,
        "samples_per_epoch_used": steps_per_epoch * global_batch,
        "samples_dropped_per_epoch": samples % global_batch,
    }


def load_epoch_schedule_from_manifest(
    manifest_path: str | Path,
    *,
    global_batch_size: int,
    num_train_epochs: int,
    trainer_max_steps: int | None = None,
) -> dict[str, int | str]:
    path = Path(manifest_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("ok") is not True or payload.get("result") != "pass":
        raise ValueError(f"M6 manifest 尚未通过：{path}")
    if payload.get("scope") != "atomic_only" or payload.get("split") != "pretrain":
        raise ValueError(f"M6 manifest 必须是 atomic_only/pretrain：{path}")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 18:
        raise ValueError(f"M6 manifest 必须包含18个任务：{path}")
    if payload.get("sampling") != "natural_proportional":
        raise ValueError(f"M6 manifest 必须使用 natural_proportional：{path}")
    task_names = [task.get("task_name") for task in tasks if isinstance(task, dict)]
    if len(task_names) != 18 or len(set(task_names)) != 18:
        raise ValueError(f"M6 manifest 必须包含18个唯一任务名：{path}")
    declared_digest = str(payload.get("manifest_digest", ""))
    if not declared_digest or declared_digest != _manifest_digest(payload):
        raise ValueError(f"M6 manifest 摘要缺失或不匹配：{path}")
    schedule = resolve_epoch_schedule(
        total_samples=int(payload.get("total_valid_clips", 0)),
        global_batch_size=global_batch_size,
        num_train_epochs=num_train_epochs,
        trainer_max_steps=trainer_max_steps,
    )
    return {
        **schedule,
        "manifest_path": str(path),
        "manifest_digest": declared_digest,
    }


def validate_global_stats_contract(
    stats_path: str | Path, *, manifest_digest: str
) -> dict[str, Any]:
    path = Path(stats_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "ok": payload.get("ok") is True and payload.get("result") == "pass",
        "atomic_pretrain": payload.get("scope") == "atomic_only"
        and payload.get("split") == "pretrain",
        "task_count": int(payload.get("task_count", -1)) == 18,
        "manifest_digest": payload.get("manifest_digest") == manifest_digest,
        "state_stats": _valid_stats_block(payload.get("observation.state"), 16),
        "action_stats": _valid_stats_block(payload.get("action"), 12),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"M6 global stats 合同失败 {failed}：{path}")
    return {
        "path": str(path),
        "checks": checks,
        "total_frames": int(payload["total_frames"]),
    }


def _valid_stats_block(block: Any, dimension: int) -> bool:
    if not isinstance(block, dict):
        return False
    for key in ("q01", "q99", "min", "max"):
        values = block.get(key)
        if not isinstance(values, list) or len(values) != dimension:
            return False
        try:
            if not all(math.isfinite(float(value)) for value in values):
                return False
        except (TypeError, ValueError):
            return False
    return all(
        float(lower) <= float(upper) for lower, upper in zip(block["q01"], block["q99"])
    )


def validate_m6_formal_training_contract(
    config: Any, *, world_size: int
) -> dict[str, Any]:
    get = config.get
    accelerator = str(get("formal_accelerator", "H100")).upper()
    if accelerator not in {"H100", "GH200"}:
        raise ValueError(f"M6正式训练不支持accelerator={accelerator!r}")
    per_device_batch = int(get("batch_size_per_gpu"))
    accumulate = int(get("accumulate_grad_batches"))
    configured_global_batch = int(get("global_batch_size"))
    actual_global_batch = per_device_batch * int(world_size) * accumulate
    zero_stage = int(get("deepspeed_stage"))
    declared_zero_stage = int(get("formal_zero_stage", -1))
    required_zero_stage = 1
    checks = {
        "four_gpu_world": int(world_size) == 4,
        "global_batch_128": configured_global_batch == 128
        and actual_global_batch == 128,
        "bf16_model_compute": str(get("precision")) == "bf16-mixed",
        "formal_zero_stage_declared": declared_zero_stage in {1, 2},
        "formal_zero1": zero_stage == declared_zero_stage == required_zero_stage,
        "no_optimizer_offload": not bool(get("deepspeed_offload_optimizer")),
        "fp32_optimizer_state_requested": bool(get("deepspeed_fp32_optimizer_states")),
        "communication_overlap": bool(get("deepspeed_overlap_comm")),
        "full_checkpoint_initially": not bool(
            get("deepspeed_exclude_frozen_parameters")
        ),
        "rgb_only": not bool(get("use_depth"))
        and float(get("depth_loss_weight")) == 0.0,
        "five_epochs": int(get("num_train_epochs")) == 5,
        "natural_sampling": str(get("dataset").get("expected_sampling"))
        == "natural_proportional",
        "full_dataset": get("train_subset_size") is None,
        "shuffle_enabled": bool(get("train_shuffle")),
        "gh200_memory_safe_loader_workers": accelerator != "GH200"
        or int(get("num_workers_per_gpu", -1)) == 4,
        "gh200_gradient_checkpointing_enabled": accelerator != "GH200"
        or bool(get("use_gradient_checkpointing", False)),
        "gh200_frozen_text_cache": accelerator != "GH200"
        or bool(get("cache_frozen_text_embeddings", False)),
        "gh200_bounded_text_cache": accelerator != "GH200"
        or int(get("max_cached_text_embeddings", 0)) == 128,
        "gh200_segment_timing": accelerator != "GH200"
        or (
            bool(get("enable_segment_timing", False))
            and int(get("segment_timing_interval_steps", 0)) > 0
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"M6正式训练配置合同失败：{failed}")
    return {
        "checks": checks,
        "accelerator": accelerator,
        "minimum_memory_gib": float(get("formal_minimum_memory_gib", 75.0)),
        "world_size": int(world_size),
        "batch_size_per_gpu": per_device_batch,
        "accumulate_grad_batches": accumulate,
        "global_batch_size": actual_global_batch,
        "num_workers_per_gpu": int(get("num_workers_per_gpu", 0)),
        "use_gradient_checkpointing": bool(
            get("use_gradient_checkpointing", False)
        ),
        "cache_frozen_text_embeddings": bool(
            get("cache_frozen_text_embeddings", False)
        ),
        "max_cached_text_embeddings": int(get("max_cached_text_embeddings", 0)),
        "enable_segment_timing": bool(get("enable_segment_timing", False)),
        "segment_timing_interval_steps": int(
            get("segment_timing_interval_steps", 0)
        ),
        "zero_stage": zero_stage,
        "required_zero_stage": required_zero_stage,
    }


def validate_h100_training_contract(config: Any, *, world_size: int) -> dict[str, Any]:
    """Backward-compatible alias for existing H100 entry points."""
    return validate_m6_formal_training_contract(config, world_size=world_size)


def build_m6_preflight_report(
    manifest_path: str | Path,
    stats_path: str | Path,
    *,
    global_batch_size: int = 128,
    num_train_epochs: int = 5,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        schedule = load_epoch_schedule_from_manifest(
            manifest_path,
            global_batch_size=global_batch_size,
            num_train_epochs=num_train_epochs,
        )
        stats = validate_global_stats_contract(
            stats_path,
            manifest_digest=str(schedule["manifest_digest"]),
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        schedule = None
        stats = None
        errors.append(f"{type(exc).__name__}: {exc}")
    ok = not errors
    return {
        "schema_version": 1,
        "stage": "M6-H100-preflight",
        "result": "pass" if ok else "fail",
        "ok": ok,
        "schedule": schedule,
        "global_stats": stats,
        "errors": errors,
    }


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
    root = Path(path).expanduser().resolve()
    optimizer_pattern = re.compile(
        r"^(?:[^/]+_)?zero_pp_rank_(\d+)_mp_rank_\d+_optim_states\.pt$"
    )
    model_states = sorted(root.glob("**/mp_rank_*_model_states.pt"))
    optimizer_states = sorted(
        item
        for item in root.glob("**/*zero_pp_rank_*_optim_states.pt")
        if optimizer_pattern.fullmatch(item.name)
    )
    optimizer_ranks = sorted(
        int(optimizer_pattern.fullmatch(item.name).group(1))
        for item in optimizer_states
    )
    checks = {
        "checkpoint_directory": root.is_dir(),
        "model_state_present": bool(model_states)
        and all(item.stat().st_size > 0 for item in model_states),
        "four_optimizer_shards": optimizer_ranks == [0, 1, 2, 3]
        and all(item.stat().st_size > 0 for item in optimizer_states),
    }
    return {
        "path": str(root),
        "model_states": [str(item) for item in model_states],
        "optimizer_states": [str(item) for item in optimizer_states],
        "optimizer_ranks": optimizer_ranks,
        "checks": checks,
    }


def resolve_m6_formal_chunk(
    checkpoint_root: str | Path | list[str | Path],
    *,
    total_steps: int,
    chunk_steps: int,
) -> dict[str, Any]:
    """Select the newest complete checkpoint and the next absolute step target."""
    root_values = (
        [checkpoint_root]
        if isinstance(checkpoint_root, (str, Path))
        else list(checkpoint_root)
    )
    if not root_values:
        raise ValueError("至少需要一个checkpoint root")
    roots = [Path(value).expanduser().resolve() for value in root_values]
    if len(set(roots)) != len(roots):
        raise ValueError(f"checkpoint roots不能重复：{roots}")
    total = int(total_steps)
    chunk = int(chunk_steps)
    if total <= 0 or chunk <= 0:
        raise ValueError("total_steps/chunk_steps 必须为正整数")
    pattern = re.compile(r"^(?:epoch=\d+-step=|final-step=)(\d+)\.ckpt$")
    complete: list[tuple[int, int, int, Path]] = []
    incomplete: list[dict[str, Any]] = []
    for root_index, root in enumerate(roots):
        if not root.exists():
            continue
        for candidate in sorted(root.iterdir()):
            match = pattern.fullmatch(candidate.name)
            if match is None or not candidate.is_dir():
                continue
            step = int(match.group(1))
            if step > total:
                raise ValueError(
                    f"检查点步数超过正式计划：step={step}, total={total}, path={candidate}"
                )
            layout = _checkpoint_layout(candidate)
            if all(layout["checks"].values()):
                final_priority = 1 if candidate.name.startswith("final-step=") else 0
                complete.append(
                    (step, final_priority, root_index, candidate.resolve())
                )
            else:
                incomplete.append(
                    {
                        "step": step,
                        "path": str(candidate.resolve()),
                        "checkpoint_root": str(root),
                        "checks": layout["checks"],
                    }
                )
    if complete:
        completed_step, _, _, resume_path = max(
            complete, key=lambda item: (item[0], item[1], item[2], str(item[3]))
        )
        resume_checkpoint = str(resume_path)
    else:
        completed_step = 0
        resume_checkpoint = None
    already_complete = completed_step == total
    if already_complete:
        target_step = total
    else:
        target_step = min(((completed_step // chunk) + 1) * chunk, total)
        if target_step <= completed_step:
            raise ValueError(
                f"无法推进正式训练：completed={completed_step}, target={target_step}"
            )
    return {
        "checkpoint_root": str(roots[0]),
        "checkpoint_roots": [str(root) for root in roots],
        "total_steps": total,
        "chunk_steps": chunk,
        "completed_step": completed_step,
        "target_step": target_step,
        "resume_checkpoint": resume_checkpoint,
        "final_chunk": not already_complete and target_step == total,
        "already_complete": already_complete,
        "complete_checkpoint_count": len(complete),
        "incomplete_checkpoints": incomplete,
    }


def quarantine_m6_incomplete_checkpoints(
    plan: dict[str, Any],
    quarantine_root: str | Path | list[str | Path],
) -> dict[str, Any]:
    """Move incomplete checkpoint directories aside without deleting evidence."""
    updated = dict(plan)
    records = list(plan.get("incomplete_checkpoints") or [])
    quarantined: list[dict[str, Any]] = []
    if records:
        checkpoint_roots = list(
            plan.get("checkpoint_roots") or [plan["checkpoint_root"]]
        )
        quarantine_values = (
            [quarantine_root]
            if isinstance(quarantine_root, (str, Path))
            else list(quarantine_root)
        )
        if len(checkpoint_roots) != len(quarantine_values):
            raise ValueError(
                "checkpoint roots与quarantine roots数量必须一致："
                f"{len(checkpoint_roots)} != {len(quarantine_values)}"
            )
        quarantine_by_checkpoint_root = {
            str(Path(checkpoint).expanduser().resolve()): Path(quarantine)
            .expanduser()
            .resolve()
            for checkpoint, quarantine in zip(
                checkpoint_roots, quarantine_values, strict=True
            )
        }
        created_roots: set[Path] = set()
        for record in records:
            source = Path(str(record["path"])).expanduser().resolve()
            source_root = str(
                Path(record.get("checkpoint_root") or plan["checkpoint_root"])
                .expanduser()
                .resolve()
            )
            root = quarantine_by_checkpoint_root[source_root]
            if root not in created_roots:
                root.mkdir(parents=True, exist_ok=False)
                created_roots.add(root)
            destination = root / source.name
            if not source.is_dir():
                raise ValueError(f"待隔离的不完整checkpoint不存在：{source}")
            if destination.exists():
                raise ValueError(f"隔离目标已存在：{destination}")
            os.replace(source, destination)
            quarantined.append(
                {
                    **record,
                    "original_path": str(source),
                    "quarantine_path": str(destination),
                }
            )
    updated["quarantined_checkpoints"] = quarantined
    return updated


def _optimizer_audit(directory: str | Path) -> dict[str, Any]:
    root = Path(directory).expanduser().resolve()
    reports = [
        _load_json(path) for path in sorted(root.glob("optimizer_state_rank_*.json"))
    ]
    ranks = sorted(int(report.get("global_rank", -1)) for report in reports)
    checks = {
        "four_rank_reports": len(reports) == 4 and ranks == [0, 1, 2, 3],
        "all_pass": bool(reports)
        and all(report.get("result") == "pass" for report in reports),
        "actual_fp32_only": bool(reports)
        and all(report.get("fp32_only") is True for report in reports),
        "nonempty_state": bool(reports)
        and all(int(report.get("floating_state_tensors", 0)) > 0 for report in reports),
    }
    return {"path": str(root), "reports": reports, "checks": checks}


def _metrics_audit(log_path: str | Path) -> dict[str, Any]:
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
    checks = {
        "metrics_present": bool(records) and not missing,
        "metrics_finite": bool(records) and not non_finite and not errors,
        "depth_loss_zero": bool(records)
        and all(
            abs(float(record["metrics"].get("train/depth_loss", math.inf))) <= 1e-8
            for record in records
        ),
        "supervision_ratio_valid": bool(ratios)
        and all(0.0 <= ratio <= 1.0 for ratio in ratios),
        "supervised_batch_observed": any(ratio > 0.0 for ratio in ratios),
    }
    return {
        "path": str(path),
        "steps": [int(record["step"]) for record in records],
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
    expected_step: int,
    expect_resume: bool,
    require_final_checkpoint: bool,
) -> dict[str, Any]:
    metadata = _load_json(metadata_path)
    result = _load_json(result_path)
    events = _load_events(events_path)
    optimizer = _optimizer_audit(optimizer_dir)
    metrics = _metrics_audit(log_path)
    training = metadata.get("training") or {}
    runtime = metadata.get("formal_runtime") or metadata.get("h100_runtime") or {}
    formal_contract = (
        metadata.get("formal_contract") or metadata.get("h100_contract") or {}
    )
    git = metadata.get("git") or {}
    sampler = (metadata.get("dataset") or {}).get("sampler_provenance") or {}
    wandb = (metadata.get("tracking") or {}).get("wandb") or {}
    checkpoint_storage = (metadata.get("checkpoint") or {}).get("storage") or {}
    complete_steps = {
        int(event.get("global_step", -1))
        for event in events
        if event.get("event")
        in {"checkpoint_save_complete", "final_checkpoint_save_complete"}
    }
    completed_checkpoint_paths = [
        str(event.get("filepath"))
        for event in events
        if event.get("event")
        in {"checkpoint_save_complete", "final_checkpoint_save_complete"}
        and int(event.get("global_step", -1)) == int(expected_step)
        and event.get("filepath")
    ]
    completed_checkpoint = (
        str(Path(completed_checkpoint_paths[-1]).expanduser().resolve())
        if completed_checkpoint_paths
        else ""
    )
    completed_checkpoint_events = [
        {
            "event": event.get("event"),
            "filepath": str(event.get("filepath")),
            "checkpoint_tier": event.get("checkpoint_tier"),
            "global_step": int(event.get("global_step", -1)),
        }
        for event in events
        if event.get("event")
        in {"checkpoint_save_complete", "final_checkpoint_save_complete"}
        and int(event.get("global_step", -1)) == int(expected_step)
        and event.get("filepath")
    ]
    checkpoint = _checkpoint_layout(completed_checkpoint) if completed_checkpoint else None
    resume_value = result.get("resume_checkpoint")
    checks = {
        "run_pass": result.get("result") == "pass",
        "expected_global_step": int(result.get("global_step", -1))
        == int(expected_step),
        "trainer_limit_reached": int(result.get("trainer_max_steps", -1))
        == int(expected_step),
        "resume_mode": bool(resume_value) is bool(expect_resume),
        "formal_contract_pass": bool(formal_contract.get("checks"))
        and all(formal_contract["checks"].values()),
        "formal_runtime_pass": bool(runtime.get("checks"))
        and all(runtime["checks"].values()),
        "clean_git": bool(git.get("commit")) and git.get("dirty") is False,
        "five_epoch_schedule": int(training.get("num_train_epochs", -1)) == 5,
        "global_batch_128": int(training.get("global_batch_size", -1)) == 128,
        "epoch_aligned_sampler": sampler.get("type") == "epoch_aligned_distributed"
        and int(sampler.get("dropped_per_epoch", -1))
        == int(training.get("samples_dropped_per_epoch", -2)),
        "checkpoint_complete": int(expected_step) in complete_steps,
        "checkpoint_layout": bool(checkpoint)
        and all(checkpoint["checks"].values()),
        "optimizer_state_pass": all(optimizer["checks"].values()),
        "metrics_pass": all(metrics["checks"].values()),
    }
    if require_final_checkpoint:
        checks["final_checkpoint_complete"] = any(
            event.get("event") == "final_checkpoint_save_complete"
            and int(event.get("global_step", -1)) == int(expected_step)
            for event in events
        )
    return {
        "metadata": str(Path(metadata_path).expanduser().resolve()),
        "result": str(Path(result_path).expanduser().resolve()),
        "events": str(Path(events_path).expanduser().resolve()),
        "run_id": metadata.get("run_id"),
        "git_commit": git.get("commit"),
        "accelerator": formal_contract.get("accelerator"),
        "zero_stage": formal_contract.get("zero_stage"),
        "manifest_digest": training.get("manifest_digest"),
        "resume_checkpoint": resume_value,
        "completed_checkpoint": completed_checkpoint,
        "schedule": training,
        "checks": checks,
        "optimizer": optimizer,
        "metrics": metrics,
        "checkpoint": checkpoint,
        "wandb": wandb,
        "checkpoint_storage": checkpoint_storage,
        "completed_checkpoint_events": completed_checkpoint_events,
    }


def build_m6_gate_report(
    *, initial: dict[str, str], resumed: dict[str, str]
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        initial_report = _run_contract(
            **initial,
            expected_step=2,
            expect_resume=False,
            require_final_checkpoint=False,
        )
        resumed_report = _run_contract(
            **resumed,
            expected_step=4,
            expect_resume=True,
            require_final_checkpoint=False,
        )
        cross_checks = {
            "different_run_ids": bool(initial_report["run_id"])
            and initial_report["run_id"] != resumed_report["run_id"],
            "same_clean_commit": bool(initial_report["git_commit"])
            and initial_report["git_commit"] == resumed_report["git_commit"],
            "same_accelerator": bool(initial_report["accelerator"])
            and initial_report["accelerator"] == resumed_report["accelerator"],
            "same_zero_stage": initial_report["zero_stage"] in {1, 2}
            and initial_report["zero_stage"] == resumed_report["zero_stage"],
            "same_manifest": bool(initial_report["manifest_digest"])
            and initial_report["manifest_digest"] == resumed_report["manifest_digest"],
            "same_full_schedule": initial_report["schedule"].get("num_training_steps")
            == resumed_report["schedule"].get("num_training_steps"),
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
        "stage": "M6-formal-accelerator-gate",
        "result": "pass" if ok else "fail",
        "ok": ok,
        "initial": initial_report,
        "resumed": resumed_report,
        "checks": cross_checks,
        "errors": errors,
    }


def build_m6_formal_chunk_report(
    *,
    metadata_path: str,
    result_path: str,
    events_path: str,
    optimizer_dir: str,
    log_path: str,
    expected_step: int,
    expected_resume_checkpoint: str | None,
    expected_wandb_run_id: str,
    expected_rolling_checkpoint_root: str,
    expected_durable_checkpoint_root: str,
) -> dict[str, Any]:
    """Audit one normally completed restartable formal-training chunk."""
    errors: list[str] = []
    try:
        expected = int(expected_step)
        expected_resume = (
            str(Path(expected_resume_checkpoint).expanduser().resolve())
            if expected_resume_checkpoint
            else None
        )
        run = _run_contract(
            metadata_path=metadata_path,
            result_path=result_path,
            events_path=events_path,
            optimizer_dir=optimizer_dir,
            log_path=log_path,
            expected_step=expected,
            expect_resume=expected_resume is not None,
            require_final_checkpoint=False,
        )
        actual_resume = (
            str(Path(run["resume_checkpoint"]).expanduser().resolve())
            if run["resume_checkpoint"]
            else None
        )
        schedule_total = int(run["schedule"].get("num_training_steps", -1))
        rolling_storage = run["checkpoint_storage"].get("rolling") or {}
        durable_storage = run["checkpoint_storage"].get("durable") or {}
        expected_rolling_root = str(
            Path(expected_rolling_checkpoint_root).expanduser().resolve()
        )
        expected_durable_root = str(
            Path(expected_durable_checkpoint_root).expanduser().resolve()
        )
        rolling_checkpoint_written = any(
            event.get("checkpoint_tier") == "rolling"
            and Path(event["filepath"])
            .expanduser()
            .resolve()
            .is_relative_to(Path(expected_rolling_root))
            for event in run["completed_checkpoint_events"]
        )
        durable_checkpoint_written = any(
            event.get("checkpoint_tier") == "durable"
            and Path(event["filepath"])
            .expanduser()
            .resolve()
            .is_relative_to(Path(expected_durable_root))
            for event in run["completed_checkpoint_events"]
        )
        final_durable_written = any(
            event.get("checkpoint_tier") == "final_durable"
            and Path(event["filepath"])
            .expanduser()
            .resolve()
            .is_relative_to(Path(expected_durable_root))
            for event in run["completed_checkpoint_events"]
        )
        checks = {
            "positive_chunk_target": expected > 0,
            "within_formal_schedule": expected <= schedule_total,
            "exact_resume_source": actual_resume == expected_resume,
            "wandb_online": run["wandb"].get("enabled") is True
            and run["wandb"].get("mode") == "online",
            "wandb_persistent_run": bool(run["wandb"].get("run_id"))
            and run["wandb"].get("resume") == "allow",
            "wandb_api_key_auth": run["wandb"].get("auth") == "api_key_env",
            "rolling_checkpoint_policy": rolling_storage.get("directory")
            == expected_rolling_root
            and int(rolling_storage.get("interval_steps", -1)) == 500
            and int(rolling_storage.get("save_top_k", -1)) == 5
            and rolling_storage.get("monitor") == "step"
            and rolling_storage.get("mode") == "max",
            "durable_checkpoint_policy": durable_storage.get("directory")
            == expected_durable_root
            and int(durable_storage.get("interval_steps", -1)) == 3000
            and int(durable_storage.get("save_top_k", 0)) == -1
            and run["checkpoint_storage"].get("final_directory")
            == expected_durable_root,
            "rolling_checkpoint_written": expected == schedule_total
            or rolling_checkpoint_written,
            "durable_checkpoint_written_when_due": expected % 3000 != 0
            or durable_checkpoint_written,
            "final_checkpoint_written_to_durable": expected != schedule_total
            or final_durable_written,
            "expected_wandb_run": bool(expected_wandb_run_id)
            and run["wandb"].get("run_id") == expected_wandb_run_id,
            "all_run_checks": all(run["checks"].values()),
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        run = None
        checks = {"artifact_loading": False}
        errors.append(f"{type(exc).__name__}: {exc}")
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        errors.append(f"未通过检查：{failed}")
    ok = not errors
    return {
        "schema_version": 1,
        "stage": "M6-formal-training-chunk",
        "result": "pass" if ok else "fail",
        "ok": ok,
        "run": run,
        "checks": checks,
        "errors": errors,
    }


def build_m6_formal_report(
    *,
    metadata_path: str,
    result_path: str,
    events_path: str,
    optimizer_dir: str,
    log_path: str,
    expected_wandb_run_id: str | None = None,
    expected_rolling_checkpoint_root: str | None = None,
    expected_durable_checkpoint_root: str | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        metadata = _load_json(metadata_path)
        expected_step = int(
            (metadata.get("training") or {}).get("num_training_steps", -1)
        )
        run = _run_contract(
            metadata_path=metadata_path,
            result_path=result_path,
            events_path=events_path,
            optimizer_dir=optimizer_dir,
            log_path=log_path,
            expected_step=expected_step,
            expect_resume=bool(_load_json(result_path).get("resume_checkpoint")),
            require_final_checkpoint=True,
        )
        checks = {
            "positive_formal_step_count": expected_step > 4,
            "all_run_checks": all(run["checks"].values()),
        }
        if expected_wandb_run_id is not None:
            checks.update(
                {
                    "wandb_online": run["wandb"].get("enabled") is True
                    and run["wandb"].get("mode") == "online",
                    "wandb_persistent_run": bool(run["wandb"].get("run_id"))
                    and run["wandb"].get("resume") == "allow",
                    "wandb_api_key_auth": run["wandb"].get("auth")
                    == "api_key_env",
                    "expected_wandb_run": run["wandb"].get("run_id")
                    == expected_wandb_run_id,
                }
            )
        if (
            expected_rolling_checkpoint_root is not None
            and expected_durable_checkpoint_root is not None
        ):
            rolling_storage = run["checkpoint_storage"].get("rolling") or {}
            durable_storage = run["checkpoint_storage"].get("durable") or {}
            expected_rolling_root = str(
                Path(expected_rolling_checkpoint_root).expanduser().resolve()
            )
            expected_durable_root = str(
                Path(expected_durable_checkpoint_root).expanduser().resolve()
            )
            checks.update(
                {
                    "rolling_checkpoint_policy": rolling_storage.get("directory")
                    == expected_rolling_root
                    and int(rolling_storage.get("interval_steps", -1)) == 500
                    and int(rolling_storage.get("save_top_k", -1)) == 5
                    and rolling_storage.get("monitor") == "step"
                    and rolling_storage.get("mode") == "max",
                    "durable_checkpoint_policy": durable_storage.get("directory")
                    == expected_durable_root
                    and int(durable_storage.get("interval_steps", -1)) == 3000
                    and int(durable_storage.get("save_top_k", 0)) == -1
                    and run["checkpoint_storage"].get("final_directory")
                    == expected_durable_root,
                }
            )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        run = None
        checks = {"artifact_loading": False}
        errors.append(f"{type(exc).__name__}: {exc}")
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        errors.append(f"未通过检查：{failed}")
    ok = not errors
    return {
        "schema_version": 1,
        "stage": "M6-formal-training",
        "result": "pass" if ok else "fail",
        "ok": ok,
        "run": run,
        "checks": checks,
        "errors": errors,
    }
