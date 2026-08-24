"""Dependency-free M3 training-run validation and provenance helpers."""

from __future__ import annotations

import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def cosine_learning_rate_at_step(
    *, base_lr: float, num_warmup_steps: int, num_training_steps: int, step: int
) -> float:
    """Return the Transformers half-cosine LR at one absolute optimizer step."""
    learning_rate = float(base_lr)
    warmup = int(num_warmup_steps)
    total = int(num_training_steps)
    current = int(step)
    if learning_rate <= 0 or warmup < 0 or total <= warmup or current < 0:
        raise ValueError("cosine学习率参数无效")
    if current < warmup:
        return learning_rate * current / max(1, warmup)
    progress = (current - warmup) / (total - warmup)
    return learning_rate * max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def resolve_learning_rate_schedule(config: Any) -> dict[str, Any]:
    """Freeze the regular cosine or an LR-continuous resume schedule."""
    get = config.get
    mode = str(get("lr_schedule_mode", "cosine"))
    base_lr = float(get("lr"))
    total_steps = int(get("num_training_steps"))
    warmup_steps = int(get("num_warmup_steps"))
    if mode == "cosine":
        return {
            "mode": mode,
            "base_learning_rate": base_lr,
            "num_warmup_steps": warmup_steps,
            "num_training_steps": total_steps,
        }
    if mode != "constant_resume":
        raise ValueError(f"不支持lr_schedule_mode={mode!r}")

    start_step = int(get("lr_continuation_start_step"))
    end_step = int(get("lr_continuation_end_step"))
    expected_resume_step = int(
        get("lr_continuation_expected_resume_step", start_step)
    )
    source_base_lr = float(get("lr_continuation_source_base_lr"))
    source_warmup = int(get("lr_continuation_source_warmup_steps"))
    source_total = int(get("lr_continuation_source_training_steps"))
    configured_lr = float(get("lr_continuation_learning_rate"))
    relative_tolerance = float(get("lr_continuation_relative_tolerance", 0.05))
    if start_step <= 0 or end_step <= start_step or end_step != total_steps:
        raise ValueError("续训学习率必须满足0 < start_step < end_step == num_training_steps")
    if not start_step <= expected_resume_step < end_step:
        raise ValueError(
            "lr_continuation_expected_resume_step必须位于续训区间内"
        )
    if not 0 < relative_tolerance <= 0.1:
        raise ValueError("lr_continuation_relative_tolerance必须位于(0, 0.1]")
    expected_lr = cosine_learning_rate_at_step(
        base_lr=source_base_lr,
        num_warmup_steps=source_warmup,
        num_training_steps=source_total,
        step=start_step,
    )
    if not math.isclose(configured_lr, expected_lr, rel_tol=1e-6, abs_tol=1e-12):
        raise ValueError(
            "续训学习率与源cosine在恢复点不连续："
            f"configured={configured_lr:.12g}, expected={expected_lr:.12g}"
        )
    if not math.isclose(base_lr, source_base_lr, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("续训optimizer base lr必须保持源实验值")
    return {
        "mode": mode,
        "base_learning_rate": base_lr,
        "num_warmup_steps": 0,
        "num_training_steps": total_steps,
        "resume_start_step": start_step,
        "expected_resume_step": expected_resume_step,
        "continuation_end_step": end_step,
        "continuation_learning_rate": configured_lr,
        "continuation_scale": configured_lr / base_lr,
        "relative_tolerance": relative_tolerance,
        "source_schedule": {
            "mode": "cosine",
            "base_learning_rate": source_base_lr,
            "num_warmup_steps": source_warmup,
            "num_training_steps": source_total,
        },
    }


def resolve_training_schedule(
    *, num_training_steps: int, trainer_max_steps: int | None = None
) -> dict[str, int]:
    scheduler_steps = int(num_training_steps)
    invocation_steps = (
        scheduler_steps if trainer_max_steps is None else int(trainer_max_steps)
    )
    if scheduler_steps <= 0:
        raise ValueError("num_training_steps 必须为正整数")
    if invocation_steps <= 0:
        raise ValueError("trainer_max_steps 必须为正整数")
    if invocation_steps > scheduler_steps:
        raise ValueError(
            "trainer_max_steps 不能超过学习率计划总步数："
            f"{invocation_steps} > {scheduler_steps}"
        )
    return {
        "num_training_steps": scheduler_steps,
        "trainer_max_steps": invocation_steps,
    }


def resolve_subset_indices(
    *, dataset_length: int, subset_size: int | None, subset_start: int = 0
) -> tuple[int, ...] | None:
    length = int(dataset_length)
    start = int(subset_start)
    if length <= 0:
        raise ValueError("训练数据集不能为空")
    if subset_size is None:
        return None
    size = int(subset_size)
    if size <= 0:
        raise ValueError("train_subset_size 必须为正整数或 null")
    if start < 0 or start + size > length:
        raise ValueError(
            "训练子集超出数据范围："
            f"start={start}, size={size}, dataset_length={length}"
        )
    return tuple(range(start, start + size))


def resolve_resume_checkpoint(value: str | os.PathLike[str] | None) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"resume checkpoint 不存在：{path}")
    return str(path)


def validate_excluded_frozen_resume_keys(
    *,
    missing_keys: list[str] | tuple[str, ...] | set[str],
    unexpected_keys: list[str] | tuple[str, ...] | set[str],
    frozen_parameter_names: list[str] | tuple[str, ...] | set[str],
) -> dict[str, Any]:
    """Allow only frozen parameters intentionally omitted by DeepSpeed saves."""
    missing = set(missing_keys)
    unexpected = set(unexpected_keys)
    frozen = set(frozen_parameter_names)
    disallowed_missing = missing - frozen
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "resume checkpoint 与当前模型不兼容："
            f"非冻结缺失参数={sorted(disallowed_missing)[:10]} "
            f"(共 {len(disallowed_missing)} 项)，"
            f"额外参数={sorted(unexpected)[:10]} (共 {len(unexpected)} 项)"
        )
    allowed_missing = sorted(missing)
    return {
        "mode": "excluded_frozen_parameters",
        "missing_frozen_count": len(allowed_missing),
        "missing_frozen_parameters": allowed_missing,
        "unexpected_count": 0,
    }


def resolve_save_last(value: bool | str | None) -> bool | str:
    if value == "link":
        return "link"
    if value is None or isinstance(value, bool):
        return bool(value)
    raise ValueError("save_last 只允许 true、false 或 link")


def resolve_checkpoint_monitor(save_top_k: int) -> dict[str, str]:
    """Rank step-based checkpoints when Lightning must retain more than one."""
    top_k = int(save_top_k)
    if top_k < -1:
        raise ValueError("save_top_k 必须大于等于 -1")
    if top_k > 1:
        return {"monitor": "step", "mode": "max"}
    return {}


def collect_git_state(repo_root: str | os.PathLike[str]) -> dict[str, Any]:
    root = Path(repo_root).resolve()

    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        status = run("status", "--porcelain")
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(status),
            "status": status.splitlines(),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def write_json_atomic(path: str | os.PathLike[str], payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return output


def _read_optional_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _parse_proc_status_bytes(status_text: str | None) -> dict[str, int]:
    if not status_text:
        return {}
    result: dict[str, int] = {}
    for line in status_text.splitlines():
        key, separator, value = line.partition(":")
        if separator and key in {"VmRSS", "VmHWM"}:
            fields = value.split()
            if fields:
                result[key] = int(fields[0]) * 1024
    return result


def _parse_memory_events(events_text: str | None) -> dict[str, int]:
    if not events_text:
        return {}
    result: dict[str, int] = {}
    for line in events_text.splitlines():
        fields = line.split()
        if len(fields) == 2:
            try:
                result[fields[0]] = int(fields[1])
            except ValueError:
                continue
    return result


def collect_memory_snapshot(
    *,
    cgroup_root: str | os.PathLike[str] = "/sys/fs/cgroup",
    proc_status_path: str | os.PathLike[str] = "/proc/self/status",
) -> dict[str, Any]:
    """Collect process and cgroup memory without importing Torch or psutil."""
    root = Path(cgroup_root)
    process_memory = _parse_proc_status_bytes(
        _read_optional_text(Path(proc_status_path))
    )
    v2_values = {
        name: _read_optional_text(root / name)
        for name in ("memory.current", "memory.peak", "memory.max")
    }
    v2_events = _parse_memory_events(_read_optional_text(root / "memory.events"))
    if any(value is not None for value in v2_values.values()) or v2_events:
        cgroup = {
            "version": 2,
            **{key: value for key, value in v2_values.items() if value is not None},
            "memory.events": v2_events,
        }
    else:
        memory_root = root / "memory"
        v1_names = (
            "memory.usage_in_bytes",
            "memory.max_usage_in_bytes",
            "memory.limit_in_bytes",
            "memory.failcnt",
        )
        v1_values = {
            name: _read_optional_text(memory_root / name) for name in v1_names
        }
        cgroup = {
            "version": 1,
            **{key: value for key, value in v1_values.items() if value is not None},
        }
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "process": process_memory,
        "cgroup": cgroup,
    }


def append_jsonl_fsync(
    path: str | os.PathLike[str], payload: dict[str, Any]
) -> Path:
    """Append one durable diagnostic event before a process can be OOM-killed."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return output
