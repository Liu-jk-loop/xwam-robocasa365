"""Dependency-free M3 training-run validation and provenance helpers."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def resolve_save_last(value: bool | str | None) -> bool | str:
    if value == "link":
        return "link"
    if value is None or isinstance(value, bool):
        return bool(value)
    raise ValueError("save_last 只允许 true、false 或 link")


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
