"""Dependency-free M3 training-run validation and provenance helpers."""

from __future__ import annotations

import json
import os
import subprocess
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
