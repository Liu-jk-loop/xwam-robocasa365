"""Atomic-only multi-task manifest and balanced Dataset adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from data.robocasa365_contract import (
    DatasetContractError,
    inspect_dataset,
    load_task_manifest,
    resolve_lerobot_root,
)


SUPPORTED_SAMPLING = {"balanced_round_robin"}


@dataclass(frozen=True)
class TaskDatasetEntry:
    task_name: str
    dataset_path: str


@dataclass(frozen=True)
class MultiTaskDatasetManifest:
    name: str
    scope: str
    sampling: str
    tasks: tuple[TaskDatasetEntry, ...]
    source_path: str


def _read_json(path: str | Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetContractError(f"无法读取多任务数据清单：{manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DatasetContractError(f"多任务数据清单顶层必须是对象：{manifest_path}")
    return manifest_path, payload


def load_multitask_dataset_manifest(
    path: str | Path,
    *,
    atomic_task_manifest: str | Path,
    expected_task_count: int | None = None,
) -> MultiTaskDatasetManifest:
    manifest_path, payload = _read_json(path)
    if payload.get("schema_version") != 1:
        raise DatasetContractError(
            f"多任务数据清单 schema_version 必须为 1：{manifest_path}"
        )
    if payload.get("scope") != "atomic_only":
        raise DatasetContractError(
            f"多任务数据清单只允许 scope=atomic_only：{manifest_path}"
        )
    if payload.get("ok") is not True:
        raise DatasetContractError(f"多任务数据清单尚未通过审计：{manifest_path}")
    sampling = str(payload.get("sampling", ""))
    if sampling not in SUPPORTED_SAMPLING:
        raise DatasetContractError(
            f"多任务 sampling 只允许 {sorted(SUPPORTED_SAMPLING)}，实际为 {sampling!r}"
        )
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise DatasetContractError(f"多任务数据清单 tasks 必须是非空数组：{manifest_path}")
    if expected_task_count is not None and len(raw_tasks) != int(expected_task_count):
        raise DatasetContractError(
            f"多任务数据清单必须包含 {expected_task_count} 个任务，实际为 {len(raw_tasks)}"
        )

    allowed = set(load_task_manifest(atomic_task_manifest).tasks)
    entries: list[TaskDatasetEntry] = []
    for index, raw_entry in enumerate(raw_tasks):
        if not isinstance(raw_entry, dict):
            raise DatasetContractError(f"tasks[{index}] 必须是对象")
        task_name = raw_entry.get("task_name")
        dataset_path = raw_entry.get("dataset_path")
        if not isinstance(task_name, str) or not task_name:
            raise DatasetContractError(f"tasks[{index}].task_name 必须是非空字符串")
        if task_name not in allowed:
            raise DatasetContractError(f"任务 {task_name!r} 不属于 atomic-only 清单")
        if not isinstance(dataset_path, str) or not dataset_path:
            raise DatasetContractError(f"tasks[{index}].dataset_path 必须是非空字符串")
        resolve_lerobot_root(dataset_path)
        entries.append(
            TaskDatasetEntry(
                task_name=task_name,
                dataset_path=str(Path(dataset_path).expanduser().resolve()),
            )
        )
    names = [entry.task_name for entry in entries]
    paths = [entry.dataset_path for entry in entries]
    if len(names) != len(set(names)):
        raise DatasetContractError(f"多任务数据清单包含重复任务：{names}")
    if len(paths) != len(set(paths)):
        raise DatasetContractError("多任务数据清单不允许多个任务复用同一数据目录")
    return MultiTaskDatasetManifest(
        name=str(payload.get("name", manifest_path.stem)),
        scope="atomic_only",
        sampling=sampling,
        tasks=tuple(entries),
        source_path=str(manifest_path),
    )


def resolve_task_dataset_directory(dataset_root: str | Path, task_name: str) -> Path:
    task_root = Path(dataset_root).expanduser().resolve() / task_name
    try:
        resolve_lerobot_root(task_root)
        return task_root
    except DatasetContractError:
        pass
    if not task_root.is_dir():
        raise DatasetContractError(f"任务目录不存在：{task_root}")
    candidates: list[Path] = []
    for child in sorted(task_root.iterdir()):
        if not child.is_dir():
            continue
        try:
            resolve_lerobot_root(child)
        except DatasetContractError:
            continue
        candidates.append(child)
    if not candidates:
        raise DatasetContractError(f"任务 {task_name} 下没有可用日期目录：{task_root}")
    if len(candidates) > 1:
        raise DatasetContractError(
            f"任务 {task_name} 存在多个可用日期目录，请用 --task-path 显式选择："
            f"{[str(path) for path in candidates]}"
        )
    return candidates[0]


def build_m3_multitask_manifest_report(
    *,
    dataset_root: str | Path,
    task_names: Sequence[str],
    atomic_task_manifest: str | Path,
    explicit_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    names = [str(name) for name in task_names]
    configured_paths = explicit_paths or {}
    errors: list[str] = []
    tasks: list[dict[str, Any]] = []
    allowed_manifest = load_task_manifest(atomic_task_manifest)
    allowed = set(allowed_manifest.tasks)
    if len(names) != 3:
        errors.append(f"M3 三任务 smoke 必须选择 3 个任务，实际为 {len(names)}")
    if len(names) != len(set(names)):
        errors.append(f"任务名称重复：{names}")
    unknown_path_tasks = sorted(set(configured_paths) - set(names))
    if unknown_path_tasks:
        errors.append(f"--task-path 指定了未选择的任务：{unknown_path_tasks}")
    for name in names:
        if name not in allowed:
            errors.append(f"任务 {name!r} 不属于 atomic-only 清单 {allowed_manifest.name}")

    for task_name in names:
        try:
            configured = configured_paths.get(task_name)
            task_path = (
                Path(configured).expanduser().resolve()
                if configured is not None
                else resolve_task_dataset_directory(dataset_root, task_name)
            )
            if configured is not None and task_name not in task_path.parts:
                raise DatasetContractError(
                    f"显式目录必须包含任务名路径分量 {task_name!r}：{task_path}"
                )
            summary = inspect_dataset(
                task_path,
                manifest=allowed_manifest,
                task_name=task_name,
                require_data=True,
                require_videos=True,
            )
            if not summary.ok:
                raise DatasetContractError("; ".join(summary.errors))
            tasks.append(
                {
                    "task_name": task_name,
                    "dataset_path": str(task_path),
                    "episodes": summary.total_episodes,
                    "frames": summary.total_frames,
                    "state_dim": summary.state_dim,
                    "action_dim": summary.action_dim,
                    "camera_keys": summary.camera_keys,
                }
            )
        except (DatasetContractError, OSError) as exc:
            errors.append(f"{task_name}: {exc}")
    ok = not errors and len(tasks) == 3
    return {
        "schema_version": 1,
        "name": "robocasa365_m3_three_task_smoke",
        "scope": "atomic_only",
        "sampling": "balanced_round_robin",
        "atomic_task_manifest": str(Path(atomic_task_manifest).expanduser()),
        "dataset_root": str(Path(dataset_root).expanduser()),
        "tasks": tasks,
        "errors": errors,
        "ok": ok,
        "result": "pass" if ok else "fail",
    }


class BalancedRoundRobinDataset:
    """Cycle tasks deterministically while preserving each task-local adapter."""

    def __init__(
        self,
        datasets: Sequence[Any],
        *,
        task_names: Sequence[str],
        manifest_path: str,
    ):
        self.datasets = tuple(datasets)
        self.task_names = tuple(str(name) for name in task_names)
        if not self.datasets or len(self.datasets) != len(self.task_names):
            raise ValueError("datasets 与 task_names 必须是等长非空序列")
        self.task_lengths = tuple(len(dataset) for dataset in self.datasets)
        if any(length <= 0 for length in self.task_lengths):
            raise ValueError(f"多任务子数据集不能为空：{self.task_lengths}")
        contracts = {
            (
                int(dataset.action_num),
                int(dataset.action_dim),
                int(dataset.proprio_dim),
            )
            for dataset in self.datasets
        }
        if len(contracts) != 1:
            raise ValueError(f"多任务 Dataset 张量合同不一致：{sorted(contracts)}")
        self.action_num, self.action_dim, self.proprio_dim = next(iter(contracts))
        self.manifest_path = str(Path(manifest_path).expanduser().resolve())
        self._length = max(self.task_lengths) * len(self.datasets)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Any:
        position = int(index)
        if position < 0:
            position += self._length
        if position < 0 or position >= self._length:
            raise IndexError(f"多任务 sample index {position} 超出范围 [0, {self._length})")
        task_index = position % len(self.datasets)
        sample_round = position // len(self.datasets)
        local_index = sample_round % self.task_lengths[task_index]
        sample = self.datasets[task_index][local_index]
        if not isinstance(sample, dict):
            return sample
        output = dict(sample)
        output["task_index"] = task_index
        output["task_name"] = self.task_names[task_index]
        if "episode_key" in output:
            output["episode_key"] = f"{self.task_names[task_index]}:{output['episode_key']}"
        return output

    def provenance(self) -> dict[str, Any]:
        return {
            "type": "balanced_round_robin",
            "manifest_path": self.manifest_path,
            "task_names": list(self.task_names),
            "task_lengths": list(self.task_lengths),
            "raw_samples": sum(self.task_lengths),
            "balanced_samples": self._length,
        }
