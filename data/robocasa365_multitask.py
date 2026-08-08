"""Atomic-only multi-task manifest and balanced Dataset adapter."""

from __future__ import annotations

import bisect
import hashlib
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
from data.robocasa365_index import load_episode_records


SUPPORTED_SAMPLING = {"balanced_round_robin", "natural_proportional"}


@dataclass(frozen=True)
class TaskDatasetEntry:
    task_name: str
    dataset_path: str
    valid_clips: int | None = None


@dataclass(frozen=True)
class MultiTaskDatasetManifest:
    name: str
    scope: str
    sampling: str
    tasks: tuple[TaskDatasetEntry, ...]
    source_path: str
    manifest_digest: str | None = None
    total_valid_clips: int | None = None


def training_manifest_digest(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("manifest_digest", None)
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: str | Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetContractError(
            f"无法读取多任务数据清单：{manifest_path}: {exc}"
        ) from exc
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
        raise DatasetContractError(
            f"多任务数据清单 tasks 必须是非空数组：{manifest_path}"
        )
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
                valid_clips=(
                    int(raw_entry["valid_clips"])
                    if raw_entry.get("valid_clips") is not None
                    else None
                ),
            )
        )
    names = [entry.task_name for entry in entries]
    paths = [entry.dataset_path for entry in entries]
    if len(names) != len(set(names)):
        raise DatasetContractError(f"多任务数据清单包含重复任务：{names}")
    if len(paths) != len(set(paths)):
        raise DatasetContractError("多任务数据清单不允许多个任务复用同一数据目录")
    manifest_digest = payload.get("manifest_digest")
    if manifest_digest is not None:
        actual_digest = training_manifest_digest(payload)
        if str(manifest_digest) != actual_digest:
            raise DatasetContractError(
                "多任务数据清单摘要不匹配："
                f"expected={manifest_digest}, actual={actual_digest}"
            )
    total_valid_clips = payload.get("total_valid_clips")
    if total_valid_clips is not None:
        declared_total = int(total_valid_clips)
        entry_counts = [entry.valid_clips for entry in entries]
        if any(count is None for count in entry_counts):
            raise DatasetContractError(
                "声明 total_valid_clips 时每个任务都必须包含 valid_clips"
            )
        if declared_total != sum(
            int(count) for count in entry_counts if count is not None
        ):
            raise DatasetContractError(
                "total_valid_clips 与任务 valid_clips 求和不一致"
            )
    return MultiTaskDatasetManifest(
        name=str(payload.get("name", manifest_path.stem)),
        scope="atomic_only",
        sampling=sampling,
        tasks=tuple(entries),
        source_path=str(manifest_path),
        manifest_digest=str(manifest_digest) if manifest_digest is not None else None,
        total_valid_clips=(
            int(total_valid_clips) if total_valid_clips is not None else None
        ),
    )


def build_atomic_training_manifest_report(
    *,
    dataset_root: str | Path,
    atomic_task_manifest: str | Path,
    explicit_paths: dict[str, str] | None = None,
    sequence_length: int = 9,
    frame_skip: int = 4,
) -> dict[str, Any]:
    """Audit every task in a versioned atomic manifest for formal training."""
    allowed_manifest = load_task_manifest(atomic_task_manifest)
    task_names = list(allowed_manifest.tasks)
    configured_paths = explicit_paths or {}
    unknown = sorted(set(configured_paths) - set(task_names))
    errors = [f"--task-path 指定了清单外任务：{unknown}"] if unknown else []
    required_span = (int(sequence_length) - 1) * int(frame_skip)
    if required_span <= 0:
        raise ValueError("sequence_length/frame_skip 必须产生正 required_span")

    tasks: list[dict[str, Any]] = []
    for task_name in task_names:
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
            _, _, episodes = load_episode_records(task_path)
            valid_clips = sum(
                max(episode.length - required_span, 0) for episode in episodes
            )
            if valid_clips <= 0:
                raise DatasetContractError(
                    f"没有满足 required_span={required_span} 的有效 clip"
                )
            tasks.append(
                {
                    "task_name": task_name,
                    "dataset_path": str(task_path),
                    "episodes": summary.total_episodes,
                    "frames": summary.total_frames,
                    "valid_clips": valid_clips,
                    "state_dim": summary.state_dim,
                    "action_dim": summary.action_dim,
                    "camera_keys": summary.camera_keys,
                }
            )
        except (DatasetContractError, OSError, ValueError) as exc:
            errors.append(f"{task_name}: {exc}")

    ok = not errors and len(tasks) == len(task_names)
    report: dict[str, Any] = {
        "schema_version": 1,
        "name": "robocasa365_m6_atomic_seen18_pretrain",
        "scope": "atomic_only",
        "split": "pretrain",
        "sampling": "natural_proportional",
        "atomic_task_manifest": str(Path(atomic_task_manifest).expanduser()),
        "dataset_root": str(Path(dataset_root).expanduser()),
        "sequence_length": int(sequence_length),
        "frame_skip": int(frame_skip),
        "required_span": required_span,
        "expected_task_count": len(task_names),
        "total_valid_clips": sum(int(task["valid_clips"]) for task in tasks),
        "total_frames": sum(int(task["frames"]) for task in tasks),
        "tasks": tasks,
        "errors": errors,
        "ok": ok,
        "result": "pass" if ok else "fail",
    }
    report["manifest_digest"] = training_manifest_digest(report)
    return report


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
            errors.append(
                f"任务 {name!r} 不属于 atomic-only 清单 {allowed_manifest.name}"
            )

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
            raise IndexError(
                f"多任务 sample index {position} 超出范围 [0, {self._length})"
            )
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
            output["episode_key"] = (
                f"{self.task_names[task_index]}:{output['episode_key']}"
            )
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


class NaturalConcatDataset:
    """Concatenate task datasets without oversampling and preserve task identity."""

    def __init__(
        self,
        datasets: Sequence[Any],
        *,
        task_names: Sequence[str],
        manifest_path: str,
        manifest_digest: str | None = None,
    ):
        self.datasets = tuple(datasets)
        self.task_names = tuple(str(name) for name in task_names)
        if not self.datasets or len(self.datasets) != len(self.task_names):
            raise ValueError("datasets 与 task_names 必须是等长非空序列")
        self.task_lengths = tuple(len(dataset) for dataset in self.datasets)
        if any(length <= 0 for length in self.task_lengths):
            raise ValueError(f"多任务子数据集不能为空：{self.task_lengths}")
        contracts = {
            (int(dataset.action_num), int(dataset.action_dim), int(dataset.proprio_dim))
            for dataset in self.datasets
        }
        if len(contracts) != 1:
            raise ValueError(f"多任务 Dataset 张量合同不一致：{sorted(contracts)}")
        self.action_num, self.action_dim, self.proprio_dim = next(iter(contracts))
        self.manifest_path = str(Path(manifest_path).expanduser().resolve())
        self.manifest_digest = manifest_digest
        total = 0
        self.cumulative_lengths: list[int] = []
        for length in self.task_lengths:
            total += length
            self.cumulative_lengths.append(total)
        self._length = total

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Any:
        position = int(index)
        if position < 0:
            position += self._length
        if position < 0 or position >= self._length:
            raise IndexError(
                f"多任务 sample index {position} 超出范围 [0, {self._length})"
            )
        task_index = bisect.bisect_right(self.cumulative_lengths, position)
        previous = 0 if task_index == 0 else self.cumulative_lengths[task_index - 1]
        local_index = position - previous
        sample = self.datasets[task_index][local_index]
        if not isinstance(sample, dict):
            return sample
        output = dict(sample)
        output["task_index"] = task_index
        output["task_name"] = self.task_names[task_index]
        if "episode_key" in output:
            output["episode_key"] = (
                f"{self.task_names[task_index]}:{output['episode_key']}"
            )
        return output

    def provenance(self) -> dict[str, Any]:
        return {
            "type": "natural_proportional",
            "manifest_path": self.manifest_path,
            "manifest_digest": self.manifest_digest,
            "task_names": list(self.task_names),
            "task_lengths": list(self.task_lengths),
            "raw_samples": self._length,
            "effective_samples": self._length,
        }
