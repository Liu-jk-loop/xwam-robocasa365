"""Dependency-free index and path resolution for RoboCasa365 LeRobot v2.1."""

from __future__ import annotations

import bisect
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data.robocasa365_contract import DatasetContractError, resolve_lerobot_root


DEFAULT_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
DEFAULT_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


@dataclass(frozen=True)
class EpisodeRecord:
    episode_index: int
    length: int
    prompts: tuple[str, ...]


@dataclass(frozen=True)
class ClipSpec:
    episode_index: int
    start_frame: int
    frame_ids: tuple[int, ...]
    action_ids: tuple[int, ...]


class ClipIndex(Sequence[tuple[int, int]]):
    """Compact mapping from a flat sample index to episode and start frame."""

    def __init__(self, episodes: Sequence[EpisodeRecord], required_span: int):
        self.episode_indices: list[int] = []
        cumulative_counts: list[int] = []
        total_clips = 0

        for episode in episodes:
            clip_count = episode.length - required_span
            if clip_count <= 0:
                continue
            total_clips += clip_count
            self.episode_indices.append(episode.episode_index)
            cumulative_counts.append(total_clips)

        self.total_clips = total_clips
        self.cumulative_counts = cumulative_counts

    def __len__(self) -> int:
        return self.total_clips

    def __getitem__(self, index: int | slice) -> tuple[int, int] | list[tuple[int, int]]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self.total_clips)
            return [self[position] for position in range(start, stop, step)]

        position = int(index)
        if position < 0:
            position += self.total_clips
        if position < 0 or position >= self.total_clips:
            raise IndexError(f"clip index {position} 超出范围 [0, {self.total_clips})")

        episode_position = bisect.bisect_right(self.cumulative_counts, position)
        previous = 0 if episode_position == 0 else self.cumulative_counts[episode_position - 1]
        return self.episode_indices[episode_position], position - previous


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetContractError(f"无法读取 JSON：{path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DatasetContractError(f"JSON 顶层必须是对象：{path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise DatasetContractError(f"JSONL 第 {line_number} 行不是对象：{path}")
                records.append(record)
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetContractError(f"无法读取 JSONL：{path}: {exc}") from exc
    return records


def _task_prompts(meta_root: Path) -> dict[int, str]:
    tasks_path = meta_root / "tasks.jsonl"
    if not tasks_path.is_file():
        raise DatasetContractError(f"缺少任务描述文件：{tasks_path}")

    prompts: dict[int, str] = {}
    for record in _read_jsonl(tasks_path):
        try:
            task_index = int(record["task_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetContractError(f"任务记录缺少有效 task_index：{record}") from exc
        prompt = record.get("task")
        if not isinstance(prompt, str) or not prompt.strip():
            raise DatasetContractError(f"任务 {task_index} 缺少非空 task 文本")
        if task_index in prompts:
            raise DatasetContractError(f"任务索引重复：{task_index}")
        prompts[task_index] = prompt.strip()
    if not prompts:
        raise DatasetContractError(f"任务描述为空：{tasks_path}")
    return prompts


def _episode_prompts(raw: Any, task_prompts: dict[int, str]) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return tuple(task_prompts.values()) if len(task_prompts) == 1 else ()

    prompts: list[str] = []
    for value in raw:
        if isinstance(value, str) and value.strip():
            prompts.append(value.strip())
        elif isinstance(value, int) and value in task_prompts:
            prompts.append(task_prompts[value])
    return tuple(dict.fromkeys(prompts))


def load_episode_records(dataset_path: str | Path) -> tuple[Path, dict[str, Any], tuple[EpisodeRecord, ...]]:
    """Load the v2.1 episode index without importing Torch, Decord, or PyArrow."""

    root = resolve_lerobot_root(dataset_path)
    meta_root = root / "meta"
    info = _read_json(meta_root / "info.json")
    version = str(info.get("codebase_version", ""))
    if not version.startswith("v2."):
        raise DatasetContractError(f"原生 loader 当前只支持 LeRobot v2.x，实际为 {version!r}")

    episodes_path = meta_root / "episodes.jsonl"
    if not episodes_path.is_file():
        raise DatasetContractError(
            f"原生 loader 当前需要 LeRobot v2 episodes.jsonl，未找到：{episodes_path}"
        )

    task_prompts = _task_prompts(meta_root)
    episodes: list[EpisodeRecord] = []
    seen_indices: set[int] = set()
    for raw in _read_jsonl(episodes_path):
        try:
            episode_index = int(raw["episode_index"])
            length = int(raw["length"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetContractError(f"episode 记录缺少有效 episode_index/length：{raw}") from exc
        if episode_index < 0 or length <= 0:
            raise DatasetContractError(f"episode 索引或长度非法：index={episode_index}, length={length}")
        if episode_index in seen_indices:
            raise DatasetContractError(f"episode_index 重复：{episode_index}")
        prompts = _episode_prompts(raw.get("tasks"), task_prompts)
        if not prompts:
            raise DatasetContractError(f"episode {episode_index} 无法解析语言任务描述")
        seen_indices.add(episode_index)
        episodes.append(EpisodeRecord(episode_index=episode_index, length=length, prompts=prompts))

    episodes.sort(key=lambda episode: episode.episode_index)
    expected_total = info.get("total_episodes")
    if expected_total is not None and int(expected_total) != len(episodes):
        raise DatasetContractError(
            f"info.total_episodes={expected_total} 与 episodes.jsonl 记录数 {len(episodes)} 不一致"
        )
    return root, info, tuple(episodes)


def _format_path(
    root: Path,
    template: str,
    *,
    episode_index: int,
    chunks_size: int,
    video_key: str | None = None,
) -> Path:
    if chunks_size <= 0:
        raise DatasetContractError(f"info.chunks_size 必须为正数，实际为 {chunks_size}")
    values = {
        "episode_chunk": episode_index // chunks_size,
        "episode_index": episode_index,
        "video_key": video_key,
    }
    try:
        relative = Path(template.format(**values))
    except (KeyError, ValueError) as exc:
        raise DatasetContractError(f"无法解析 LeRobot 路径模板 {template!r}: {exc}") from exc
    if relative.is_absolute() or ".." in relative.parts:
        raise DatasetContractError(f"LeRobot 路径模板必须解析为数据根目录内的相对路径：{relative}")
    return root / relative


def episode_data_path(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    template = info.get("data_path", DEFAULT_DATA_PATH)
    if not isinstance(template, str) or not template:
        raise DatasetContractError("info.data_path 必须是非空字符串")
    chunks_size = int(info.get("chunks_size", 1000))
    return _format_path(root, template, episode_index=episode_index, chunks_size=chunks_size)


def episode_video_path(root: Path, info: dict[str, Any], episode_index: int, video_key: str) -> Path:
    template = info.get("video_path", DEFAULT_VIDEO_PATH)
    if not isinstance(template, str) or not template:
        raise DatasetContractError("info.video_path 必须是非空字符串")
    chunks_size = int(info.get("chunks_size", 1000))
    return _format_path(
        root,
        template,
        episode_index=episode_index,
        chunks_size=chunks_size,
        video_key=video_key,
    )


def build_clip_spec(
    episode_index: int,
    start_frame: int,
    *,
    sequence_length: int,
    frame_skip: int,
    action_skip: int,
) -> ClipSpec:
    if sequence_length <= 1:
        raise ValueError(f"sequence_length 必须大于 1，实际为 {sequence_length}")
    if frame_skip <= 0 or action_skip <= 0:
        raise ValueError(f"frame_skip/action_skip 必须为正数，实际为 {frame_skip}/{action_skip}")
    if frame_skip % action_skip != 0:
        raise ValueError(
            f"frame_skip 必须能被 action_skip 整除，实际为 {frame_skip}/{action_skip}"
        )

    action_span = (sequence_length - 1) * frame_skip
    frame_ids = tuple(start_frame + offset * frame_skip for offset in range(sequence_length))
    action_ids = tuple(range(start_frame + action_skip - 1, start_frame + action_span, action_skip))
    return ClipSpec(
        episode_index=episode_index,
        start_frame=start_frame,
        frame_ids=frame_ids,
        action_ids=action_ids,
    )
