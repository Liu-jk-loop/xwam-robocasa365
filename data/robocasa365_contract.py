"""Dependency-free validation for the RoboCasa365 LeRobot dataset contract."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


EXPECTED_STATE_DIM = 16
EXPECTED_ACTION_DIM = 12
EXPECTED_CAMERA_KEYS = (
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
)
REQUIRED_META_FILES = (
    "info.json",
    "tasks.jsonl",
    "modality.json",
    "embodiment.json",
)


class DatasetContractError(ValueError):
    """Raised when metadata cannot be parsed as a supported dataset contract."""


@dataclass(frozen=True)
class TaskManifest:
    name: str
    scope: str
    tasks: tuple[str, ...]
    split: str | None = None
    source: str | None = None
    robocasa_release: str | None = None
    source_file: str | None = None


@dataclass
class DatasetSummary:
    requested_root: str
    lerobot_root: str | None
    task_name: str | None
    manifest_name: str | None
    codebase_version: str | None
    fps: float | None
    total_episodes: int | None
    total_frames: int | None
    task_descriptions: int
    episode_records: int | None
    state_dim: int | None
    action_dim: int | None
    camera_keys: list[str]
    data_files: int | None
    video_files_by_camera: dict[str, int] | None
    warnings: list[str]
    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ok"] = self.ok
        return payload


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
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


def load_task_manifest(path: str | Path) -> TaskManifest:
    manifest_path = Path(path).expanduser().resolve()
    payload = _read_json(manifest_path)
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks or not all(isinstance(task, str) and task for task in tasks):
        raise DatasetContractError(f"任务清单必须包含非空字符串数组 tasks：{manifest_path}")
    if len(tasks) != len(set(tasks)):
        raise DatasetContractError(f"任务清单包含重复任务：{manifest_path}")
    scope = payload.get("scope")
    if scope != "atomic_only":
        raise DatasetContractError(f"当前项目只接受 scope=atomic_only，实际为 {scope!r}：{manifest_path}")
    return TaskManifest(
        name=str(payload.get("name", manifest_path.stem)),
        scope=scope,
        tasks=tuple(tasks),
        split=_optional_string(payload.get("split")),
        source=_optional_string(payload.get("source")),
        robocasa_release=_optional_string(payload.get("robocasa_release")),
        source_file=_optional_string(payload.get("source_file")),
    )


def resolve_lerobot_root(path: str | Path) -> Path:
    requested = Path(path).expanduser().resolve()
    for candidate in (requested, requested / "lerobot"):
        if (candidate / "meta" / "info.json").is_file():
            return candidate
    raise DatasetContractError(
        f"未找到 LeRobot 元数据。期望存在 {requested / 'meta' / 'info.json'} "
        f"或 {requested / 'lerobot' / 'meta' / 'info.json'}"
    )


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _feature_dim(features: dict[str, Any], key: str) -> int | None:
    feature = features.get(key)
    if not isinstance(feature, dict):
        return None
    shape = feature.get("shape")
    if not isinstance(shape, list) or len(shape) != 1:
        return None
    return _optional_int(shape[0])


def _find_known_tasks(value: Any, allowed: set[str]) -> set[str]:
    matches: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            matches.update(_find_known_tasks(key, allowed))
            matches.update(_find_known_tasks(child, allowed))
    elif isinstance(value, list):
        for child in value:
            matches.update(_find_known_tasks(child, allowed))
    elif isinstance(value, str):
        if value in allowed:
            matches.add(value)
        else:
            matches.update(task for task in allowed if value.endswith(f"/{task}") or value.endswith(f":{task}"))
    return matches


def infer_task_name(root: Path, manifest: TaskManifest, explicit_task_name: str | None = None) -> str | None:
    allowed = set(manifest.tasks)
    if explicit_task_name is not None:
        return explicit_task_name

    path_matches = {part for part in root.parts if part in allowed}
    metadata_path = root / "extras" / "dataset_meta.json"
    metadata_matches: set[str] = set()
    if metadata_path.is_file():
        metadata_matches = _find_known_tasks(_read_json(metadata_path), allowed)

    matches = path_matches | metadata_matches
    if len(matches) == 1:
        return next(iter(matches))
    if len(matches) > 1:
        raise DatasetContractError(f"数据目录中识别到多个任务 {sorted(matches)}，请显式传入 --task-name")
    return None


def _count_files(root: Path, pattern: str) -> int:
    return sum(1 for path in root.glob(pattern) if path.is_file())


def inspect_dataset(
    dataset_path: str | Path,
    *,
    manifest: TaskManifest | None = None,
    task_name: str | None = None,
    require_data: bool = False,
    require_videos: bool = False,
) -> DatasetSummary:
    requested_root = str(Path(dataset_path).expanduser())
    warnings: list[str] = []
    errors: list[str] = []

    try:
        root = resolve_lerobot_root(dataset_path)
    except DatasetContractError as exc:
        return DatasetSummary(
            requested_root=requested_root,
            lerobot_root=None,
            task_name=task_name,
            manifest_name=manifest.name if manifest else None,
            codebase_version=None,
            fps=None,
            total_episodes=None,
            total_frames=None,
            task_descriptions=0,
            episode_records=None,
            state_dim=None,
            action_dim=None,
            camera_keys=[],
            data_files=None,
            video_files_by_camera=None,
            warnings=warnings,
            errors=[str(exc)],
        )

    meta_root = root / "meta"
    for filename in REQUIRED_META_FILES:
        if not (meta_root / filename).is_file():
            errors.append(f"缺少官方元数据文件：meta/{filename}")

    info = _read_json(meta_root / "info.json")
    features = info.get("features")
    if not isinstance(features, dict):
        features = {}
        errors.append("meta/info.json 缺少 features 对象")

    state_dim = _feature_dim(features, "observation.state")
    action_dim = _feature_dim(features, "action")
    if state_dim != EXPECTED_STATE_DIM:
        errors.append(f"observation.state 维度应为 {EXPECTED_STATE_DIM}，实际为 {state_dim}")
    if action_dim != EXPECTED_ACTION_DIM:
        errors.append(f"action 维度应为 {EXPECTED_ACTION_DIM}，实际为 {action_dim}")

    camera_keys = sorted(key for key in features if key.startswith("observation.images."))
    missing_cameras = sorted(set(EXPECTED_CAMERA_KEYS) - set(camera_keys))
    if missing_cameras:
        errors.append(f"缺少相机特征：{missing_cameras}")

    tasks_path = meta_root / "tasks.jsonl"
    task_records = _read_jsonl(tasks_path) if tasks_path.is_file() else []

    episodes_path = meta_root / "episodes.jsonl"
    episodes_dir = meta_root / "episodes"
    episode_records: int | None
    if episodes_path.is_file():
        episode_records = len(_read_jsonl(episodes_path))
    elif episodes_dir.is_dir():
        episode_records = None
        warnings.append("检测到 LeRobot v3 分块 episodes 元数据；本轮只核对 info/tasks，逐 episode 审计将在后续适配中启用")
    else:
        episode_records = None
        errors.append("缺少 meta/episodes.jsonl 或 meta/episodes/ 分块元数据")

    resolved_task_name = task_name
    if manifest is not None:
        try:
            resolved_task_name = infer_task_name(root, manifest, task_name)
        except DatasetContractError as exc:
            errors.append(str(exc))
        if resolved_task_name is None:
            errors.append("无法确认数据集所属任务；atomic-only 模式必须通过目录、dataset_meta.json 或 --task-name 明确任务")
        elif resolved_task_name not in manifest.tasks:
            errors.append(f"任务 {resolved_task_name!r} 不属于 atomic-only 清单 {manifest.name!r}")

    data_files: int | None = None
    if require_data:
        data_files = _count_files(root / "data", "**/*.parquet")
        if data_files == 0:
            errors.append("data/ 下未找到 Parquet 文件")

    video_files_by_camera: dict[str, int] | None = None
    if require_videos:
        video_files_by_camera = {}
        for camera_key in EXPECTED_CAMERA_KEYS:
            count = _count_files(root / "videos", f"**/{camera_key}/*.mp4")
            video_files_by_camera[camera_key] = count
            if count == 0:
                errors.append(f"videos/ 下未找到相机 {camera_key} 的 MP4 文件")

    return DatasetSummary(
        requested_root=requested_root,
        lerobot_root=str(root),
        task_name=resolved_task_name,
        manifest_name=manifest.name if manifest else None,
        codebase_version=_optional_string(info.get("codebase_version")),
        fps=_optional_float(info.get("fps")),
        total_episodes=_optional_int(info.get("total_episodes")),
        total_frames=_optional_int(info.get("total_frames")),
        task_descriptions=len(task_records),
        episode_records=episode_records,
        state_dim=state_dim,
        action_dim=action_dim,
        camera_keys=camera_keys,
        data_files=data_files,
        video_files_by_camera=video_files_by_camera,
        warnings=warnings,
        errors=errors,
    )


def inspect_many(
    dataset_paths: Iterable[str | Path],
    *,
    manifest: TaskManifest | None = None,
    require_data: bool = False,
    require_videos: bool = False,
) -> list[DatasetSummary]:
    return [
        inspect_dataset(
            path,
            manifest=manifest,
            require_data=require_data,
            require_videos=require_videos,
        )
        for path in dataset_paths
    ]
