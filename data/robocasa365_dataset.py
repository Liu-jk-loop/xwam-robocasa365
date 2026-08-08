"""Native RoboCasa365 LeRobot v2.1 RGB-only adapter for X-WAM."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from torch.utils.data import Dataset

from data.augmentation import VideoAugmentation
from data.robocasa365_contract import (
    EXPECTED_ACTION_DIM,
    EXPECTED_STATE_DIM,
    inspect_dataset,
    load_task_manifest,
)
from data.robocasa365_index import (
    ClipIndex,
    ClipSpec,
    EpisodeRecord,
    build_clip_spec,
    episode_data_path,
    episode_video_path,
    load_episode_records,
)
from data.robocasa365_schema import PandaOmronTensorCodec


SUPPORTED_NORMALIZATION = {"none", "panda_omron_v1"}


class RoboCasa365Dataset(Dataset):
    """Read official LeRobot v2.1 Parquet and three synchronized RGB videos."""

    def __init__(
        self,
        dataset_path: str,
        task_name: str,
        task_manifest: str,
        camera_keys: list[str] | tuple[str, ...],
        camera_types: list[str] | tuple[str, ...],
        sequence_length: int,
        frame_skip: int,
        action_skip: int,
        video_size: list[int] | tuple[int, int],
        *,
        augment: bool = False,
        crop_ratio: float = 0.95,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.05,
        use_depth: bool = False,
        normalization: str = "none",
        schema_path: str | None = None,
        statistics_path: str | None = None,
        parquet_cache_size: int = 8,
        video_cache_size: int = 12,
    ):
        if use_depth:
            raise ValueError(
                "RoboCasa365 M1 loader 仅支持 depth=disabled；cached depth 将在 M5 单独接入"
            )
        if normalization not in SUPPORTED_NORMALIZATION:
            raise ValueError(
                f"normalization 只允许 {sorted(SUPPORTED_NORMALIZATION)}，实际为 {normalization!r}"
            )
        if normalization == "panda_omron_v1" and not schema_path:
            raise ValueError(
                "normalization='panda_omron_v1' 时必须提供版本化 schema_path"
            )
        if len(camera_keys) != 3:
            raise ValueError(
                f"X-WAM 当前要求固定三路 RGB 相机，实际为 {len(camera_keys)} 路"
            )
        if len(camera_types) != len(camera_keys):
            raise ValueError("camera_types 数量必须与 camera_keys 一致")
        if len(set(camera_keys)) != len(camera_keys):
            raise ValueError("camera_keys 不允许重复")
        invalid_types = sorted(set(camera_types) - {"static", "dynamic"})
        if invalid_types:
            raise ValueError(
                f"camera_types 只允许 static/dynamic，实际包含 {invalid_types}"
            )

        self.sequence_length = int(sequence_length)
        self.frame_skip = int(frame_skip)
        self.action_skip = int(action_skip)
        self.action_num = self.frame_skip // self.action_skip
        self.action_span = (self.sequence_length - 1) * self.frame_skip
        build_clip_spec(
            0,
            0,
            sequence_length=self.sequence_length,
            frame_skip=self.frame_skip,
            action_skip=self.action_skip,
        )

        self.video_size = self._validate_video_size(video_size)
        self.camera_keys = tuple(str(key) for key in camera_keys)
        self.camera_type_mask = torch.tensor(
            [0 if camera_type == "static" else 1 for camera_type in camera_types],
            dtype=torch.long,
        )
        self.proprio_dim = EXPECTED_STATE_DIM
        self.action_dim = EXPECTED_ACTION_DIM
        self.normalization = normalization
        self.statistics_path = (
            str(Path(statistics_path).expanduser().resolve())
            if statistics_path is not None
            else None
        )
        self.augment = bool(augment)
        self.augmentation = VideoAugmentation(
            crop_ratio=crop_ratio,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )
        self.parquet_cache_size = max(0, int(parquet_cache_size))
        self.video_cache_size = max(0, int(video_cache_size))
        self._parquet_cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = (
            OrderedDict()
        )
        self._video_cache: OrderedDict[tuple[int, str], VideoReader] = OrderedDict()

        manifest = load_task_manifest(task_manifest)
        summary = inspect_dataset(
            dataset_path,
            manifest=manifest,
            task_name=task_name,
            require_data=True,
            require_videos=True,
        )
        if not summary.ok:
            raise ValueError(
                "RoboCasa365 数据契约失败：\n- " + "\n- ".join(summary.errors)
            )
        missing_selected = sorted(set(self.camera_keys) - set(summary.camera_keys))
        if missing_selected:
            raise ValueError(f"配置选择了元数据中不存在的相机：{missing_selected}")

        self.dataset_root, self.info, self.episodes = load_episode_records(dataset_path)
        self.tensor_codec = (
            PandaOmronTensorCodec.from_dataset(
                dataset_path,
                schema_path,
                statistics_path=self.statistics_path,
            )
            if normalization == "panda_omron_v1"
            else None
        )
        if self.tensor_codec is not None:
            if self.tensor_codec.schema.state.dimension != self.proprio_dim:
                raise ValueError(
                    "schema state dimension 与 loader 不一致："
                    f"{self.tensor_codec.schema.state.dimension} != {self.proprio_dim}"
                )
            if self.tensor_codec.schema.action.dimension != self.action_dim:
                raise ValueError(
                    "schema action dimension 与 loader 不一致："
                    f"{self.tensor_codec.schema.action.dimension} != {self.action_dim}"
                )
        self.episode_by_index: dict[int, EpisodeRecord] = {
            episode.episode_index: episode for episode in self.episodes
        }
        self.data_list = ClipIndex(self.episodes, required_span=self.action_span)
        if not self.data_list:
            raise ValueError(
                f"没有可用 clip：sequence_length={self.sequence_length}, frame_skip={self.frame_skip}"
            )
        self.fps = float(self.info["fps"]) / self.frame_skip
        if self.fps <= 0:
            raise ValueError(f"info.fps/frame_skip 必须为正数，实际为 {self.fps}")
        self._validate_episode_files()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_parquet_cache"] = OrderedDict()
        state["_video_cache"] = OrderedDict()
        return state

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int) -> dict[str, Any]:
        clip = self.clip_spec(index)
        episode = self.episode_by_index[clip.episode_index]
        states, actions = self._load_episode_arrays(clip.episode_index, episode.length)
        frame_ids = np.asarray(clip.frame_ids, dtype=np.int64)
        action_ids = np.asarray(clip.action_ids, dtype=np.int64)

        video = torch.stack(
            [
                self._read_video_frames(clip.episode_index, camera_key, frame_ids)
                for camera_key in self.camera_keys
            ],
            dim=0,
        )
        selected_states = states[frame_ids].copy()
        selected_actions = actions[action_ids].copy()
        if self.tensor_codec is not None:
            selected_states = self.tensor_codec.encode_state(selected_states, clip=True)
            selected_actions = self.tensor_codec.encode_action(
                selected_actions, clip=True
            )
        proprios = torch.from_numpy(
            np.ascontiguousarray(selected_states, dtype=np.float32)
        )
        selected_actions = torch.from_numpy(
            np.ascontiguousarray(selected_actions, dtype=np.float32)
        )
        prompt_index = (
            int(torch.randint(len(episode.prompts), ()).item()) if self.augment else 0
        )

        data = {
            "video": video,
            "fps": self.fps,
            "proprios": proprios,
            "proprio_mask": torch.ones_like(proprios),
            "actions": selected_actions,
            "action_mask": torch.ones_like(selected_actions),
            "camera_type_mask": self.camera_type_mask.clone(),
            "prompt": episode.prompts[prompt_index],
            "episode_key": f"episode_{clip.episode_index:06d}",
        }
        return self.augmentation(data) if self.augment else data

    def clip_spec(self, index: int) -> ClipSpec:
        episode_index, start_frame = self.data_list[index]
        return build_clip_spec(
            episode_index,
            start_frame,
            sequence_length=self.sequence_length,
            frame_skip=self.frame_skip,
            action_skip=self.action_skip,
        )

    def sample_paths(self, index: int) -> dict[str, Any]:
        clip = self.clip_spec(index)
        return {
            "parquet": str(
                episode_data_path(self.dataset_root, self.info, clip.episode_index)
            ),
            "videos": {
                camera_key: str(
                    episode_video_path(
                        self.dataset_root, self.info, clip.episode_index, camera_key
                    )
                )
                for camera_key in self.camera_keys
            },
        }

    @staticmethod
    def _validate_video_size(
        video_size: list[int] | tuple[int, int],
    ) -> tuple[int, int]:
        if len(video_size) != 2:
            raise ValueError(f"video_size 必须是 [height, width]，实际为 {video_size}")
        height, width = int(video_size[0]), int(video_size[1])
        if height <= 0 or width <= 0:
            raise ValueError(f"video_size 必须为正数，实际为 {(height, width)}")
        return height, width

    def _validate_episode_files(self) -> None:
        missing: list[str] = []
        for episode in self.episodes:
            data_path = episode_data_path(
                self.dataset_root, self.info, episode.episode_index
            )
            if not data_path.is_file():
                missing.append(str(data_path))
            for camera_key in self.camera_keys:
                video_path = episode_video_path(
                    self.dataset_root, self.info, episode.episode_index, camera_key
                )
                if not video_path.is_file():
                    missing.append(str(video_path))
            if len(missing) >= 8:
                break
        if missing:
            raise FileNotFoundError(
                "RoboCasa365 episode 文件缺失（最多展示 8 个）：\n- "
                + "\n- ".join(missing[:8])
            )

    @staticmethod
    def _column_to_matrix(
        table: Any, column_name: str, expected_dim: int, path: Path
    ) -> np.ndarray:
        if column_name not in table.column_names:
            raise ValueError(
                f"Parquet 缺少列 {column_name!r}：{path}；实际列={table.column_names}"
            )
        try:
            matrix = np.asarray(
                table[column_name].combine_chunks().to_pylist(), dtype=np.float32
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Parquet 列 {column_name!r} 无法转换为 float32 矩阵：{path}: {exc}"
            ) from exc
        if matrix.ndim != 2 or matrix.shape[1] != expected_dim:
            raise ValueError(
                f"Parquet 列 {column_name!r} 应为 [T,{expected_dim}]，实际为 {matrix.shape}：{path}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError(f"Parquet 列 {column_name!r} 包含 NaN/Inf：{path}")
        return np.ascontiguousarray(matrix)

    def _load_episode_arrays(
        self, episode_index: int, expected_length: int
    ) -> tuple[np.ndarray, np.ndarray]:
        cached = self._parquet_cache.get(episode_index)
        if cached is not None:
            self._parquet_cache.move_to_end(episode_index)
            return cached

        path = episode_data_path(self.dataset_root, self.info, episode_index)
        try:
            # Episode files are small; reading the full schema avoids treating dotted
            # LeRobot feature names as nested-field selectors in some PyArrow builds.
            table = pq.read_table(path)
        except Exception as exc:
            raise ValueError(f"读取 Parquet 失败：{path}: {exc}") from exc
        if table.num_rows != expected_length:
            raise ValueError(
                f"episode {episode_index} 元数据长度 {expected_length} 与 Parquet 行数 {table.num_rows} 不一致：{path}"
            )

        states = self._column_to_matrix(
            table, "observation.state", self.proprio_dim, path
        )
        actions = self._column_to_matrix(table, "action", self.action_dim, path)
        frame_index = np.asarray(
            table["frame_index"].combine_chunks().to_pylist(), dtype=np.int64
        )
        episode_indices = np.asarray(
            table["episode_index"].combine_chunks().to_pylist(), dtype=np.int64
        )
        if not np.array_equal(frame_index, np.arange(expected_length, dtype=np.int64)):
            raise ValueError(f"Parquet frame_index 不是从 0 连续递增：{path}")
        if not np.all(episode_indices == episode_index):
            raise ValueError(
                f"Parquet episode_index 与文件 episode {episode_index} 不一致：{path}"
            )

        value = (states, actions)
        if self.parquet_cache_size > 0:
            self._parquet_cache[episode_index] = value
            self._parquet_cache.move_to_end(episode_index)
            while len(self._parquet_cache) > self.parquet_cache_size:
                self._parquet_cache.popitem(last=False)
        return value

    def _video_reader(self, episode_index: int, camera_key: str) -> VideoReader:
        cache_key = (episode_index, camera_key)
        cached = self._video_cache.get(cache_key)
        if cached is not None:
            self._video_cache.move_to_end(cache_key)
            return cached

        path = episode_video_path(
            self.dataset_root, self.info, episode_index, camera_key
        )
        try:
            reader = VideoReader(str(path), ctx=cpu(0))
        except Exception as exc:
            raise ValueError(f"打开 MP4 失败：{path}: {exc}") from exc
        if self.video_cache_size > 0:
            self._video_cache[cache_key] = reader
            self._video_cache.move_to_end(cache_key)
            while len(self._video_cache) > self.video_cache_size:
                self._video_cache.popitem(last=False)
        return reader

    def _read_video_frames(
        self, episode_index: int, camera_key: str, frame_ids: np.ndarray
    ) -> torch.Tensor:
        reader = self._video_reader(episode_index, camera_key)
        if int(frame_ids[-1]) >= len(reader):
            path = episode_video_path(
                self.dataset_root, self.info, episode_index, camera_key
            )
            raise ValueError(
                f"请求帧 {int(frame_ids[-1])} 超出视频长度 {len(reader)}：{path}"
            )
        try:
            frames = reader.get_batch(frame_ids.tolist()).asnumpy()
        except Exception as exc:
            path = episode_video_path(
                self.dataset_root, self.info, episode_index, camera_key
            )
            raise ValueError(f"解码 MP4 帧失败：{path}: {exc}") from exc
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"视频帧应为 [T,H,W,3]，实际为 {frames.shape}")

        tensor = (
            torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float()
        )
        tensor = tensor.div_(127.5).sub_(1.0)
        if tuple(tensor.shape[-2:]) != self.video_size:
            tensor = F.interpolate(
                tensor,
                size=self.video_size,
                mode="bilinear",
                align_corners=False,
                antialias=False,
            )
        return tensor
