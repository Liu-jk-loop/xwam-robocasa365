"""Single dataset construction boundary for legacy and RoboCasa365 formats."""

from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf

from data.robot_dataset import RobotDataset


def _as_dict(config: Any) -> dict[str, Any]:
    if OmegaConf.is_config(config):
        payload = OmegaConf.to_container(config, resolve=True)
    else:
        payload = dict(config)
    if not isinstance(payload, dict):
        raise TypeError("dataset config 必须解析为字典")
    return payload


def build_dataset(config: Any, *, use_depth: bool, augment: bool | None = None):
    payload = _as_dict(config)
    dataset_format = payload.pop("format", "legacy_json_video")
    training_ready = bool(payload.pop("training_ready", True))
    configured_augment = bool(payload.pop("augment", True))
    payload["augment"] = configured_augment if augment is None else bool(augment)

    if dataset_format == "legacy_json_video":
        payload["use_depth"] = bool(use_depth)
        return RobotDataset(**payload)

    if dataset_format == "robocasa365_lerobot_v21":
        if not training_ready:
            raise RuntimeError(
                "RoboCasa365 M1 配置目前仅供 batch audit；M2 完成 state/action schema 与 normalization 后才允许训练"
            )
        from data.robocasa365_dataset import RoboCasa365Dataset

        payload["use_depth"] = bool(use_depth)
        return RoboCasa365Dataset(**payload)

    raise ValueError(f"不支持的数据格式：{dataset_format!r}")
