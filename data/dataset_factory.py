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
        from data.robocasa365_multitask import (
            BalancedRoundRobinDataset,
            load_multitask_dataset_manifest,
        )

        payload["use_depth"] = bool(use_depth)
        multitask_manifest_path = payload.pop("multitask_manifest", None)
        expected_task_count = payload.pop("expected_task_count", None)
        if multitask_manifest_path is not None:
            if "dataset_path" in payload or "task_name" in payload:
                raise ValueError(
                    "multitask_manifest 不能与单任务 dataset_path/task_name 同时配置"
                )
            atomic_task_manifest = payload.get("task_manifest")
            if not atomic_task_manifest:
                raise ValueError("多任务 Dataset 必须配置 atomic task_manifest")
            manifest = load_multitask_dataset_manifest(
                multitask_manifest_path,
                atomic_task_manifest=atomic_task_manifest,
                expected_task_count=expected_task_count,
            )
            datasets = []
            for entry in manifest.tasks:
                child_payload = dict(payload)
                child_payload["dataset_path"] = entry.dataset_path
                child_payload["task_name"] = entry.task_name
                datasets.append(RoboCasa365Dataset(**child_payload))
            return BalancedRoundRobinDataset(
                datasets,
                task_names=[entry.task_name for entry in manifest.tasks],
                manifest_path=manifest.source_path,
            )
        return RoboCasa365Dataset(**payload)

    raise ValueError(f"不支持的数据格式：{dataset_format!r}")
