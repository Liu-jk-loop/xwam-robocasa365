#!/usr/bin/env python3
"""审计 PandaOmron schema、normalization round-trip 和公开 X-WAM checkpoint shape。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.robocasa365_contract import inspect_dataset, load_task_manifest
from data.robocasa365_index import episode_data_path, load_episode_records
from data.robocasa365_schema import (
    PandaOmronTensorCodec,
    load_panda_omron_schema,
    load_statistics,
    validate_dataset_modality,
)


DEFAULT_SCHEMA = REPO_ROOT / "configs" / "schemas" / "robocasa365_panda_omron_v1.json"
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "tasks" / "robocasa365_atomic_seen.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "cluster" / "robocasa365_m2_contract_latest.json"


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _resolve_checkpoint(path: str | Path) -> Path:
    requested = Path(path).expanduser().resolve()
    candidates = (
        requested,
        requested / "checkpoint" / "mp_rank_00_model_states.pt",
        requested / "pretrained" / "checkpoints" / "last.ckpt" / "checkpoint" / "mp_rank_00_model_states.pt",
        requested / "checkpoints" / "last.ckpt" / "checkpoint" / "mp_rank_00_model_states.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "未找到 mp_rank_00_model_states.pt；检查 checkpoint 根目录或直接传入文件："
        + str(requested)
    )


def _checkpoint_inventory(checkpoint_path: Path) -> dict[str, Any]:
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
    state = checkpoint.get("module", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint 顶层或 module 不是 state dict：{checkpoint_path}")

    tensor_shapes: dict[str, list[int]] = {}
    tensor_dtypes: dict[str, str] = {}
    for key, value in state.items():
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            tensor_shapes[str(key)] = [int(dim) for dim in value.shape]
            tensor_dtypes[str(key)] = str(value.dtype)

    boundary = {
        key: {"shape": shape, "dtype": tensor_dtypes[key]}
        for key, shape in tensor_shapes.items()
        if "action_encoder" in key
        or "action_decoder" in key
        or "proprio_encoder" in key
        or "proprio_decoder" in key
    }

    def find_shape(suffix: str) -> list[int] | None:
        matches = [shape for key, shape in tensor_shapes.items() if key.endswith(suffix)]
        if len(matches) > 1:
            raise ValueError(f"checkpoint 中参数后缀 {suffix!r} 匹配多个 key")
        return matches[0] if matches else None

    action_encoder = find_shape("model.action_encoder.0.weight")
    action_decoder = find_shape("model.action_decoder.2.weight")
    proprio_encoder = find_shape("model.proprio_encoder.0.weight")
    proprio_decoder = find_shape("model.proprio_decoder.2.weight")
    inferred = {
        "action_input_dim": action_encoder[-1] if action_encoder else None,
        "action_output_dim": action_decoder[0] if action_decoder else None,
        "proprio_input_dim": proprio_encoder[-1] if proprio_encoder else None,
        "proprio_output_dim": proprio_decoder[0] if proprio_decoder else None,
    }
    return {
        "path": str(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size,
        "tensor_count": len(tensor_shapes),
        "boundary_parameters": boundary,
        "inferred_dimensions": inferred,
    }


def _statistics_summary(statistics: Any) -> dict[str, Any]:
    import numpy as np

    span = statistics.q99 - statistics.q01
    return {
        "dimension": int(statistics.q01.shape[0]),
        "q01": statistics.q01.tolist(),
        "q99": statistics.q99.tolist(),
        "degenerate_quantile_dimensions": np.flatnonzero(span <= 1e-8).astype(int).tolist(),
        "min": statistics.minimum.tolist() if statistics.minimum is not None else None,
        "max": statistics.maximum.tolist() if statistics.maximum is not None else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="单个 RoboCasa365 atomic 任务目录或 lerobot/。")
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--task-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA))
    parser.add_argument("--checkpoint", required=True, help="X-WAM checkpoint 根目录、last.ckpt 或 pt 文件。")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--output", "--log-file", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    output_path = Path(args.output).expanduser().resolve()
    report: dict[str, Any] = {
        "schema_version": 1,
        "git_commit": _git_commit(),
        "dataset": str(Path(args.dataset).expanduser()),
        "task_name": args.task_name,
        "result": "fail",
        "ok": False,
    }

    try:
        import numpy as np
        import pyarrow
        import pyarrow.parquet as pq
        import torch

        manifest = load_task_manifest(args.task_manifest)
        dataset_contract = inspect_dataset(
            args.dataset,
            manifest=manifest,
            task_name=args.task_name,
            require_data=True,
            require_videos=True,
        )
        if not dataset_contract.ok:
            raise ValueError("atomic-only 数据契约失败：\n- " + "\n- ".join(dataset_contract.errors))

        schema = load_panda_omron_schema(args.schema)
        modality = validate_dataset_modality(args.dataset, schema)
        state_stats = load_statistics(args.dataset, schema.state)
        action_stats = load_statistics(args.dataset, schema.action)
        codec = PandaOmronTensorCodec(schema, state_stats, action_stats)

        root, info, episodes = load_episode_records(args.dataset)
        episode_by_index = {episode.episode_index: episode for episode in episodes}
        if args.episode_index not in episode_by_index:
            raise ValueError(f"episode_index {args.episode_index} 不存在")
        parquet_path = episode_data_path(root, info, args.episode_index)
        table = pq.read_table(parquet_path)
        raw_state = np.asarray(table[schema.state.original_key].combine_chunks().to_pylist(), dtype=np.float32)
        raw_action = np.asarray(table[schema.action.original_key].combine_chunks().to_pylist(), dtype=np.float32)

        state_unclipped = codec.encode_state(raw_state, clip=False)
        action_unclipped = codec.encode_action(raw_action, clip=False)
        state_normalized = codec.encode_state(raw_state, clip=True)
        action_normalized = codec.encode_action(raw_action, clip=True)
        state_roundtrip = codec.decode_state(state_unclipped)
        action_roundtrip = codec.decode_action(action_unclipped)
        environment_action = codec.decode_action(action_normalized, discretize_control_mode=True)

        control_slice = next(
            slice(component.start, component.end)
            for component in schema.action.components
            if component.name == "control_mode"
        )
        control_values, control_counts = np.unique(raw_action[..., control_slice], return_counts=True)
        checkpoint = _checkpoint_inventory(_resolve_checkpoint(args.checkpoint))
        inferred = checkpoint["inferred_dimensions"]

        checks = {
            "atomic_only_dataset_contract": dataset_contract.ok,
            "modality_matches_versioned_schema": True,
            "state_shape_16": raw_state.ndim == 2 and raw_state.shape[1] == schema.state.dimension == 16,
            "action_shape_12": raw_action.ndim == 2 and raw_action.shape[1] == schema.action.dimension == 12,
            "state_unclipped_roundtrip": float(np.max(np.abs(state_roundtrip - raw_state))) < 1e-5,
            "action_unclipped_roundtrip": float(np.max(np.abs(action_roundtrip - raw_action))) < 1e-5,
            "normalized_state_finite": bool(np.isfinite(state_normalized).all()),
            "normalized_action_finite": bool(np.isfinite(action_normalized).all()),
            "normalized_state_range": float(np.max(np.abs(state_normalized))) <= 1.00001,
            "normalized_action_range": float(np.max(np.abs(action_normalized))) <= 1.00001,
            "control_mode_is_sign": set(np.round(control_values, 6).tolist()).issubset({-1.0, 1.0}),
            "environment_action_keeps_12d": environment_action.shape == raw_action.shape,
            "checkpoint_legacy_action_14d": inferred["action_input_dim"] == inferred["action_output_dim"] == 14,
            "checkpoint_proprio_16d": inferred["proprio_input_dim"] == inferred["proprio_output_dim"] == 16,
        }

        report.update(
            {
                "versions": {
                    "python": sys.version.split()[0],
                    "torch": torch.__version__,
                    "numpy": np.__version__,
                    "pyarrow": pyarrow.__version__,
                },
                "schema": {
                    "path": str(Path(args.schema).expanduser().resolve()),
                    "schema_id": schema.schema_id,
                    "canonical_sha256": schema.canonical_sha256,
                    "state_dimension": schema.state.dimension,
                    "action_dimension": schema.action.dimension,
                    "checkpoint_mapping": schema.checkpoint_mapping,
                },
                "dataset_contract": dataset_contract.to_dict(),
                "modality": modality,
                "statistics": {
                    "state": _statistics_summary(state_stats),
                    "action": _statistics_summary(action_stats),
                },
                "sample": {
                    "episode_index": args.episode_index,
                    "parquet": str(parquet_path),
                    "frames": int(raw_state.shape[0]),
                    "state_components": codec.component_summary(raw_state, "state"),
                    "action_components": codec.component_summary(raw_action, "action"),
                    "control_mode_values": [
                        {"value": float(value), "count": int(count)}
                        for value, count in zip(control_values, control_counts)
                    ],
                    "state_unclipped_roundtrip_max_error": float(np.max(np.abs(state_roundtrip - raw_state))),
                    "action_unclipped_roundtrip_max_error": float(np.max(np.abs(action_roundtrip - raw_action))),
                    "state_clip_fraction": float(np.mean(np.abs(state_unclipped) > 1.0)),
                    "action_clip_fraction": float(np.mean(np.abs(action_unclipped) > 1.0)),
                },
                "checkpoint": checkpoint,
                "mapping_plan": {
                    "exact_load": "所有名称和 shape 都匹配、且不属于 schema-sensitive boundary 的参数",
                    "action_partial_copy": "legacy 左臂 0:7 -> PandaOmron 机械臂 5:12",
                    "action_initialize": "PandaOmron base_motion+control_mode 0:5",
                    "proprio_initialize": "proprio encoder/decoder 边界层因 16D 语义变化全部重新初始化",
                    "strictness": "任何未声明 shape mismatch 都阻塞加载",
                },
                "checks": checks,
                "ok": all(checks.values()),
            }
        )
        report["result"] = "pass" if report["ok"] else "fail"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()

    _write_report(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"报告已写入：{output_path}", file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
