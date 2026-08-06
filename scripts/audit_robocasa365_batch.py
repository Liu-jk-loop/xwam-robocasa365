#!/usr/bin/env python3
"""读取一个真实 RoboCasa365 RGB-only batch，并输出机器可读验收报告。"""

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


DEFAULT_MANIFEST = REPO_ROOT / "configs" / "tasks" / "robocasa365_atomic_seen.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "cluster" / "robocasa365_batch_latest.json"
DEFAULT_CAMERAS = (
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
)
DEFAULT_CAMERA_TYPES = ("static", "static", "dynamic")


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


def _tensor_summary(value: Any) -> dict[str, Any]:
    import torch

    if not isinstance(value, torch.Tensor):
        return {"type": type(value).__name__, "value": value}
    output: dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(torch.isfinite(value).all().item()) if value.is_floating_point() else True,
    }
    if value.numel() > 0 and (value.is_floating_point() or value.dtype != torch.bool):
        output["min"] = float(value.min().item())
        output["max"] = float(value.max().item())
    return output


def _same_sample(first: dict[str, Any], second: dict[str, Any]) -> bool:
    import torch

    if first.keys() != second.keys():
        return False
    for key in first:
        left, right = first[key], second[key]
        if isinstance(left, torch.Tensor):
            if not isinstance(right, torch.Tensor) or not torch.equal(left, right):
                return False
        elif left != right:
            return False
    return True


def _write_report(output_path: Path, report: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="单个 atomic 任务目录或其 lerobot/ 子目录。")
    parser.add_argument("--task-name", required=True, help="任务类名，例如 CloseFridge。")
    parser.add_argument("--task-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--index", type=int, default=0, help="要读取的全局 clip 索引。")
    parser.add_argument("--sequence-length", type=int, default=9)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--action-skip", type=int, default=1)
    parser.add_argument("--video-height", type=int, default=256)
    parser.add_argument("--video-width", type=int, default=320)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker 数；首次验收建议为 0。")
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
        import decord
        import numpy
        import pyarrow
        import torch
        from torch.utils.data import DataLoader

        from data.robocasa365_dataset import RoboCasa365Dataset

        config = {
            "dataset_path": args.dataset,
            "task_name": args.task_name,
            "task_manifest": args.task_manifest,
            "camera_keys": list(DEFAULT_CAMERAS),
            "camera_types": list(DEFAULT_CAMERA_TYPES),
            "sequence_length": args.sequence_length,
            "frame_skip": args.frame_skip,
            "action_skip": args.action_skip,
            "video_size": [args.video_height, args.video_width],
            "augment": False,
            "use_depth": False,
            "normalization": "none",
        }
        dataset = RoboCasa365Dataset(**config)
        first = dataset[args.index]
        second = dataset[args.index]
        clip = dataset.clip_spec(args.index)

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
            multiprocessing_context="forkserver" if args.num_workers > 0 else None,
        )
        batch = next(iter(loader)) if args.index == 0 else next(iter(DataLoader([first], batch_size=1)))

        checks = {
            "dataset_nonempty": len(dataset) > 0,
            "video_shape": list(first["video"].shape)
            == [3, args.sequence_length, 3, args.video_height, args.video_width],
            "video_range": float(first["video"].min()) >= -1.00001
            and float(first["video"].max()) <= 1.00001,
            "proprio_shape": list(first["proprios"].shape) == [args.sequence_length, 16],
            "action_shape": list(first["actions"].shape)
            == [(args.sequence_length - 1) * args.frame_skip // args.action_skip, 12],
            "proprio_mask_all_valid": bool(torch.all(first["proprio_mask"] == 1).item()),
            "action_mask_all_valid": bool(torch.all(first["action_mask"] == 1).item()),
            "camera_type_mask": first["camera_type_mask"].tolist() == [0, 0, 1],
            "depth_absent": "depths" not in first,
            "finite_tensors": all(
                bool(torch.isfinite(first[key]).all().item())
                for key in ("video", "proprios", "actions")
            ),
            "deterministic_without_augmentation": _same_sample(first, second),
            "batch_dimension": list(batch["video"].shape[:2]) == [1, 3],
        }

        report.update(
            {
                "versions": {
                    "python": sys.version.split()[0],
                    "torch": torch.__version__,
                    "numpy": numpy.__version__,
                    "pyarrow": pyarrow.__version__,
                    "decord": decord.__version__,
                },
                "resolved_config": config,
                "lerobot_root": str(dataset.dataset_root),
                "episodes": len(dataset.episodes),
                "valid_clips": len(dataset),
                "clip": {
                    "global_index": args.index,
                    "episode_index": clip.episode_index,
                    "start_frame": clip.start_frame,
                    "frame_ids": list(clip.frame_ids),
                    "action_ids": list(clip.action_ids),
                },
                "paths": dataset.sample_paths(args.index),
                "sample": {key: _tensor_summary(value) for key, value in first.items()},
                "batch": {key: _tensor_summary(value) for key, value in batch.items()},
                "checks": checks,
                "normalization_status": "M1 raw state/action；禁止用于正式训练，M2 后启用归一化",
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
