#!/usr/bin/env python3
"""逐 episode 加载 RoboCasa365 MJCF/state 并验证三路 RGB-D 渲染对齐。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.robocasa365_contract import load_task_manifest  # noqa: E402
from data.robocasa365_multitask import resolve_task_dataset_directory  # noqa: E402
from project_tools.robocasa365_depth_render import (  # noqa: E402
    parse_frame_fractions,
    probe_task_depth_replay,
)
from project_tools.training_run import (  # noqa: E402
    collect_git_state,
    write_json_atomic,
)


def _explicit_paths(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        task, separator, path = value.partition("=")
        if not separator or not task or not path:
            raise ValueError(f"--task-path 必须为 TaskName=/absolute/path，实际为 {value!r}")
        if task in parsed:
            raise ValueError(f"--task-path 重复指定任务：{task}")
        parsed[task] = path
    return parsed


def _select_tasks(args: argparse.Namespace) -> list[tuple[str, Path]]:
    explicit = _explicit_paths(args.task_path)
    if args.dataset:
        if not args.task_name:
            raise ValueError("--dataset 模式必须提供 --task-name")
        if explicit:
            raise ValueError("--dataset 模式不接受 --task-path")
        return [(args.task_name, Path(args.dataset).expanduser().resolve())]
    if args.task_name:
        raise ValueError("--task-name 只用于 --dataset 模式")
    manifest = load_task_manifest(args.task_manifest)
    unknown = sorted(set(explicit) - set(manifest.tasks))
    if unknown:
        raise ValueError(f"--task-path 包含 manifest 外任务：{unknown}")
    return [
        (
            task_name,
            Path(explicit[task_name]).expanduser().resolve()
            if task_name in explicit
            else resolve_task_dataset_directory(args.dataset_root, task_name),
        )
        for task_name in manifest.tasks
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="单任务数据目录或其 lerobot 子目录。")
    source.add_argument("--dataset-root", help="包含各任务目录的 pretrain/atomic 根目录。")
    parser.add_argument("--task-name", help="与 --dataset 配套的任务名。")
    parser.add_argument(
        "--task-manifest",
        default="configs/tasks/robocasa365_atomic9_fastwam_overlap.json",
        help="--dataset-root 模式的 atomic 任务清单。",
    )
    parser.add_argument(
        "--task-path",
        action="append",
        default=[],
        help="多日期任务显式指定 TaskName=/absolute/path。",
    )
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=3,
        help="每任务回放前 N 个 episode；0 表示全部。",
    )
    parser.add_argument(
        "--frame-fractions",
        default="0,0.5,1",
        help="每 episode 在 [0,1] 时间上抽查的帧位置。",
    )
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument(
        "--max-rgb-mae",
        type=float,
        default=12.0,
        help="翻转后的 MuJoCo RGB 与原 MP4 帧的最大允许 MAE。",
    )
    parser.add_argument("--artifacts-dir", required=True, help="RGB/depth 数组和对比图输出目录。")
    parser.add_argument("--output", required=True, help="JSON 机器报告路径。")
    args = parser.parse_args()

    tasks: list[dict[str, object]] = []
    errors: list[str] = []
    try:
        frame_fractions = parse_frame_fractions(args.frame_fractions)
        selected = _select_tasks(args)
        for task_name, dataset_path in selected:
            print(f"[RGBD-P1] task={task_name} dataset={dataset_path}", flush=True)
            report = probe_task_depth_replay(
                dataset_path,
                task_name=task_name,
                episodes_per_task=args.episodes_per_task,
                frame_fractions=frame_fractions,
                height=args.height,
                width=args.width,
                max_rgb_mae=args.max_rgb_mae,
                artifacts_root=args.artifacts_dir,
            )
            tasks.append(report)
            print(
                f"[RGBD-P1] task={task_name} result={report['result']} "
                f"episodes={report['episodes_passed']}/{report['episodes_requested']} "
                f"state_widths={report['state_widths']}",
                flush=True,
            )
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")

    failed_tasks = [str(item.get("task_name")) for item in tasks if item.get("ok") is not True]
    ok = bool(tasks) and not errors and not failed_tasks
    report = {
        "schema_version": 1,
        "phase": "RGBD-P1-per-episode-render-probe",
        "scope": "atomic_only",
        "git": collect_git_state(REPO_ROOT),
        "task_manifest": args.task_manifest if args.dataset_root else None,
        "episodes_per_task": args.episodes_per_task,
        "frame_fractions": args.frame_fractions,
        "render_shape": [args.height, args.width],
        "max_rgb_mae": args.max_rgb_mae,
        "expected_tasks": len(tasks) + len(errors),
        "passed_tasks": sum(item.get("ok") is True for item in tasks),
        "failed_tasks": failed_tasks,
        "tasks": tasks,
        "artifacts_dir": str(Path(args.artifacts_dir).expanduser().resolve()),
        "depth_encoding_decision": {
            "semantic_target": "inverse_depth",
            "metric_depth_saved": True,
            "preview_is_training_encoding": False,
            "training_uint8_formula": "not_frozen_by_this_probe",
        },
        "errors": errors,
        "ok": ok,
        "result": "pass" if ok else "fail",
    }
    output = write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"RGB-D per-episode render report: {output.resolve()}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
