#!/usr/bin/env python3
"""审计透明物体的原始/forced-opaque depth 与 PointMap 同状态差异。"""

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

from data.robocasa365_multitask import resolve_task_dataset_directory  # noqa: E402
from project_tools.robocasa365_depth_render import parse_frame_fractions  # noqa: E402
from project_tools.robocasa365_transparent_pointmap import (  # noqa: E402
    audit_transparent_task_pointmaps,
)
from project_tools.training_run import (  # noqa: E402
    collect_git_state,
    write_json_atomic,
)


def _patterns(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise ValueError("--target-name-patterns 不能为空")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="单任务数据目录或其lerobot子目录。")
    source.add_argument("--dataset-root", help="pretrain/atomic数据根目录。")
    parser.add_argument("--task-name", default="CloseBlenderLid")
    parser.add_argument("--episodes-per-task", type=int, default=3)
    parser.add_argument("--frame-fractions", default="0,0.5,1")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--target-height", type=int, default=256)
    parser.add_argument("--target-width", type=int, default=320)
    parser.add_argument(
        "--target-name-patterns",
        default=r"blender.*lid,lid.*blender",
        help="显式blender_lid实体不可用时采用的严格正则兜底。",
    )
    parser.add_argument("--minimum-depth-change-mm", type=float, default=1.0)
    parser.add_argument("--max-reprojection-error-px", type=float, default=1e-3)
    parser.add_argument(
        "--max-float16-roundtrip-error-mm",
        type=float,
        default=2.0,
    )
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    errors: list[str] = []
    task_report: dict[str, object] | None = None
    try:
        dataset_path = (
            Path(args.dataset).expanduser().resolve()
            if args.dataset
            else resolve_task_dataset_directory(args.dataset_root, args.task_name)
        )
        task_report = audit_transparent_task_pointmaps(
            dataset_path,
            task_name=args.task_name,
            episodes_per_task=args.episodes_per_task,
            frame_fractions=parse_frame_fractions(args.frame_fractions),
            height=args.height,
            width=args.width,
            target_height=args.target_height,
            target_width=args.target_width,
            target_name_patterns=_patterns(args.target_name_patterns),
            minimum_depth_change_m=args.minimum_depth_change_mm / 1000.0,
            max_reprojection_error_px=args.max_reprojection_error_px,
            max_float16_roundtrip_error_m=(
                args.max_float16_roundtrip_error_mm / 1000.0
            ),
            artifacts_root=args.artifacts_dir,
        )
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")

    ok = task_report is not None and task_report.get("ok") is True and not errors
    report = {
        "schema_version": 1,
        "phase": "POINTMAP-P0-transparent-surface-audit",
        "scope": "atomic_only",
        "git": collect_git_state(REPO_ROOT),
        "task": task_report,
        "thresholds": {
            "minimum_depth_change_mm": args.minimum_depth_change_mm,
            "max_reprojection_error_px": args.max_reprojection_error_px,
            "max_float16_roundtrip_error_mm": args.max_float16_roundtrip_error_mm,
        },
        "errors": errors,
        "ok": ok,
        "result": "pass" if ok else "fail",
    }
    output = write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Transparent PointMap report: {output.resolve()}")
    if task_report is not None:
        print(
            f"transparent_pointmap_conclusion={task_report['conclusion']['status']}",
            flush=True,
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
