#!/usr/bin/env python3
"""审计 RoboCasa365 LeRobot 元数据、atomic-only 范围和可选媒体文件。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.robocasa365_contract import inspect_dataset, load_task_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "tasks" / "robocasa365_atomic_seen.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="单个 RoboCasa365 数据集目录或其 lerobot 子目录。")
    parser.add_argument("--task-name", help="显式任务类名；目录或 dataset_meta.json 无法推断时必须提供。")
    parser.add_argument(
        "--task-manifest",
        default=str(DEFAULT_MANIFEST),
        help=f"atomic-only 任务清单（默认：{DEFAULT_MANIFEST}）。",
    )
    parser.add_argument("--require-data", action="store_true", help="同时要求 data/ 下存在 Parquet 文件。")
    parser.add_argument("--require-videos", action="store_true", help="同时要求三路相机均存在 MP4 文件。")
    parser.add_argument("--output", help="可选 JSON 报告输出路径；默认只打印到标准输出。")
    args = parser.parse_args()

    manifest = load_task_manifest(args.task_manifest)
    summary = inspect_dataset(
        args.dataset,
        manifest=manifest,
        task_name=args.task_name,
        require_data=args.require_data,
        require_videos=args.require_videos,
    )
    report = json.dumps(summary.to_dict(), ensure_ascii=False, indent=2)
    print(report)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report + "\n", encoding="utf-8")

    return 0 if summary.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
