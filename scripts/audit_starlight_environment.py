#!/usr/bin/env python3
"""审计当前 Conda 环境是否适合复用为星光 X-WAM policy 环境。"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from project_tools.starlight_environment import audit_environment


DEFAULT_MANIFEST = REPO_ROOT / "configs" / "environment" / "xwam_starlight.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "cluster" / "starlight_environment_latest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help=f"环境契约文件（默认：{DEFAULT_MANIFEST}）。",
    )
    parser.add_argument(
        "--output",
        "--log-file",
        dest="output",
        default=str(DEFAULT_OUTPUT),
        help=f"持久化 JSON 日志路径（默认：{DEFAULT_OUTPUT}）。",
    )
    parser.add_argument("--require-gpu", action="store_true", help="要求当前节点能被 Torch 识别为 CUDA GPU 节点。")
    parser.add_argument(
        "--include-deepspeed-report",
        action="store_true",
        help="同时运行官方 ds_report；不会主动预编译 DeepSpeed ops。",
    )
    parser.add_argument(
        "--skip-runtime-imports",
        action="store_true",
        help="只读已安装包版本，不在隔离子进程中测试关键 import。",
    )
    args = parser.parse_args()

    fatal_error = False
    try:
        report = audit_environment(
            args.manifest,
            REPO_ROOT,
            require_gpu=args.require_gpu,
            run_imports=not args.skip_runtime_imports,
            include_deepspeed_report=args.include_deepspeed_report,
        )
    except Exception as exc:  # 保证意外异常也能写出诊断日志
        fatal_error = True
        report = {
            "schema_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "ok": False,
            "clone_base_ok": False,
            "reuse_recommendation": "audit_failed",
            "fatal_error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    output_path = Path(args.output).expanduser().resolve()
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(output_path.name + ".tmp")
        temporary_path.write_text(payload + "\n", encoding="utf-8")
        temporary_path.replace(output_path)
    except OSError as exc:
        print(payload)
        print(f"无法写入审计日志 {output_path}：{exc}", file=sys.stderr)
        return 2

    print(f"审计日志已写入：{output_path}", file=sys.stderr)
    print(payload)
    if fatal_error:
        return 2
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
