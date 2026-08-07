#!/usr/bin/env python3
"""审计当前 Conda 环境能否复用为 RoboCasa365 atomic simulator。"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from project_tools.robocasa365_simulator_environment import audit_simulator_environment


DEFAULT_MANIFEST = REPO_ROOT / "configs" / "environment" / "robocasa365_simulator.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "cluster" / "robocasa365_simulator_environment_latest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="simulator 环境契约 JSON。")
    parser.add_argument(
        "--output",
        "--log-file",
        dest="output",
        default=str(DEFAULT_OUTPUT),
        help="持久化 JSON 日志路径。",
    )
    parser.add_argument(
        "--runtime-smoke",
        action="store_true",
        help="创建 CloseFridge target 环境，验证 reset、12D 动作、单步和 EGL 三路 RGB。",
    )
    parser.add_argument(
        "--runtime-timeout",
        type=int,
        default=600,
        help="runtime smoke 子进程超时秒数（默认 600）。",
    )
    args = parser.parse_args()

    fatal_error = False
    try:
        report = audit_simulator_environment(
            args.manifest,
            REPO_ROOT,
            runtime_smoke=args.runtime_smoke,
            runtime_timeout=args.runtime_timeout,
        )
    except Exception as exc:
        fatal_error = True
        report = {
            "schema_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "ok": False,
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
