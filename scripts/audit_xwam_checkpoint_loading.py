#!/usr/bin/env python3
"""在星光上实例化 12D X-WAM，并持久化 checkpoint 适配证据。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-config",
        default="configs/model/wan22_5b_robocasa365_atomic_m2.yaml",
    )
    parser.add_argument("--wan-checkpoint-dir", required=True)
    parser.add_argument("--xwam-checkpoint", default=None)
    parser.add_argument(
        "--schema", default="configs/schemas/robocasa365_panda_omron_v1.json"
    )
    parser.add_argument(
        "--mode", choices=("xwam_pretrained", "wan_base"), default="xwam_pretrained"
    )
    parser.add_argument("--action-num", type=int, default=4)
    parser.add_argument(
        "--output", default="logs/cluster/xwam_checkpoint_loading_latest.json"
    )
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()

    try:
        import torch
        from omegaconf import OmegaConf

        from runners.xwam_runner import XWAMRunner
        from utils.xwam_checkpoint_loader import initialize_xwam_runner

        config = OmegaConf.load(args.model_config)
        config.wan_checkpoint_dir = args.wan_checkpoint_dir
        config.pretrained_checkpoint = args.xwam_checkpoint
        config.initialization_mode = args.mode
        config.checkpoint_schema_path = args.schema
        config.action_num = args.action_num
        config.action_dim = 12
        config.proprio_dim = 16
        config.use_depth = False

        runner = XWAMRunner(config, run_depth=False)
        report = initialize_xwam_runner(
            runner,
            mode=args.mode,
            checkpoint=args.xwam_checkpoint,
            schema_path=args.schema,
            report_path=output,
        )
        report["git_commit"] = _git_commit()
        report["model_config"] = str(Path(args.model_config).expanduser().resolve())
        report["environment"] = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        _write(output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"报告已写入：{output}", file=sys.stderr)
        return 0
    except Exception as exc:
        if output.is_file():
            try:
                report = json.loads(output.read_text(encoding="utf-8"))
            except Exception:
                report = {}
        else:
            report = {}
        report.update(
            {
                "git_commit": _git_commit(),
                "result": "fail",
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        _write(output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"报告已写入：{output}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
