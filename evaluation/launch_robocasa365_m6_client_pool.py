#!/usr/bin/env python3
"""Launch selected RoboCasa simulator clients from the M6 topology."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.robocasa365_m6_topology import (  # noqa: E402
    load_m6_evaluation_topology,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--episodes-per-task", type=int)
    parser.add_argument(
        "--cuda-visible-devices",
        default="0,1,2,3",
        help="传给simulator client的CUDA_VISIBLE_DEVICES；单GPU评测使用0。",
    )
    parser.add_argument(
        "--client-id",
        type=int,
        action="append",
        dest="client_ids",
        help="只启动指定client；可重复。默认启动0..15。",
    )
    args = parser.parse_args()
    if args.episodes_per_task is not None and args.episodes_per_task <= 0:
        parser.error("episodes per task 必须为正")
    visible_devices = args.cuda_visible_devices.split(",")
    if not visible_devices or any(not item.isdigit() for item in visible_devices):
        parser.error("cuda visible devices必须是逗号分隔的非负整数")
    if args.client_ids is not None:
        if len(set(args.client_ids)) != len(args.client_ids):
            parser.error("client id不允许重复")
        if any(client_id not in range(16) for client_id in args.client_ids):
            parser.error("client id必须位于0..15")
    return args


def main() -> int:
    args = _parse_args()
    topology = load_m6_evaluation_topology(args.topology, REPO_ROOT)
    client_ids = sorted(args.client_ids or range(16))
    output_root = Path(args.output_root).expanduser().resolve()
    log_root = Path(args.log_root).expanduser().resolve() / "clients"
    log_root.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen[Any]] = []
    handles: list[Any] = []
    stopping = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True
        for process in processes:
            if process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        for client_id in client_ids:
            client = topology["clients"][client_id]
            comparison_group = client.get("comparison_group")
            client_output_root = (
                output_root / str(comparison_group)
                if comparison_group is not None
                else output_root
            )
            log_handle = (log_root / f"client_{client_id:02d}.log").open(
                "a", encoding="utf-8"
            )
            handles.append(log_handle)
            command = [
                sys.executable,
                str(REPO_ROOT / "evaluation" / "run_robocasa365_m6_client.py"),
                "--topology",
                args.topology,
                "--client-id",
                str(client_id),
                "--output-root",
                str(client_output_root),
            ]
            if args.episodes_per_task is not None:
                command.extend(
                    ["--episodes-per-task", str(args.episodes_per_task)]
                )
            environment = dict(os.environ)
            environment["MUJOCO_GL"] = "egl"
            environment["PYOPENGL_PLATFORM"] = "egl"
            # Legacy robosuite import validates MUJOCO_EGL_DEVICE_ID against
            # the physical ids in CUDA_VISIBLE_DEVICES. The child keeps this
            # value through RoboCasa registration, then switches it to local 0
            # immediately before the one-device EGL context is constructed.
            physical_egl_device = args.cuda_visible_devices.split(",", 1)[0]
            environment["MUJOCO_EGL_DEVICE_ID"] = physical_egl_device
            environment["XWAM_ROBOCASA_IMPORT_EGL_DEVICE"] = physical_egl_device
            environment["EGL_DEVICE_ID"] = "0"
            environment["ROBOSUITE_RENDER_GPU_DEVICE_ID"] = "0"
            environment["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            processes.append(process)
            print(
                f"[START] client={client_id} server={client['server_id']} "
                f"cuda_visible={args.cuda_visible_devices} "
                f"mujoco_egl_import={physical_egl_device} egl_runtime=0 "
                f"group={comparison_group} "
                f"tasks={[row['name'] for row in client['tasks']]}",
                flush=True,
            )
        return_codes = [process.wait() for process in processes]
        failed = [index for index, code in enumerate(return_codes) if code != 0]
        if stopping:
            print("[STOP] client pool interrupted; the interrupted episode will restart")
            return 130
        if failed:
            print(f"[FAIL] clients exited nonzero: {failed}", file=sys.stderr)
            return 1
        print("[PASS] all 16 clients completed", flush=True)
        return 0
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
