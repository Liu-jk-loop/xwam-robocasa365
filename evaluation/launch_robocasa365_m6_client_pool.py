#!/usr/bin/env python3
"""Launch all sixteen RoboCasa simulator clients from the M6 topology."""

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
    args = parser.parse_args()
    if args.episodes_per_task is not None and args.episodes_per_task <= 0:
        parser.error("episodes per task 必须为正")
    return args


def main() -> int:
    args = _parse_args()
    topology = load_m6_evaluation_topology(args.topology, REPO_ROOT)
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
        for client_id in range(16):
            client = topology["clients"][client_id]
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
                str(output_root),
            ]
            if args.episodes_per_task is not None:
                command.extend(
                    ["--episodes-per-task", str(args.episodes_per_task)]
                )
            environment = dict(os.environ)
            environment["MUJOCO_GL"] = "egl"
            environment["PYOPENGL_PLATFORM"] = "egl"
            # The validated RoboCasa EDF exposes one EGL device even when the
            # Slurm step owns four GPUs. Match the working FastWAM client path.
            environment["MUJOCO_EGL_DEVICE_ID"] = "0"
            environment["EGL_DEVICE_ID"] = "0"
            environment["ROBOSUITE_RENDER_GPU_DEVICE_ID"] = "0"
            environment["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
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
                f"egl_device=0 tasks={[row['name'] for row in client['tasks']]}",
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
