#!/usr/bin/env python3
"""Launch eight fixed broker/server pairs for one M6 evaluation allocation."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.robocasa365_m6_topology import (  # noqa: E402
    load_m6_evaluation_topology,
    tasks_for_server,
)
from project_tools.training_run import write_json_atomic  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须为对象：{path}")
    return payload


def _stop_processes(processes: list[subprocess.Popen[Any]]) -> list[int]:
    for process in reversed(processes):
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
    deadline = time.monotonic() + 60
    for process in reversed(processes):
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.terminate()
    for process in reversed(processes):
        if process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    return [int(process.returncode or 0) for process in processes]


def _wait_server_ready(
    process: subprocess.Popen[Any], report_path: Path, timeout_seconds: float
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        return_code = process.poll()
        if report_path.is_file():
            report = _read_json(report_path)
            if report.get("result") == "ready" and report.get("ok") is True:
                return report
            if report.get("result") == "fail":
                raise RuntimeError(f"policy server startup failed: {report.get('error')}")
        if return_code is not None:
            raise RuntimeError(f"policy server exited before READY: code={return_code}")
        time.sleep(2)
    raise TimeoutError(f"policy server READY timeout：{report_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--wan-checkpoint-dir", required=True)
    parser.add_argument("--multitask-manifest", required=True)
    parser.add_argument("--statistics-path", required=True)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--stop-file", required=True)
    parser.add_argument("--startup-timeout-seconds", type=float, default=1200)
    args = parser.parse_args()
    if args.startup_timeout_seconds <= 0:
        parser.error("startup timeout 必须为正")
    return args


def main() -> int:
    args = _parse_args()
    topology = load_m6_evaluation_topology(args.topology, REPO_ROOT)
    log_root = Path(args.log_root).expanduser().resolve()
    server_root = log_root / "servers"
    server_root.mkdir(parents=True, exist_ok=True)
    ready_file = Path(args.ready_file).expanduser().resolve()
    stop_file = Path(args.stop_file).expanduser().resolve()
    ready_file.unlink(missing_ok=True)
    stop_file.unlink(missing_ok=True)

    processes: list[subprocess.Popen[Any]] = []
    log_handles: list[Any] = []
    server_reports: list[dict[str, Any]] = []
    stopping = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        for server_id in range(8):
            server = topology["servers"][server_id]
            broker_log = (server_root / f"broker_{server_id}.log").open(
                "a", encoding="utf-8"
            )
            log_handles.append(broker_log)
            broker = subprocess.Popen(
                [
                    sys.executable,
                    str(REPO_ROOT / "evaluation" / "run_robocasa365_policy_broker.py"),
                    "--frontend-port",
                    str(server["frontend_port"]),
                    "--backend-port",
                    str(server["backend_port"]),
                ],
                cwd=REPO_ROOT,
                stdout=broker_log,
                stderr=subprocess.STDOUT,
            )
            processes.append(broker)
        time.sleep(2)
        for process in processes:
            if process.poll() is not None:
                raise RuntimeError(f"broker startup failed: code={process.returncode}")

        # Load one server at a time so eight mmap/checkpoint initializations do not
        # create an avoidable Store or host-memory spike.
        for server_id in range(8):
            server = topology["servers"][server_id]
            assigned_tasks = tasks_for_server(topology, server_id)
            report_path = server_root / f"server_{server_id}_status.json"
            server_log = (server_root / f"server_{server_id}.log").open(
                "a", encoding="utf-8"
            )
            log_handles.append(server_log)
            command = [
                sys.executable,
                str(REPO_ROOT / "evaluation" / "robocasa365_policy_server.py"),
                "--experiment-dir",
                args.experiment_dir,
                "--checkpoint",
                args.checkpoint,
                "--wan-checkpoint-dir",
                args.wan_checkpoint_dir,
                "--multitask-manifest",
                args.multitask_manifest,
                "--statistics-path",
                args.statistics_path,
                "--broker-port",
                str(server["backend_port"]),
                "--denoise-steps",
                str(topology["video_denoise_steps"]),
                "--action-denoise-steps",
                str(topology["action_denoise_steps"]),
                "--model-seed",
                str(topology["model_seed"]),
                "--minimum-requests",
                "1",
                "--graceful-stop-is-pass",
                "--startup-report",
                str(report_path),
                "--disable-request-journal",
                "--compact-report",
            ]
            for task in assigned_tasks:
                command.extend(["--allowed-task", task])
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(server["gpu"])
            policy = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=server_log,
                stderr=subprocess.STDOUT,
            )
            processes.append(policy)
            report = _wait_server_ready(
                policy, report_path, args.startup_timeout_seconds
            )
            server_reports.append(
                {
                    "server_id": server_id,
                    "gpu": server["gpu"],
                    "tasks": assigned_tasks,
                    "report": str(report_path),
                    "checkpoint": report["checkpoint"],
                    "model_seed": report["model_seed"],
                }
            )
            print(
                f"[READY] server={server_id} gpu={server['gpu']} tasks={assigned_tasks}",
                flush=True,
            )

        write_json_atomic(
            ready_file,
            {
                "schema_version": 1,
                "result": "ready",
                "ok": True,
                "topology": topology["name"],
                "model_seed": topology["model_seed"],
                "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
                "servers": server_reports,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        print(f"[READY] policy pool: {ready_file}", flush=True)
        while not stopping and not stop_file.is_file():
            for process in processes:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"policy pool process exited early: pid={process.pid} code={process.returncode}"
                    )
            time.sleep(2)
        return_codes = _stop_processes(processes)
        # Brokers normally return 0 on SIGINT. Servers return 0 only after at least
        # one successful request and no failed requests.
        ok = all(code == 0 for code in return_codes)
        summary = {
            "schema_version": 1,
            "result": "pass" if ok else "fail",
            "ok": ok,
            "return_codes": return_codes,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json_atomic(server_root / "policy_pool_summary.json", summary)
        return 0 if ok else 1
    except Exception as exc:
        write_json_atomic(
            server_root / "policy_pool_summary.json",
            {
                "schema_version": 1,
                "result": "fail",
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        print(f"[FAIL] policy pool: {type(exc).__name__}: {exc}", file=sys.stderr)
        _stop_processes(processes)
        return 1
    finally:
        for handle in log_handles:
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
