#!/usr/bin/env python3
"""Run one client queue from the M6 8-server/16-client evaluation topology."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.robocasa365_m6_topology import (  # noqa: E402
    load_m6_evaluation_topology,
)
from project_tools.training_run import write_json_atomic  # noqa: E402


DEFAULT_TOPOLOGY = (
    REPO_ROOT
    / "configs"
    / "evaluation"
    / "robocasa365_m6_atomic18_8server_16client.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须为对象：{path}")
    return payload


def _completed_summary(episode_root: Path) -> tuple[Path, dict[str, Any]] | None:
    candidates = sorted(
        episode_root.glob("*/summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        payload = _read_json(path)
        if payload.get("result") == "pass":
            return path.parent, payload
    return None


def _resumable_run(episode_root: Path) -> Path | None:
    candidates = sorted(
        episode_root.glob("*/**/progress.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    # progress.json lives under <run>/<task>/<episode>; recover the run root.
    for progress_path in candidates:
        episode_dir = progress_path.parent
        run_dir = episode_dir.parents[1]
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file() or _read_json(summary_path).get("result") != "pass":
            return run_dir
    return None


def _episode_config(
    topology: dict[str, Any], *, task: str, seed: int, frontend_port: int
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": f"robocasa365_m6_{task}_seed{seed}",
        "scope": "atomic_only",
        "policy": "xwam_broker_named_12d",
        "protocol": "xwam.robocasa365.atomic.v1",
        "task_manifest": topology["task_manifest"],
        "panda_omron_schema": topology["panda_omron_schema"],
        "task": task,
        "episodes": 1,
        "seed_start": int(seed),
        "scene": dict(topology["scene"]),
        "rollout": {
            "max_steps": int(topology["task_horizons"][task]),
            "action_chunk_length": int(topology["replan_steps"]),
            "minimum_policy_requests": 1,
            "stop_on_success": True,
            "depth_mode": "disabled",
        },
        "recovery": {
            "enabled": True,
            "mode": "deterministic_action_replay",
            "state_replay_atol": 0.00001,
        },
        "network": {
            "broker_address": "127.0.0.1",
            "broker_frontend_port": int(frontend_port),
            "request_timeout_seconds": float(topology["request_timeout_seconds"]),
        },
        "inference": {"cfg": float(topology["cfg"])},
        "video": dict(topology["video"]),
    }


def _run_one_episode(
    *,
    topology: dict[str, Any],
    task: str,
    seed: int,
    frontend_port: int,
    episode_root: Path,
) -> tuple[Path, dict[str, Any]]:
    episode_root.mkdir(parents=True, exist_ok=True)
    completed = _completed_summary(episode_root)
    if completed is not None:
        print(f"[SKIP] completed task={task} seed={seed} run={completed[0]}", flush=True)
        return completed

    command = [
        sys.executable,
        str(REPO_ROOT / "evaluation" / "run_robocasa365_policy_rollout_resumable.py"),
    ]
    resumable = _resumable_run(episode_root)
    if resumable is not None:
        command.extend(["--resume-run-dir", str(resumable)])
        print(f"[RESUME] task={task} seed={seed} run={resumable}", flush=True)
    else:
        config_path = episode_root / "episode_config.json"
        write_json_atomic(
            config_path,
            _episode_config(
                topology,
                task=task,
                seed=seed,
                frontend_port=frontend_port,
            ),
        )
        command.extend(
            ["--config", str(config_path), "--output-root", str(episode_root)]
        )
        print(f"[START] task={task} seed={seed}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    completed = _completed_summary(episode_root)
    if completed is None:
        raise RuntimeError(f"episode 命令成功但缺少 pass summary：{episode_root}")
    return completed


def _write_client_summary(
    *,
    output_root: Path,
    client: dict[str, Any],
    server: dict[str, Any],
    topology: dict[str, Any],
    episode_rows: list[dict[str, Any]],
    expected_episodes: int,
) -> Path:
    task_rows = []
    for task_entry in client["tasks"]:
        task = task_entry["name"]
        rows = [row for row in episode_rows if row["task"] == task]
        successes = sum(bool(row["success"]) for row in rows)
        task_rows.append(
            {
                "task": task,
                "episodes_expected": expected_episodes,
                "episodes_completed": len(rows),
                "successes": successes,
                "success_rate": successes / len(rows) if rows else 0.0,
                "fastwam_reference_success_percent": task_entry[
                    "fastwam_reference_success_percent"
                ],
            }
        )
    complete = all(row["episodes_completed"] == expected_episodes for row in task_rows)
    path = output_root / f"client_{client['client_id']:02d}" / "client_summary.json"
    write_json_atomic(
        path,
        {
            "schema_version": 1,
            "result": "pass" if complete else "running",
            "scope": "atomic_only",
            "topology": topology["name"],
            "client_id": client["client_id"],
            "server_id": client["server_id"],
            "gpu": server["gpu"],
            "model_seed": topology["model_seed"],
            "seed_start": topology["seed_start"],
            "episodes_per_task": expected_episodes,
            "replan_steps": topology["replan_steps"],
            "action_denoise_steps": topology["action_denoise_steps"],
            "tasks": task_rows,
            "episodes": episode_rows,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", default=str(DEFAULT_TOPOLOGY))
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--episodes-per-task", type=int)
    args = parser.parse_args()
    if args.client_id not in range(16):
        parser.error("client id 必须位于0..15")
    if args.episodes_per_task is not None and args.episodes_per_task <= 0:
        parser.error("episodes per task 必须为正")
    return args


def main() -> int:
    args = _parse_args()
    topology = load_m6_evaluation_topology(args.topology, REPO_ROOT)
    client = topology["clients"][args.client_id]
    server = topology["servers"][client["server_id"]]
    episodes_per_task = int(
        args.episodes_per_task or topology["episodes_per_task"]
    )
    output_root = Path(args.output_root).expanduser().resolve()
    episode_rows: list[dict[str, Any]] = []
    for task_entry in client["tasks"]:
        task = task_entry["name"]
        for episode_index in range(episodes_per_task):
            seed = int(topology["seed_start"]) + episode_index
            episode_root = (
                output_root
                / f"client_{args.client_id:02d}"
                / task
                / f"seed_{seed:06d}"
            )
            run_dir, summary = _run_one_episode(
                topology=topology,
                task=task,
                seed=seed,
                frontend_port=server["frontend_port"],
                episode_root=episode_root,
            )
            aggregate = summary["aggregate"]
            episode_rows.append(
                {
                    "task": task,
                    "episode_index": episode_index,
                    "seed": seed,
                    "success": int(aggregate["successes"]) == 1,
                    "steps": int(aggregate["total_steps"]),
                    "run_dir": str(run_dir),
                    "summary": str(run_dir / "summary.json"),
                }
            )
            path = _write_client_summary(
                output_root=output_root,
                client=client,
                server=server,
                topology=topology,
                episode_rows=episode_rows,
                expected_episodes=episodes_per_task,
            )
            print(
                f"[PROGRESS] client={args.client_id} task={task} "
                f"episode={episode_index + 1}/{episodes_per_task} summary={path}",
                flush=True,
            )
    path = _write_client_summary(
        output_root=output_root,
        client=client,
        server=server,
        topology=topology,
        episode_rows=episode_rows,
        expected_episodes=episodes_per_task,
    )
    print(f"[PASS] client {args.client_id} completed: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
