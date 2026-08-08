#!/usr/bin/env python3
"""审计 M4.3 完整 horizon client 证据和 policy server 请求日志。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.robocasa365_benchmark import load_policy_full_config  # noqa: E402
from evaluation.robocasa365_rollout_recovery import (  # noqa: E402
    validate_rollout_progress,
)
from project_tools.training_run import write_json_atomic  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 JSON：{path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须为对象：{path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"无法读取 server request journal：{path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"server journal 第{line_number}行不是合法 JSON") from exc
        if not isinstance(record, dict):
            raise ValueError(f"server journal 第{line_number}行顶层不是对象")
        records.append(record)
    return records


def audit_rollout(
    *, run_dir: str | Path, server_request_journal: str | Path, repo_root: str | Path
) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    repo = Path(repo_root).resolve()
    requested = root / "requested_config.json"
    metadata = _read_json(root / "metadata.json")
    summary = _read_json(root / "summary.json")
    config = load_policy_full_config(requested, repo)
    episode_id = f"episode_000_seed_{int(config['seed_start']):06d}"
    episode_dir = root / config["task"] / episode_id
    episode = _read_json(episode_dir / "episode.json")
    progress = _read_json(episode_dir / "progress.json")
    validate_rollout_progress(progress)
    journal_path = Path(server_request_journal).expanduser().resolve()
    journal = _read_jsonl(journal_path)

    checks: dict[str, bool] = {}
    errors: list[str] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks[name] = bool(condition)
        if not condition:
            errors.append(f"{name}: {detail}")

    git = metadata.get("git", {})
    check("metadata_git_clean", git.get("dirty") is False, str(git))
    check(
        "run_identity",
        metadata.get("run_id") == summary.get("run_id") == progress.get("run_id"),
        "metadata/summary/progress run_id 不一致",
    )
    check(
        "atomic_protocol",
        all(
            payload.get("scope") == "atomic_only"
            and payload.get("protocol") == "xwam.robocasa365.atomic.v1"
            for payload in (metadata["resolved_config"], summary, episode, progress)
        ),
        "scope/protocol 漂移",
    )
    check("summary_pass", summary.get("result") == "pass", str(summary.get("result")))
    check("episode_pass", episode.get("result") == "pass", str(episode.get("result")))
    check("progress_pass", progress.get("result") == "pass", str(progress.get("result")))

    steps = int(progress["steps"])
    requests = progress["request_records"]
    expected_requests = math.ceil(steps / int(config["rollout"]["action_chunk_length"]))
    check(
        "full_horizon_or_success",
        bool(progress["success"])
        or (
            steps == int(config["official_horizon"])
            and progress["end_reason"] in {"max_steps", "truncated"}
        ),
        f"steps={steps}, success={progress['success']}, end={progress['end_reason']}",
    )
    check(
        "request_count",
        len(requests)
        == expected_requests
        == int(episode.get("policy_requests", -1))
        == int(summary.get("policy_requests", -1)),
        "progress/expected/episode/summary request count不一致："
        f"{len(requests)}/{expected_requests}/{episode.get('policy_requests')}/"
        f"{summary.get('policy_requests')}",
    )
    aggregate = summary.get("aggregate", {})
    check(
        "aggregate_complete",
        aggregate.get("episodes_expected") == 1
        and aggregate.get("episodes_completed") == 1
        and aggregate.get("episodes_failed") == 0
        and aggregate.get("episodes_missing") == 0
        and aggregate.get("total_steps") == steps,
        str(aggregate),
    )
    checkpoints = {record["checkpoint"] for record in requests}
    check("single_checkpoint", len(checkpoints) == 1, str(sorted(checkpoints)))
    check(
        "request_shapes",
        all(record.get("returned_action_shape") == [32, 12] for record in requests),
        "存在非 [32,12] response",
    )
    observation_diagnostics = episode.get("observation_diagnostics", {})
    action_diagnostics = episode.get("action_diagnostics", {})
    executed_actions = np.asarray(
        [record["action"] for record in progress["step_records"]], dtype=np.float32
    )
    derived_base_nonzero = int(
        np.count_nonzero(np.linalg.norm(executed_actions[:, 0:4], axis=1) > 1e-8)
    )
    check(
        "state_action_dimensions",
        observation_diagnostics.get("state_dimension") == 16
        and action_diagnostics.get("action_dimension") == 12,
        f"state/action={observation_diagnostics.get('state_dimension')}/"
        f"{action_diagnostics.get('action_dimension')}",
    )
    check(
        "base_action_exercised",
        derived_base_nonzero > 0
        and int(action_diagnostics.get("base_nonzero_steps", -1))
        == derived_base_nonzero,
        f"derived/episode={derived_base_nonzero}/"
        f"{action_diagnostics.get('base_nonzero_steps')}",
    )
    check(
        "control_mode_discrete",
        bool(np.isin(executed_actions[:, 4], np.asarray([-1.0, 1.0])).all()),
        "存在非-1/+1 control_mode",
    )
    check(
        "executed_actions_bounded",
        bool(
            np.isfinite(executed_actions).all()
            and (executed_actions >= -1.0 - 1e-6).all()
            and (executed_actions <= 1.0 + 1e-6).all()
        ),
        f"range=[{float(executed_actions.min())},{float(executed_actions.max())}]",
    )
    recovery = episode.get("recovery", {})
    replay_error = float(recovery.get("replay_max_state_error", math.inf))
    check(
        "intentional_resume_verified",
        recovery.get("resumed") is True
        and int(recovery.get("replayed_steps", 0)) >= 8
        and math.isfinite(replay_error)
        and replay_error <= float(config["recovery"]["state_replay_atol"]),
        str(recovery),
    )

    requested_ids = {record["request_id"] for record in requests}
    matching_journal = [
        record for record in journal if record.get("request_id") in requested_ids
    ]
    client_requests = {record["request_id"]: record for record in requests}
    journal_pass_ids: set[str] = set()
    for record in matching_journal:
        client = client_requests[record["request_id"]]
        if (
            record.get("result") == "pass"
            and record.get("task") == config["task"]
            and record.get("episode_id") == progress["episode_id"]
            and int(record.get("step_id", -1)) == int(client["step_id"])
            and int(record.get("inference_seed", -1)) == int(client["inference_seed"])
            and record.get("action_shape") == [32, 12]
            and record.get("checkpoint") == client["checkpoint"]
            and record.get("git_commit") == git.get("commit")
        ):
            journal_pass_ids.add(record["request_id"])
    journal_failures = [
        record for record in matching_journal if record.get("result") != "pass"
    ]
    check(
        "server_journal_complete",
        journal_pass_ids == requested_ids,
        f"missing={sorted(requested_ids - journal_pass_ids)[:10]}",
    )
    check(
        "server_journal_no_failures",
        not journal_failures,
        f"failures={journal_failures[:3]}",
    )

    video = episode.get("video", {})
    video_path = Path(str(video.get("path", "")))
    frame_dir = Path(str(video.get("frame_cache", "")))
    frames = sorted(frame_dir.glob("frame_*.png")) if frame_dir.is_dir() else []
    expected_frame_steps = {0, steps}
    expected_frame_steps.update(range(int(config["video"]["stride"]), steps + 1, int(config["video"]["stride"])))
    actual_frame_steps: set[int] = set()
    for frame in frames:
        try:
            actual_frame_steps.add(int(frame.stem.removeprefix("frame_")))
        except ValueError:
            errors.append(f"video_frame_names: 非法帧名 {frame.name}")
            checks["video_frame_names"] = False
    checks.setdefault("video_frame_names", True)
    check(
        "durable_frames_complete",
        actual_frame_steps == expected_frame_steps
        and int(video.get("frame_count", -1)) == len(frames),
        f"expected={sorted(expected_frame_steps)}, actual={sorted(actual_frame_steps)}",
    )
    check(
        "video_nonempty",
        video_path.is_file() and video_path.stat().st_size > 0,
        str(video_path),
    )

    return {
        "schema_version": 1,
        "result": "pass" if not errors else "fail",
        "ok": not errors,
        "run_dir": str(root),
        "server_request_journal": str(journal_path),
        "git": git,
        "task": config["task"],
        "official_horizon": config["official_horizon"],
        "steps": steps,
        "success": bool(progress["success"]),
        "policy_requests": len(requests),
        "checkpoint": next(iter(checkpoints)) if len(checkpoints) == 1 else None,
        "video_frames": len(frames),
        "checks": checks,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--server-request-journal", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        report = audit_rollout(
            run_dir=args.run_dir,
            server_request_journal=args.server_request_journal,
            repo_root=REPO_ROOT,
        )
    except Exception as exc:
        report = {
            "schema_version": 1,
            "result": "fail",
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
