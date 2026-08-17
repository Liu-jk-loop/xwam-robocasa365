#!/usr/bin/env python3
"""Audit Clariden 4xGH200 step-2 checkpoint and step-4 resume evidence."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from project_tools.clariden_training import (  # noqa: E402
    build_clariden_4gpu_resume_report,
)
from project_tools.training_run import write_json_atomic  # noqa: E402


ORCHESTRATION_ONLY_PATHS = {
    "configs/environment/xwam_clariden.json",
    "deployment/clariden/smoke_train_resume_xwam.sbatch",
    "docs/ARCHITECTURE.md",
    "docs/CHANGELOG.md",
    "docs/PROGRESS.md",
    "project_tools/clariden_training.py",
    "scripts/audit_clariden_4gpu_resume.py",
    "scripts/resolve_clariden_initial_checkpoint.py",
    "tests/test_clariden_deployment.py",
    "tests/test_clariden_training.py",
}


def _artifact_arguments(parser: argparse.ArgumentParser, prefix: str) -> None:
    for name in ("metadata", "result", "events", "optimizer-dir", "log"):
        parser.add_argument(f"--{prefix}-{name}", required=True)


def _metadata_commit(path: str) -> str:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    commit = str((payload.get("git") or {}).get("commit") or "")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError(f"metadata中的Git commit无效：{path}")
    return commit


def _orchestration_only_commit_compatibility(
    *, initial_metadata: str, resumed_metadata: str
) -> dict[str, object]:
    initial_commit = _metadata_commit(initial_metadata)
    resumed_commit = _metadata_commit(resumed_metadata)
    process = subprocess.run(
        ["git", "diff", "--name-only", initial_commit, resumed_commit, "--"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    changed_paths = sorted(line for line in process.stdout.splitlines() if line)
    disallowed_paths = sorted(set(changed_paths) - ORCHESTRATION_ONLY_PATHS)
    return {
        "ok": bool(changed_paths) and not disallowed_paths,
        "mode": "orchestration_only_commit_delta",
        "initial_commit": initial_commit,
        "resumed_commit": resumed_commit,
        "changed_paths": changed_paths,
        "disallowed_paths": disallowed_paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _artifact_arguments(parser, "initial")
    _artifact_arguments(parser, "resumed")
    parser.add_argument(
        "--allow-orchestration-only-commit-delta",
        action="store_true",
        help=(
            "允许复用旧initial，但两个commit之间只能改动本审计器冻结的编排、"
            "审计、测试和文档文件"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--expect-depth",
        action="store_true",
        help="要求dataset.use_depth=true且每个记录step的depth loss有限并大于0。",
    )
    args = parser.parse_args()
    commit_compatibility = None
    if args.allow_orchestration_only_commit_delta:
        commit_compatibility = _orchestration_only_commit_compatibility(
            initial_metadata=args.initial_metadata,
            resumed_metadata=args.resumed_metadata,
        )
    report = build_clariden_4gpu_resume_report(
        initial={
            "metadata_path": args.initial_metadata,
            "result_path": args.initial_result,
            "events_path": args.initial_events,
            "optimizer_dir": args.initial_optimizer_dir,
            "log_path": args.initial_log,
        },
        resumed={
            "metadata_path": args.resumed_metadata,
            "result_path": args.resumed_result,
            "events_path": args.resumed_events,
            "optimizer_dir": args.resumed_optimizer_dir,
            "log_path": args.resumed_log,
        },
        commit_compatibility=commit_compatibility,
        expect_depth=args.expect_depth,
    )
    output = write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Clariden 4-GPU resume audit: {output.resolve()}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
