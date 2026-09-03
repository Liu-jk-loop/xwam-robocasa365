#!/usr/bin/env python3
"""逐任务生成可恢复PointMap缓存，并发布Atomic多任务索引。"""

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

from data.robocasa365_contract import load_task_manifest  # noqa: E402
from project_tools.robocasa365_depth_encoding import file_sha256  # noqa: E402
from project_tools.robocasa365_pointmap_cache import (  # noqa: E402
    POINTMAP_RENDER_POLICY,
)
from project_tools.training_run import collect_git_state, write_json_atomic  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON顶层必须为对象：{path}")
    return payload


def _task_records(
    training_manifest: dict[str, Any], expected_names: list[str]
) -> list[dict[str, Any]]:
    if (
        training_manifest.get("ok") is not True
        or training_manifest.get("result") != "pass"
        or training_manifest.get("scope") != "atomic_only"
        or training_manifest.get("split") != "pretrain"
    ):
        raise ValueError("Atomic训练manifest尚未通过atomic_only/pretrain审计")
    records = training_manifest.get("tasks")
    if not isinstance(records, list):
        raise ValueError("Atomic训练manifest缺少tasks")
    names = [str(item.get("task_name")) for item in records]
    if names != expected_names:
        raise ValueError(
            f"PointMap任务顺序与训练manifest不一致：{names} != {expected_names}"
        )
    for item in records:
        if int(item.get("episodes", 0)) <= 0:
            raise ValueError(f"任务{item.get('task_name')}缺少有效episode计数")
        dataset_path = item.get("dataset_path")
        if not isinstance(dataset_path, str) or not dataset_path:
            raise ValueError(f"任务{item.get('task_name')}缺少精确dataset_path")
        if str(item.get("task_name")) not in Path(dataset_path).parts:
            raise ValueError(
                f"任务{item.get('task_name')}的dataset_path不包含任务名："
                f"{dataset_path}"
            )
    return records


def _build_index(
    *,
    task_names: list[str],
    artifact_root: Path,
    cache_root: Path,
    manifest_output: Path,
    audit_output: Path,
    task_manifest_path: Path,
    training_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    task_entries: list[dict[str, Any]] = []
    contract_digests: set[str] = set()
    errors: list[str] = []
    for task_name in task_names:
        task_manifest = artifact_root / "tasks" / f"{task_name}_manifest.json"
        task_audit = artifact_root / "tasks" / f"{task_name}_audit.json"
        try:
            manifest = _read_json(task_manifest)
            audit = _read_json(task_audit)
            checks = {
                "manifest_pass": manifest.get("ok") is True
                and manifest.get("result") == "pass",
                "audit_pass": audit.get("ok") is True
                and audit.get("result") == "pass",
                "atomic_only": manifest.get("scope")
                == audit.get("scope")
                == "atomic_only",
                "task_name": manifest.get("task_name")
                == audit.get("task_name")
                == task_name,
                "cache_root": Path(str(manifest.get("cache_root", "")))
                .expanduser()
                .resolve()
                == cache_root
                and Path(str(audit.get("cache_root", "")))
                .expanduser()
                .resolve()
                == cache_root,
                "manifest_binding": Path(str(audit.get("manifest_path", "")))
                .expanduser()
                .resolve()
                == task_manifest.resolve(),
                "render_policy": manifest.get("render_policy")
                == audit.get("render_policy")
                == POINTMAP_RENDER_POLICY,
                "audit_checks": isinstance(audit.get("checks"), dict)
                and bool(audit["checks"])
                and all(audit["checks"].values()),
            }
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                raise ValueError(f"checks={failed}")
            contract_digests.add(str(manifest.get("contract_sha256")))
            task_entries.append(
                {
                    "task_name": task_name,
                    "episode_count": int(manifest["episode_count"]),
                    "array_count": int(manifest["array_count"]),
                    "total_source_frames": int(audit["total_source_frames"]),
                    "total_cache_bytes": int(audit["total_cache_bytes"]),
                    "manifest_path": str(task_manifest.resolve()),
                    "manifest_sha256": file_sha256(task_manifest),
                    "audit_path": str(task_audit.resolve()),
                    "audit_sha256": file_sha256(task_audit),
                }
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{task_name}: {type(exc).__name__}: {exc}")
    if len(contract_digests) != 1:
        errors.append(f"PointMap数值合同摘要不唯一：{sorted(contract_digests)}")
    checks = {
        "atomic_only": True,
        "task_count": len(task_entries) == len(task_names),
        "task_order": [item["task_name"] for item in task_entries] == task_names,
        "single_contract": len(contract_digests) == 1,
        "all_task_artifacts_pass": not errors,
    }
    ok = all(checks.values()) and not errors
    created_at = datetime.now(timezone.utc).isoformat()
    index = {
        "schema_version": 1,
        "stage": "PointMap-P5-Atomic-index",
        "scope": "atomic_only",
        "task_count": len(task_entries),
        "task_names": task_names,
        "cache_root": str(cache_root),
        "contract_sha256": next(iter(contract_digests), None),
        "render_policy": POINTMAP_RENDER_POLICY,
        "task_manifest": str(task_manifest_path),
        "task_manifest_sha256": file_sha256(task_manifest_path),
        "training_manifest": str(training_manifest_path),
        "training_manifest_sha256": file_sha256(training_manifest_path),
        "episode_count": sum(item["episode_count"] for item in task_entries),
        "array_count": sum(item["array_count"] for item in task_entries),
        "total_source_frames": sum(
            item["total_source_frames"] for item in task_entries
        ),
        "total_cache_bytes": sum(item["total_cache_bytes"] for item in task_entries),
        "tasks": task_entries,
        "git": collect_git_state(REPO_ROOT),
        "errors": errors,
        "ok": ok,
        "result": "pass" if ok else "fail",
        "generated_at": created_at,
    }
    write_json_atomic(manifest_output, index)
    audit_tasks = [dict(item) for item in task_entries]
    audit = {
        "schema_version": 1,
        "stage": "PointMap-P5-Atomic-index-audit",
        "scope": "atomic_only",
        "task_count": len(audit_tasks),
        "task_names": task_names,
        "cache_root": str(cache_root),
        "manifest_path": str(manifest_output.resolve()),
        "manifest_sha256": file_sha256(manifest_output),
        "contract_sha256": index["contract_sha256"],
        "render_policy": POINTMAP_RENDER_POLICY,
        "tasks": audit_tasks,
        "checks": checks,
        "errors": errors,
        "ok": ok,
        "result": "pass" if ok else "fail",
        "audited_at": created_at,
    }
    write_json_atomic(audit_output, audit)
    return index, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--task-manifest", required=True)
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--expected-task-count", type=int, required=True)
    parser.add_argument("--transparent-audit", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--index-only", action="store_true")
    args = parser.parse_args()

    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index必须位于[0, num_shards)")
    if args.generate_only and args.index_only:
        raise ValueError("--generate-only与--index-only互斥")

    task_manifest_path = Path(args.task_manifest).expanduser().resolve()
    training_manifest_path = Path(args.training_manifest).expanduser().resolve()
    task_names = list(load_task_manifest(task_manifest_path).tasks)
    if len(task_names) != args.expected_task_count:
        raise ValueError(
            f"任务数漂移：{len(task_names)} != {args.expected_task_count}"
        )
    training_records = _task_records(
        _read_json(training_manifest_path), task_names
    )
    artifact_root = Path(args.artifact_root).expanduser().resolve()
    task_artifact_root = artifact_root / "tasks"
    task_artifact_root.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_root).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    generator = REPO_ROOT / "scripts/build_and_audit_robocasa365_pointmap_cache.py"
    selected_records = (
        []
        if args.index_only
        else training_records[args.shard_index :: args.num_shards]
    )
    if not args.index_only:
        print(
            f"[PointMap-Atomic] shard={args.shard_index}/{args.num_shards} "
            f"tasks={[item['task_name'] for item in selected_records]}",
            flush=True,
        )
    for position, record in enumerate(selected_records, start=1):
        task_name = str(record["task_name"])
        episodes = int(record["episodes"])
        task_output = task_artifact_root / f"{task_name}_manifest.json"
        task_audit = task_artifact_root / f"{task_name}_audit.json"
        print(
            f"[PointMap-Atomic] shard_task={position}/{len(selected_records)} "
            f"name={task_name} episodes={episodes}",
            flush=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(generator),
                "--dataset-root",
                args.dataset_root,
                "--dataset-path",
                str(record["dataset_path"]),
                "--task-manifest",
                str(task_manifest_path),
                "--task-name",
                task_name,
                "--transparent-audit",
                args.transparent_audit,
                "--cache-root",
                str(cache_root),
                "--source-height",
                "256",
                "--source-width",
                "256",
                "--target-height",
                "256",
                "--target-width",
                "320",
                "--episodes-per-task",
                "0",
                "--expected-episode-count",
                str(episodes),
                "--expected-array-count",
                str(episodes * 3),
                "--progress-interval",
                str(args.progress_interval),
                "--manifest-output",
                str(task_output),
                "--audit-output",
                str(task_audit),
            ],
            check=True,
        )

    if args.generate_only:
        print(
            f"[PASS] PointMap cache shard {args.shard_index}/{args.num_shards} complete"
        )
        return 0

    manifest_output = Path(args.manifest_output).expanduser().resolve()
    audit_output = Path(args.audit_output).expanduser().resolve()
    index, audit = _build_index(
        task_names=task_names,
        artifact_root=artifact_root,
        cache_root=cache_root,
        manifest_output=manifest_output,
        audit_output=audit_output,
        task_manifest_path=task_manifest_path,
        training_manifest_path=training_manifest_path,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"pointmap_cache_manifest={manifest_output}")
    print(f"pointmap_cache_audit={audit_output}")
    if not index["ok"] or not audit["ok"]:
        return 1
    print("[PASS] Full Atomic9 resumable forced-opaque PointMap cache ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
