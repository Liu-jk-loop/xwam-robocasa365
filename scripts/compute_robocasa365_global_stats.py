#!/usr/bin/env python3
"""从已审计的 Atomic-Seen 18 manifest 精确计算跨任务 q01/q99。"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.robocasa365_index import episode_data_path, load_episode_records  # noqa: E402
from data.robocasa365_multitask import (  # noqa: E402
    load_multitask_dataset_manifest,
)
from project_tools.training_run import write_json_atomic  # noqa: E402


def _matrix(table, key: str, dimension: int, path: Path) -> np.ndarray:
    if key not in table.column_names:
        raise ValueError(f"Parquet 缺少 {key!r}：{path}")
    values = np.asarray(table[key].combine_chunks().to_pylist(), dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != dimension:
        raise ValueError(f"{key} 应为 [T,{dimension}]，实际为 {values.shape}：{path}")
    if not np.isfinite(values).all():
        raise ValueError(f"{key} 包含 NaN/Inf：{path}")
    return values


def _stats(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "q01": np.quantile(values, 0.01, axis=0).astype(np.float64).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).astype(np.float64).tolist(),
        "min": np.min(values, axis=0).astype(np.float64).tolist(),
        "max": np.max(values, axis=0).astype(np.float64).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--task-manifest",
        default="configs/tasks/robocasa365_atomic_seen.json",
    )
    parser.add_argument(
        "--output",
        default="logs/cluster/robocasa365_m6_atomic_seen18_global_stats.json",
    )
    parser.add_argument("--work-dir", help="可选临时 memmap 父目录。")
    args = parser.parse_args()

    manifest = load_multitask_dataset_manifest(
        args.manifest,
        atomic_task_manifest=args.task_manifest,
        expected_task_count=18,
    )
    if manifest.sampling != "natural_proportional":
        raise ValueError("M6 global stats 只接受 natural_proportional manifest")
    if manifest.manifest_digest is None:
        raise ValueError("M6 manifest 缺少 manifest_digest")

    episode_sets = []
    total_rows = 0
    for entry in manifest.tasks:
        root, info, episodes = load_episode_records(entry.dataset_path)
        episode_sets.append((entry, root, info, episodes))
        total_rows += sum(episode.length for episode in episodes)
    if total_rows <= 0:
        raise ValueError("manifest 中没有可用于统计的 frame")

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    work_parent = (
        Path(args.work_dir).expanduser().resolve() if args.work_dir else output.parent
    )
    work_parent.mkdir(parents=True, exist_ok=True)

    import pyarrow.parquet as pq

    with tempfile.TemporaryDirectory(prefix="xwam-stats-", dir=work_parent) as tmp:
        tmp_root = Path(tmp)
        states = np.memmap(
            tmp_root / "states.f32", mode="w+", dtype=np.float32, shape=(total_rows, 16)
        )
        actions = np.memmap(
            tmp_root / "actions.f32",
            mode="w+",
            dtype=np.float32,
            shape=(total_rows, 12),
        )
        offset = 0
        task_rows: dict[str, int] = {}
        for entry, root, info, episodes in episode_sets:
            before = offset
            for episode in episodes:
                path = episode_data_path(root, info, episode.episode_index)
                table = pq.read_table(path)
                if table.num_rows != episode.length:
                    raise ValueError(
                        f"episode长度漂移：metadata={episode.length}, parquet={table.num_rows}: {path}"
                    )
                next_offset = offset + episode.length
                states[offset:next_offset] = _matrix(
                    table, "observation.state", 16, path
                )
                actions[offset:next_offset] = _matrix(table, "action", 12, path)
                offset = next_offset
            task_rows[entry.task_name] = offset - before
        states.flush()
        actions.flush()
        if offset != total_rows:
            raise RuntimeError(
                f"统计行数不一致：written={offset}, expected={total_rows}"
            )
        payload = {
            "schema_version": 1,
            "scope": "atomic_only",
            "split": "pretrain",
            "task_count": len(manifest.tasks),
            "manifest_path": manifest.source_path,
            "manifest_digest": manifest.manifest_digest,
            "total_frames": total_rows,
            "task_frames": task_rows,
            "observation.state": _stats(states),
            "action": _stats(actions),
            "ok": True,
            "result": "pass",
        }
        write_json_atomic(output, payload)

    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"M6 global stats: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
