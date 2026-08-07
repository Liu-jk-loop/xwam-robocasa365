from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from data.robocasa365_contract import DatasetContractError
from data.robocasa365_multitask import (
    BalancedRoundRobinDataset,
    build_m3_multitask_manifest_report,
    load_multitask_dataset_manifest,
    resolve_task_dataset_directory,
)
from project_tools.multitask_training import build_multitask_short_report


TASKS = ("PickPlaceCounterToCabinet", "OpenCabinet", "TurnOnMicrowave")


class _FakeDataset:
    action_num = 8
    action_dim = 12
    proprio_dim = 16

    def __init__(self, name: str, length: int):
        self.name = name
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, object]:
        return {"episode_key": f"episode_{index}", "value": (self.name, index)}


def _metric_line(step: int, task_index: int, supervised: bool) -> str:
    action = 0.8 if supervised else 0.0
    proprio = 0.9 if supervised else 0.0
    total = 0.2 + action + proprio
    supervision = 1.0 if supervised else 0.0
    return (
        f"[METRICS] Step: {step:07d} - speed: 0.007 it/s, "
        f"train/video_loss: 0.2, train/action_loss: {action}, "
        f"train/proprio_loss: {proprio}, "
        f"train/action_proprio_supervision_ratio: {supervision}, "
        f"train/task_index: {float(task_index)}, train/depth_loss: 0.0, "
        f"train/loss: {total}"
    )


class M3MultitaskTest(unittest.TestCase):
    def test_balanced_round_robin_cycles_unequal_task_lengths(self) -> None:
        wrapper = BalancedRoundRobinDataset(
            [_FakeDataset("a", 2), _FakeDataset("b", 1), _FakeDataset("c", 3)],
            task_names=("a", "b", "c"),
            manifest_path="manifest.json",
        )
        self.assertEqual(len(wrapper), 9)
        self.assertEqual(
            [wrapper[index]["task_index"] for index in range(9)],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
        )
        self.assertEqual(wrapper[4]["value"], ("b", 0))
        self.assertEqual(wrapper[-1]["episode_key"], "c:episode_2")
        self.assertEqual(wrapper.provenance()["task_lengths"], [2, 1, 3])

    def test_resolver_requires_explicit_path_when_dates_are_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for date in ("20250819", "20250820"):
                meta = root / "OpenCabinet" / date / "lerobot" / "meta"
                meta.mkdir(parents=True)
                (meta / "info.json").write_text("{}")
            with self.assertRaisesRegex(DatasetContractError, "--task-path"):
                resolve_task_dataset_directory(root, "OpenCabinet")

    def test_manifest_enforces_atomic_scope_and_exact_task_count(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = []
            for task_name in TASKS:
                task_path = root / task_name
                meta = task_path / "lerobot" / "meta"
                meta.mkdir(parents=True)
                (meta / "info.json").write_text("{}")
                entries.append(
                    {"task_name": task_name, "dataset_path": str(task_path)}
                )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "name": "test",
                        "scope": "atomic_only",
                        "sampling": "balanced_round_robin",
                        "tasks": entries,
                        "ok": True,
                    }
                )
            )
            loaded = load_multitask_dataset_manifest(
                manifest_path,
                atomic_task_manifest=repo_root
                / "configs/tasks/robocasa365_atomic_seen.json",
                expected_task_count=3,
            )
            self.assertEqual([entry.task_name for entry in loaded.tasks], list(TASKS))

    def test_manifest_report_rejects_explicit_path_for_unselected_task(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        report = build_m3_multitask_manifest_report(
            dataset_root="/missing",
            task_names=TASKS,
            atomic_task_manifest=repo_root
            / "configs/tasks/robocasa365_atomic_seen.json",
            explicit_paths={"CloseFridge": "/missing/CloseFridge/20250819"},
        )
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("未选择的任务" in error for error in report["errors"]), report
        )

    def test_short_training_audit_accepts_original_sampling_semantics(self) -> None:
        lines = [
            "{'clean_action_ratio': 0.5, 'enable_checkpointing': False}",
            f"Training dataset provenance: {{'task_names': {list(TASKS)!r}}}",
        ]
        lines.extend(
            _metric_line(step, step % 3, supervised=step % 4 != 2)
            for step in range(12)
        )
        lines.extend(
            [
                "`Trainer.fit` stopped: `max_steps=12` reached.",
                "Run result: /tmp/result.json (pass)",
            ]
        )
        report = build_multitask_short_report("\n".join(lines), task_names=TASKS)
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["observed_task_indices"], [0, 1, 2] * 4)
        self.assertTrue(report["supervised_steps"])
        self.assertTrue(report["unsupervised_steps"])

    def test_short_training_audit_accepts_distributed_shuffled_prefix(self) -> None:
        observed = [0, 2, 1, 1, 1, 1, 1, 0, 2, 2, 0, 2]
        lines = [
            "{'clean_action_ratio': 0.5, 'enable_checkpointing': False}",
            f"Training dataset provenance: {{'task_names': {list(TASKS)!r}}}",
        ]
        lines.extend(
            _metric_line(step, task_index, supervised=step % 4 != 2)
            for step, task_index in enumerate(observed)
        )
        lines.extend(
            [
                "`Trainer.fit` stopped: `max_steps=12` reached.",
                "Run result: /tmp/result.json (pass)",
            ]
        )
        report = build_multitask_short_report("\n".join(lines), task_names=TASKS)
        self.assertTrue(report["ok"], report)
        self.assertEqual(
            report["task_counts"],
            {
                "PickPlaceCounterToCabinet": 3,
                "OpenCabinet": 5,
                "TurnOnMicrowave": 4,
            },
        )

    def test_short_training_audit_rejects_missing_task_coverage(self) -> None:
        lines = [
            "{'clean_action_ratio': 0.5, 'enable_checkpointing': False}",
            f"Training dataset provenance: {{'task_names': {list(TASKS)!r}}}",
        ]
        lines.extend(_metric_line(step, 0, supervised=step % 2 == 0) for step in range(12))
        lines.extend(
            [
                "`Trainer.fit` stopped: `max_steps=12` reached.",
                "Run result: /tmp/result.json (pass)",
            ]
        )
        report = build_multitask_short_report("\n".join(lines), task_names=TASKS)
        self.assertFalse(report["ok"])
        self.assertFalse(report["checks"]["all_tasks_observed"])
        self.assertFalse(report["checks"]["task_count_spread_within_tolerance"])

    def test_short_training_audit_rejects_invalid_task_index(self) -> None:
        observed = [0, 1, 2] * 3 + [0, 1, 3]
        lines = [
            "{'clean_action_ratio': 0.5, 'enable_checkpointing': False}",
            f"Training dataset provenance: {{'task_names': {list(TASKS)!r}}}",
        ]
        lines.extend(
            _metric_line(step, task_index, supervised=step % 2 == 0)
            for step, task_index in enumerate(observed)
        )
        lines.extend(
            [
                "`Trainer.fit` stopped: `max_steps=12` reached.",
                "Run result: /tmp/result.json (pass)",
            ]
        )
        report = build_multitask_short_report("\n".join(lines), task_names=TASKS)
        self.assertFalse(report["ok"])
        self.assertFalse(report["checks"]["valid_task_indices"])


if __name__ == "__main__":
    unittest.main()
