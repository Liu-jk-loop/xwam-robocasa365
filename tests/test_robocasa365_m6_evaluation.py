from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from evaluation.robocasa365_m6_topology import (
    load_m6_evaluation_topology,
    tasks_for_server,
)
from evaluation.robocasa365_policy_server import validate_checkpoint_task
from scripts.aggregate_robocasa365_m6_evaluation import aggregate


REPO_ROOT = Path(__file__).resolve().parents[1]
TOPOLOGY = (
    REPO_ROOT
    / "configs/evaluation/robocasa365_m6_atomic18_8server_16client.json"
)


class RoboCasa365M6EvaluationTest(unittest.TestCase):
    def test_topology_exactly_matches_requested_assignment(self) -> None:
        topology = load_m6_evaluation_topology(TOPOLOGY, REPO_ROOT)
        self.assertEqual(topology["model_seed"], 42)
        self.assertEqual(topology["seed_start"], 42)
        self.assertEqual(topology["episodes_per_task"], 50)
        self.assertEqual(topology["replan_steps"], 20)
        self.assertEqual(topology["action_denoise_steps"], 10)
        self.assertEqual(topology["video_denoise_steps"], 50)
        self.assertEqual(len(topology["servers"]), 8)
        self.assertEqual(len(topology["clients"]), 16)
        self.assertEqual(
            [task["name"] for task in topology["clients"][6]["tasks"]],
            ["OpenStandMixerHead", "CloseToasterOvenDoor"],
        )
        self.assertEqual(
            [task["name"] for task in topology["clients"][7]["tasks"]],
            ["SlideDishwasherRack", "TurnOnElectricKettle"],
        )
        self.assertEqual(tasks_for_server(topology, 0), ["OpenCabinet", "NavigateKitchen"])
        self.assertEqual(
            tasks_for_server(topology, 6),
            ["OpenStandMixerHead", "CloseToasterOvenDoor", "CoffeeSetupMug"],
        )
        self.assertEqual(
            [topology["servers"][index]["gpu"] for index in range(8)],
            [0, 1, 2, 3, 0, 1, 2, 3],
        )

    def test_multitask_server_rejects_task_outside_assigned_subset(self) -> None:
        allowed = ["OpenCabinet", "NavigateKitchen"]
        validate_checkpoint_task("OpenCabinet", allowed)
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            validate_checkpoint_task("CloseFridge", allowed)

    def test_aggregate_requires_all_clients_and_tasks(self) -> None:
        topology = load_m6_evaluation_topology(TOPOLOGY, REPO_ROOT)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for client_id, client in topology["clients"].items():
                tasks = []
                for task in client["tasks"]:
                    tasks.append(
                        {
                            "task": task["name"],
                            "episodes_expected": 1,
                            "episodes_completed": 1,
                            "successes": 1,
                            "success_rate": 1.0,
                            "fastwam_reference_success_percent": task[
                                "fastwam_reference_success_percent"
                            ],
                        }
                    )
                path = root / f"client_{client_id:02d}" / "client_summary.json"
                path.parent.mkdir(parents=True)
                path.write_text(
                    json.dumps(
                        {
                            "result": "pass",
                            "client_id": client_id,
                            "server_id": client["server_id"],
                            "model_seed": topology["model_seed"],
                            "seed_start": topology["seed_start"],
                            "tasks": tasks,
                        }
                    ),
                    encoding="utf-8",
                )
            report = aggregate(
                topology_path=TOPOLOGY,
                output_root=root,
                episodes_per_task=1,
                eval_id="unit-eval",
                checkpoint="/tmp/unit-checkpoint.ckpt",
            )
            self.assertTrue(report["ok"], report["errors"])
            self.assertEqual(report["eval_id"], "unit-eval")
            self.assertEqual(
                report["checkpoint"], str(Path("/tmp/unit-checkpoint.ckpt").resolve())
            )
            self.assertEqual(report["overall"]["tasks"], 18)
            self.assertEqual(report["overall"]["episodes"], 18)
            self.assertEqual(report["overall"]["success_rate"], 1.0)
            (root / "client_15/client_summary.json").unlink()
            failed = aggregate(
                topology_path=TOPOLOGY,
                output_root=root,
                episodes_per_task=1,
            )
            self.assertFalse(failed["ok"])
            self.assertTrue(any("client summary" in error for error in failed["errors"]))

    def test_new_entrypoints_have_dependency_light_help(self) -> None:
        scripts = (
            REPO_ROOT / "evaluation/run_robocasa365_m6_client.py",
            REPO_ROOT / "evaluation/launch_robocasa365_m6_policy_pool.py",
            REPO_ROOT / "evaluation/launch_robocasa365_m6_client_pool.py",
            REPO_ROOT / "scripts/aggregate_robocasa365_m6_evaluation.py",
        )
        for script in scripts:
            result = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=REPO_ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 0, f"{script}: {result.stderr}")


if __name__ == "__main__":
    unittest.main()
