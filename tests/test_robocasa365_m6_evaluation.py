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
        self.assertEqual(topology["max_steps_per_episode"], 1000)
        self.assertEqual(topology["scene"]["split"], "target")
        self.assertIsNone(topology["scene"]["layout_id"])
        self.assertIsNone(topology["scene"]["style_id"])
        self.assertEqual(topology["video"]["stride"], 1)
        self.assertEqual(topology["video"]["fps"], 20)
        self.assertFalse(topology["video"]["durable_frames"])
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
                for task in client["tasks"]:
                    path = root / task["name"] / "result.json"
                    path.parent.mkdir(parents=True)
                    path.write_text(
                        json.dumps(
                            {
                                "result": "pass",
                                "task": task["name"],
                                "client_id": client_id,
                                "server_id": client["server_id"],
                                "split": "target",
                                "model_seed": 42,
                                "seed_start": 42,
                                "episodes_expected": 1,
                                "episodes_completed": 1,
                                "n_success": 1,
                                "success_rate": 1.0,
                                "mean_inference_time_s": 1.0,
                                "max_steps": 1000,
                                "replan_steps": 20,
                                "action_denoise_steps": 10,
                                "video_fps": 20,
                                "episodes": [{"seed": 42, "success": True}],
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
            self.assertEqual(report["overall"]["micro_success_rate"], 1.0)
            (root / "TurnOnSinkFaucet/result.json").unlink()
            failed = aggregate(
                topology_path=TOPOLOGY,
                output_root=root,
                episodes_per_task=1,
            )
            self.assertFalse(failed["ok"])
            self.assertTrue(any("task result" in error for error in failed["errors"]))

    def test_new_entrypoints_have_dependency_light_help(self) -> None:
        scripts = (
            REPO_ROOT / "evaluation/run_robocasa365_m6_client.py",
            REPO_ROOT / "evaluation/launch_robocasa365_m6_policy_pool.py",
            REPO_ROOT / "evaluation/launch_robocasa365_m6_client_pool.py",
            REPO_ROOT / "scripts/aggregate_robocasa365_m6_evaluation.py",
            REPO_ROOT / "scripts/probe_robocasa365_eval_runtime.py",
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

    def test_formal_client_is_lean_and_does_not_reenter_m43(self) -> None:
        client = (REPO_ROOT / "evaluation/run_robocasa365_m6_client.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("run_robocasa365_policy_rollout_resumable.py", client)
        self.assertNotIn("progress.json", client)
        self.assertIn("writer.append_data", client)
        self.assertNotIn("imageio.imwrite", client)
        self.assertIn("context.socket(zmq.DEALER)", client)
        self.assertNotIn("context.socket(zmq.REQ)", client)

        policy_pool = (
            REPO_ROOT / "evaluation/launch_robocasa365_m6_policy_pool.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"--disable-request-journal"', policy_pool)
        self.assertIn('"--compact-report"', policy_pool)


if __name__ == "__main__":
    unittest.main()
