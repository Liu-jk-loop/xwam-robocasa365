from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from evaluation.robocasa365_m6_topology import (
    load_m6_evaluation_topology,
    tasks_for_server,
)
from evaluation.run_robocasa365_m6_client import (
    _create_environment,
    _summarize_base_diagnostics,
)
from evaluation.robocasa365_policy_server import validate_checkpoint_task
from scripts.aggregate_robocasa365_m6_evaluation import aggregate
from scripts.summarize_robocasa365_base_diagnostics import (
    summarize as summarize_base_diagnostics,
)
from scripts.summarize_robocasa365_single_task_evaluation import (
    summarize as summarize_single_task,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TOPOLOGY = (
    REPO_ROOT
    / "configs/evaluation/robocasa365_m6_atomic18_8server_16client.json"
)


class RoboCasa365M6EvaluationTest(unittest.TestCase):
    def test_environment_switches_physical_import_id_to_local_egl_zero(self) -> None:
        captured: dict[str, object] = {}

        class FakeGym:
            @staticmethod
            def make(name: str, **kwargs: object) -> object:
                captured["name"] = name
                captured["kwargs"] = kwargs
                captured["egl"] = os.environ["MUJOCO_EGL_DEVICE_ID"]
                return object()

        with mock.patch.dict(
            sys.modules,
            {"gymnasium": FakeGym, "robocasa": object()},
        ), mock.patch.dict(
            os.environ,
            {
                "CUDA_VISIBLE_DEVICES": "3",
                "MUJOCO_EGL_DEVICE_ID": "3",
                "XWAM_ROBOCASA_IMPORT_EGL_DEVICE": "3",
            },
            clear=False,
        ):
            _create_environment("CloseFridge", "target")
            _create_environment("CloseFridge", "target")

        self.assertEqual(captured["name"], "robocasa/CloseFridge")
        self.assertEqual(captured["kwargs"], {"split": "target"})
        self.assertEqual(captured["egl"], "0")

    def test_single_task_summary_validates_success_rate_and_contract(self) -> None:
        report = summarize_single_task(
            {
                "result": "pass",
                "task": "CloseFridge",
                "episodes_expected": 2,
                "episodes_completed": 2,
                "model_seed": 42,
                "seed_start": 42,
                "max_steps": 1000,
                "replan_steps": 20,
                "action_denoise_steps": 10,
                "video_fps": 20,
                "n_success": 1,
                "success_rate": 0.5,
                "mean_inference_time_s": 1.25,
                "episodes": [
                    {"video": "/tmp/episode_000.mp4"},
                    {"video": "/tmp/episode_001.mp4"},
                ],
            },
            task="CloseFridge",
            episodes=2,
            eval_id="ratio05-step1000",
            checkpoint="/tmp/final-step=1000.ckpt",
        )
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(report["success_percent"], 50.0)
        self.assertEqual(len(report["videos"]), 2)

    def test_base_action_diagnostics_separate_policy_and_execution(self) -> None:
        report = _summarize_base_diagnostics(
            [
                np.asarray([0.5, 0.0, 0.0, 0.0], dtype=np.float32),
                np.zeros(4, dtype=np.float32),
                np.asarray([0.0, -0.25, 0.0, 0.0], dtype=np.float32),
            ],
            [-1.0, 1.0, -1.0],
            [
                np.asarray([0.1, 0.0, 0.0], dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
            ],
        )
        self.assertEqual(report["steps"], 3)
        self.assertEqual(report["base_command_nonzero_steps"], 2)
        self.assertEqual(report["base_position_delta_nonzero_steps"], 1)
        self.assertEqual(report["commanded_but_stationary_steps"], 1)
        self.assertEqual(
            report["control_mode_counts"],
            {"-1": 2, "+1": 1, "other": 0},
        )
        self.assertAlmostEqual(report["base_position_total_displacement"], 0.1)

    def test_base_diagnostic_summary_classifies_near_zero_policy(self) -> None:
        report = summarize_base_diagnostics(
            {
                "task": "NavigateKitchen",
                "episodes": [
                    {
                        "seed": 42,
                        "base_action_diagnostics": {
                            "steps": 100,
                            "base_command_nonzero_fraction": 0.0,
                            "base_command_rms_per_dim": [0.0, 0.0, 0.0, 0.0],
                            "control_mode_counts": {
                                "-1": 100,
                                "+1": 0,
                                "other": 0,
                            },
                            "base_position_delta_nonzero_fraction": 0.0,
                            "base_position_delta_rms_per_dim": [0.0, 0.0, 0.0],
                            "base_position_total_displacement": 0.0,
                            "commanded_but_stationary_steps": 0,
                        },
                    }
                ],
            }
        )
        self.assertEqual(report["verdict"], "policy_base_output_near_zero")

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
            REPO_ROOT / "scripts/summarize_robocasa365_single_task_evaluation.py",
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

    def test_launchers_accept_single_server_and_client_selection(self) -> None:
        policy = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "evaluation/launch_robocasa365_m6_policy_pool.py"),
                "--help",
            ],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        client = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "evaluation/launch_robocasa365_m6_client_pool.py"),
                "--help",
            ],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertIn("--server-id SERVER_IDS", policy.stdout)
        self.assertIn("--single-task-checkpoint", policy.stdout)
        self.assertIn("--cuda-device CUDA_DEVICE", policy.stdout)
        self.assertIn("--client-id CLIENT_IDS", client.stdout)
        self.assertIn("--cuda-visible-devices CUDA_VISIBLE_DEVICES", client.stdout)

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
