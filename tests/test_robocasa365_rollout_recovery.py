from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.robocasa365_benchmark import BenchmarkContractError, load_policy_full_config
from evaluation.robocasa365_rollout_recovery import (
    append_executed_step,
    append_policy_request,
    compare_replayed_state,
    create_rollout_progress,
    pending_action,
    validate_rollout_progress,
)
from scripts.audit_robocasa365_policy_rollout import audit_rollout


REPO_ROOT = Path(__file__).resolve().parents[1]
FULL_CONFIG = (
    REPO_ROOT / "configs" / "evaluation" / "robocasa365_close_fridge_m4_full.json"
)


class RoboCasa365RolloutRecoveryTest(unittest.TestCase):
    def test_full_config_uses_official_horizon_and_recovery(self) -> None:
        config = load_policy_full_config(FULL_CONFIG, REPO_ROOT)
        self.assertEqual(config["rollout"]["max_steps"], 900)
        self.assertEqual(config["official_horizon"], 900)
        self.assertEqual(config["rollout"]["action_chunk_length"], 4)
        self.assertEqual(config["recovery"]["mode"], "deterministic_action_replay")
        self.assertTrue(config["video"]["durable_frames"])

    def test_full_config_rejects_short_horizon_or_disabled_recovery(self) -> None:
        payload = json.loads(FULL_CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.json"
            payload["rollout"]["max_steps"] = 899
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(BenchmarkContractError):
                load_policy_full_config(path, REPO_ROOT)
            payload["rollout"]["max_steps"] = 900
            payload["recovery"]["enabled"] = False
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(BenchmarkContractError):
                load_policy_full_config(path, REPO_ROOT)

    def test_progress_persists_request_before_each_action(self) -> None:
        progress = create_rollout_progress(
            run_id="run",
            task="CloseFridge",
            episode_id="episode_000_seed_000000",
            seed=0,
            max_steps=900,
            action_chunk_length=4,
            environment={"layout_id": 1, "style_id": 1, "language": "close"},
            initial_state=np.zeros(16, dtype=np.float32),
        )
        actions = np.arange(48, dtype=np.float32).reshape(4, 12) / 100.0
        append_policy_request(
            progress,
            request_id="request-0",
            step_id=0,
            actions_to_execute=actions,
            returned_action_shape=[32, 12],
            inference_seed=1,
            inference_seconds=2.0,
            round_trip_seconds=2.1,
            checkpoint="/checkpoint.pt",
            returned_action_min=np.zeros(12),
            returned_action_max=np.ones(12),
        )
        request_index, offset, action = pending_action(progress)
        self.assertEqual((request_index, offset), (0, 0))
        np.testing.assert_array_equal(action, actions[0])
        append_executed_step(
            progress,
            request_index=request_index,
            chunk_offset=offset,
            action=action,
            state_after=np.ones(16, dtype=np.float32),
            success=False,
            terminated=False,
            truncated=False,
        )
        self.assertEqual(progress["steps"], 1)
        self.assertEqual(progress["request_records"][0]["executed_action_steps"], 1)
        validate_rollout_progress(progress)
        restored = json.loads(json.dumps(progress))
        validate_rollout_progress(restored)
        _, restored_offset, restored_action = pending_action(restored)
        self.assertEqual(restored_offset, 1)
        np.testing.assert_array_equal(restored_action, actions[1])

    def test_replay_state_drift_is_blocking(self) -> None:
        self.assertAlmostEqual(
            compare_replayed_state(
                np.zeros(16), np.full(16, 1e-6), atol=1e-5, label="ok"
            ),
            1e-6,
            places=10,
        )
        with self.assertRaises(BenchmarkContractError):
            compare_replayed_state(
                np.zeros(16), np.full(16, 1e-3), atol=1e-5, label="drift"
            )

    def test_audit_accepts_complete_900_step_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            episode_dir = run_dir / "CloseFridge" / "episode_000_seed_000000"
            frame_dir = episode_dir / "video_frames"
            frame_dir.mkdir(parents=True)
            config = json.loads(FULL_CONFIG.read_text(encoding="utf-8"))
            (run_dir / "requested_config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            metadata = {
                "run_id": "run",
                "git": {"commit": "abc", "branch": "dev/atomic-robocasa365", "dirty": False},
                "resolved_config": {
                    "scope": "atomic_only",
                    "protocol": "xwam.robocasa365.atomic.v1",
                },
            }
            (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            request_records = []
            step_records = []
            journal = []
            checkpoint = "/checkpoint/model_states.pt"
            action = [0.1, 0.0, 0.0, 0.0, -1.0] + [0.0] * 7
            state = [0.0] * 16
            for request_index in range(225):
                step_id = request_index * 4
                request_id = f"episode_000_seed_000000-step-{step_id:06d}"
                request_records.append(
                    {
                        "request_id": request_id,
                        "step_id": step_id,
                        "returned_action_shape": [32, 12],
                        "returned_action_steps": 32,
                        "planned_action_steps": 4,
                        "executed_action_steps": 4,
                        "actions_to_execute": [action] * 4,
                        "inference_seed": request_index,
                        "inference_seconds": 1.0,
                        "round_trip_seconds": 1.1,
                        "checkpoint": checkpoint,
                        "returned_action_min": action,
                        "returned_action_max": action,
                    }
                )
                journal.append(
                    {
                        "request_id": request_id,
                        "result": "pass",
                        "task": "CloseFridge",
                        "episode_id": "episode_000_seed_000000",
                        "step_id": step_id,
                        "inference_seed": request_index,
                        "action_shape": [32, 12],
                        "checkpoint": checkpoint,
                        "git_commit": "abc",
                    }
                )
                for offset in range(4):
                    step_records.append(
                        {
                            "step_id": step_id + offset,
                            "request_index": request_index,
                            "chunk_offset": offset,
                            "action": action,
                            "state_after": state,
                            "success": False,
                            "terminated": False,
                            "truncated": False,
                        }
                    )
            progress = {
                "schema_version": 1,
                "recovery_mode": "deterministic_action_replay",
                "result": "pass",
                "protocol": "xwam.robocasa365.atomic.v1",
                "scope": "atomic_only",
                "run_id": "run",
                "task": "CloseFridge",
                "episode_id": "episode_000_seed_000000",
                "seed": 0,
                "max_steps": 900,
                "action_chunk_length": 4,
                "environment": {},
                "initial_state": state,
                "steps": 900,
                "success": False,
                "terminated": False,
                "truncated": False,
                "end_reason": "max_steps",
                "request_records": request_records,
                "step_records": step_records,
            }
            (episode_dir / "progress.json").write_text(
                json.dumps(progress), encoding="utf-8"
            )
            episode = {
                "result": "pass",
                "scope": "atomic_only",
                "protocol": "xwam.robocasa365.atomic.v1",
                "policy_requests": 225,
                "recovery": {
                    "resumed": True,
                    "replayed_steps": 8,
                    "replay_max_state_error": 0.0,
                },
                "observation_diagnostics": {"state_dimension": 16},
                "action_diagnostics": {
                    "action_dimension": 12,
                    "base_nonzero_steps": 900,
                },
                "video": {
                    "path": str(episode_dir / "rollout.mp4"),
                    "frame_cache": str(frame_dir),
                    "frame_count": 46,
                },
            }
            (episode_dir / "episode.json").write_text(
                json.dumps(episode), encoding="utf-8"
            )
            summary = {
                "run_id": "run",
                "result": "pass",
                "scope": "atomic_only",
                "protocol": "xwam.robocasa365.atomic.v1",
                "policy_requests": 225,
                "aggregate": {
                    "episodes_expected": 1,
                    "episodes_completed": 1,
                    "episodes_failed": 0,
                    "episodes_missing": 0,
                    "total_steps": 900,
                },
            }
            (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            for step in range(0, 901, 20):
                (frame_dir / f"frame_{step:06d}.png").write_bytes(b"png")
            (episode_dir / "rollout.mp4").write_bytes(b"video")
            journal_path = run_dir / "server.jsonl"
            journal_path.write_text(
                "\n".join(json.dumps(record) for record in journal) + "\n",
                encoding="utf-8",
            )
            report = audit_rollout(
                run_dir=run_dir,
                server_request_journal=journal_path,
                repo_root=REPO_ROOT,
            )
            self.assertTrue(report["ok"], report["errors"])
            self.assertEqual(report["steps"], 900)
            self.assertEqual(report["policy_requests"], 225)


if __name__ == "__main__":
    unittest.main()
