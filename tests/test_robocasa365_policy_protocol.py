from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.robocasa365_benchmark import load_policy_smoke_config
from evaluation.robocasa365_policy_server import (
    resolve_model_state_checkpoint,
    validate_checkpoint_task,
)
from evaluation.robocasa365_protocol import (
    PolicyProtocolError,
    decode_message,
    deterministic_inference_seed,
    encode_message,
    make_error_response,
    make_request,
    make_success_response,
    validate_response,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "evaluation" / "robocasa365_close_fridge_m4_policy_smoke.json"


def _request() -> dict:
    return make_request(
        request_id="episode_000-step-000000",
        task="CloseFridge",
        episode_id="episode_000",
        seed=0,
        step_id=0,
        prompt="Close the fridge door.",
        video=np.zeros((3, 256, 256, 3), dtype=np.uint8),
        state=np.arange(16, dtype=np.float32),
        cfg=0.0,
    )


class RoboCasa365PolicyProtocolTest(unittest.TestCase):
    def test_request_wire_roundtrip_preserves_exact_tensors(self) -> None:
        request = _request()
        decoded = decode_message(encode_message(request))
        self.assertEqual(decoded["request_id"], request["request_id"])
        self.assertEqual(decoded["task"], "CloseFridge")
        np.testing.assert_array_equal(decoded["video"], request["video"])
        np.testing.assert_array_equal(decoded["state"], request["state"])

    def test_response_wire_roundtrip_requires_matching_request_and_12d(self) -> None:
        request = _request()
        response = make_success_response(
            request,
            actions=np.zeros((32, 12), dtype=np.float32),
            inference_seed=123,
            inference_seconds=1.5,
            checkpoint="/checkpoint/model_states.pt",
        )
        decoded = decode_message(encode_message(response))
        validated = validate_response(decoded, expected_request=request)
        self.assertEqual(validated["actions"].shape, (32, 12))
        mismatch = dict(request, request_id="different")
        with self.assertRaises(PolicyProtocolError):
            validate_response(decoded, expected_request=mismatch)

    def test_protocol_rejects_wrong_camera_shape_and_server_error(self) -> None:
        with self.assertRaises(PolicyProtocolError):
            make_request(
                request_id="r",
                task="CloseFridge",
                episode_id="e",
                seed=0,
                step_id=0,
                prompt="close",
                video=np.zeros((1, 256, 256, 3), dtype=np.uint8),
                state=np.zeros(16, dtype=np.float32),
                cfg=0.0,
            )
        error = decode_message(
            encode_message(
                make_error_response(
                    _request(),
                    error_type="RuntimeError",
                    error_message="inference failed",
                )
            )
        )
        with self.assertRaisesRegex(PolicyProtocolError, "inference failed"):
            validate_response(error, expected_request=_request())

    def test_inference_seed_is_deterministic_and_step_specific(self) -> None:
        first = deterministic_inference_seed("CloseFridge", "episode_000", 0)
        self.assertEqual(first, deterministic_inference_seed("CloseFridge", "episode_000", 0))
        self.assertNotEqual(first, deterministic_inference_seed("CloseFridge", "episode_000", 4))
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 2**32)

    def test_policy_smoke_config_is_one_request_atomic_gate(self) -> None:
        config = load_policy_smoke_config(CONFIG_PATH, REPO_ROOT)
        self.assertEqual(config["scope"], "atomic_only")
        self.assertEqual(config["task"], "CloseFridge")
        self.assertEqual(config["official_horizon"], 900)
        self.assertEqual(config["rollout"]["max_steps"], 4)
        self.assertEqual(config["rollout"]["action_chunk_length"], 4)
        self.assertEqual(config["rollout"]["minimum_policy_requests"], 1)
        self.assertEqual(config["protocol"], "xwam.robocasa365.atomic.v1")

    def test_policy_config_rejects_composite_or_oversized_chunk(self) -> None:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.json"
            payload["scope"] = "composite"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_policy_smoke_config(path, REPO_ROOT)
            payload["scope"] = "atomic_only"
            payload["rollout"]["action_chunk_length"] = 5
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_policy_smoke_config(path, REPO_ROOT)

    def test_checkpoint_resolver_accepts_deepspeed_directory_and_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment = Path(temp_dir) / "experiment"
            checkpoint = experiment / "checkpoints" / "last.ckpt" / "checkpoint"
            checkpoint.mkdir(parents=True)
            model_state = checkpoint / "mp_rank_00_model_states.pt"
            model_state.write_bytes(b"checkpoint")
            self.assertEqual(resolve_model_state_checkpoint(experiment, None), model_state.resolve())
            self.assertEqual(
                resolve_model_state_checkpoint(experiment, model_state), model_state.resolve()
            )

    def test_server_rejects_task_different_from_single_task_checkpoint(self) -> None:
        validate_checkpoint_task("CloseFridge", "CloseFridge")
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            validate_checkpoint_task("OpenCabinet", "CloseFridge")

    def test_new_cli_help_is_dependency_light(self) -> None:
        scripts = (
            "run_robocasa365_policy_broker.py",
            "robocasa365_policy_server.py",
            "run_robocasa365_policy_rollout.py",
            "run_robocasa365_policy_rollout_resumable.py",
        )
        for script in scripts:
            result = subprocess.run(
                [sys.executable, str(REPO_ROOT / "evaluation" / script), "--help"],
                cwd=REPO_ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 0, f"{script}: {result.stderr}")

        audit = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "audit_robocasa365_policy_rollout.py"),
                "--help",
            ],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(audit.returncode, 0, audit.stderr)


if __name__ == "__main__":
    unittest.main()
