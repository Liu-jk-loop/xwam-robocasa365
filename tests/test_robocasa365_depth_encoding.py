from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from project_tools.robocasa365_depth_encoding import (
    DepthEncodingError,
    decoded_frame_audit,
    depth_cache_sidecar_path,
    depth_cache_video_path,
    encode_inverse_metric_depth,
    freeze_global_inverse_depth_encoding,
    sample_inverse_metric_depth,
    validate_task_depth_cache,
    validate_frozen_encoding,
)


def _spec() -> dict[str, object]:
    return freeze_global_inverse_depth_encoding(
        [np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)],
        quantile_low=0.0,
        quantile_high=1.0,
        calibration={"seed": 42},
    )


class RoboCasa365DepthEncodingTests(unittest.TestCase):
    def test_global_encoding_is_nearer_brighter_and_three_channel(self):
        metric = np.array([[1.0, 0.5], [0.25, 2.0]], dtype=np.float32)
        encoded, report = encode_inverse_metric_depth(metric, _spec())
        self.assertEqual(encoded.shape, (2, 2, 3))
        self.assertEqual(encoded.dtype, np.uint8)
        self.assertTrue(np.array_equal(encoded[..., 0], encoded[..., 1]))
        self.assertGreater(encoded[1, 0, 0], encoded[0, 0, 0])
        self.assertEqual(report["invalid_pixels"], 0)

    def test_invalid_depth_maps_to_zero(self):
        metric = np.array([[0.0, np.nan], [-1.0, 1.0]], dtype=np.float32)
        encoded, report = encode_inverse_metric_depth(metric, _spec())
        self.assertTrue(np.all(encoded[:1] == 0))
        self.assertEqual(encoded[1, 0, 0], 0)
        self.assertEqual(report["invalid_pixels"], 3)

    def test_encoding_digest_detects_mutation(self):
        spec = copy.deepcopy(_spec())
        spec["inverse_depth_high_per_m"] = 9.0
        with self.assertRaisesRegex(DepthEncodingError, "digest"):
            validate_frozen_encoding(spec)

    def test_sampling_is_deterministic_for_same_seed(self):
        depth = np.linspace(0.1, 5.0, 100, dtype=np.float32).reshape(10, 10)
        first = sample_inverse_metric_depth(
            depth, pixels_per_frame=10, rng=np.random.default_rng(7)
        )
        second = sample_inverse_metric_depth(
            depth, pixels_per_frame=10, rng=np.random.default_rng(7)
        )
        np.testing.assert_array_equal(first, second)

    def test_decoded_audit_tolerates_small_h264_error(self):
        expected = np.full((4, 4, 3), 100, dtype=np.uint8)
        decoded = expected.copy()
        decoded[..., 1] += 1
        report = decoded_frame_audit(
            decoded, expected, max_mae=1.0, max_channel_delta=2
        )
        self.assertTrue(report["ok"])

    def test_task_cache_contract_binds_all_episode_cameras(self):
        cameras = (
            "observation.images.robot0_agentview_left",
            "observation.images.robot0_agentview_right",
            "observation.images.robot0_eye_in_hand",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_root = root / "cache"
            encoding_path = root / "encoding.json"
            manifest_path = root / "manifest.json"
            spec = _spec()
            encoding_path.write_text(json.dumps(spec), encoding="utf-8")
            camera_reports = []
            for camera in cameras:
                video = depth_cache_video_path(
                    cache_root,
                    task_name="CloseFridge",
                    episode_index=0,
                    camera_key=camera,
                    chunks_size=1000,
                )
                video.parent.mkdir(parents=True, exist_ok=True)
                video.write_bytes(b"fake-video")
                sidecar = depth_cache_sidecar_path(video)
                report = {
                    "task_name": "CloseFridge",
                    "episode_index": 0,
                    "camera_key": camera,
                    "video_path": str(video),
                    "sidecar_path": str(sidecar),
                    "encoding_sha256": spec["encoding_sha256"],
                    "frame_count": 5,
                    "source_identity": {"states": {"sha256": "a" * 64}},
                    "source_rgb_video_sha256": "b" * 64,
                    "ok": True,
                }
                sidecar.write_text(json.dumps(report), encoding="utf-8")
                camera_reports.append(report)
            manifest = {
                "cache_root": str(cache_root.resolve()),
                "encoding_sha256": spec["encoding_sha256"],
                "tasks": [
                    {
                        "task_name": "CloseFridge",
                        "ok": True,
                        "episodes": [
                            {
                                "episode_index": 0,
                                "episode_length": 5,
                                "cameras": camera_reports,
                            }
                        ],
                    }
                ],
                "ok": True,
                "result": "pass",
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            contract = validate_task_depth_cache(
                cache_root=cache_root,
                encoding_path=encoding_path,
                manifest_path=manifest_path,
                task_name="CloseFridge",
                episode_lengths={0: 5},
                camera_keys=cameras,
                chunks_size=1000,
            )
            self.assertEqual(contract["episode_count"], 1)
            self.assertEqual(contract["video_count"], 3)
            self.assertEqual(contract["encoding_sha256"], spec["encoding_sha256"])


if __name__ == "__main__":
    unittest.main()
