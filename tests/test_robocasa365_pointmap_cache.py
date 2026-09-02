from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from project_tools.robocasa365_pointmap import POINTMAP_CONTRACT_NAME
from project_tools.robocasa365_pointmap_cache import (
    POINTMAP_RENDER_POLICY,
    audit_pointmap_cache_array,
    expected_array_shape,
    frozen_pointmap_cache_contract,
    pointmap_cache_path,
    pointmap_sidecar_path,
    validate_task_pointmap_cache,
    validate_manifest_artifact,
    validate_resume_pair,
    validate_transparent_policy_evidence,
)


class RoboCasa365PointMapCacheTests(unittest.TestCase):
    def _contract(self):
        return frozen_pointmap_cache_contract(
            source_height=2,
            source_width=2,
            target_height=2,
            target_width=3,
        )

    def test_contract_freezes_forced_opaque_and_uncompressed_tchw(self):
        contract = self._contract()
        self.assertEqual(contract["pointmap"]["name"], POINTMAP_CONTRACT_NAME)
        self.assertEqual(contract["render_policy"]["name"], POINTMAP_RENDER_POLICY)
        self.assertEqual(contract["storage"]["dtype"], "float16")
        self.assertEqual(contract["storage"]["layout"], "TCHW")
        self.assertEqual(contract["storage"]["compression"], "none")
        self.assertEqual(len(contract["contract_sha256"]), 64)

    def test_p1_requires_conclusive_transparent_policy_evidence(self):
        report = {
            "ok": True,
            "result": "pass",
            "scope": "atomic_only",
            "phase": "pointmap_transparent_surface_audit",
            "git": {"commit": "c" * 40},
            "task": {
                "task_name": "CloseBlenderLid",
                "episodes_passed": 3,
                "target_visible_pixels": 100,
                "changed_target_pixels": 2,
                "conclusion": {
                    "status": "forced_opaque_required",
                    "formal_cache_policy": POINTMAP_RENDER_POLICY,
                },
            },
        }
        evidence = validate_transparent_policy_evidence(
            report, report_sha256="a" * 64
        )
        self.assertEqual(evidence["changed_target_pixels"], 2)
        report["task"]["changed_target_pixels"] = 0
        with self.assertRaisesRegex(RuntimeError, "target_depth_changed"):
            validate_transparent_policy_evidence(report, report_sha256="a" * 64)

    def test_cache_path_is_per_episode_per_camera(self):
        path = pointmap_cache_path(
            "/tmp/cache",
            task_name="CloseFridge",
            episode_index=1001,
            camera_key="observation.images.robot0_eye_in_hand",
            chunks_size=1000,
        )
        self.assertEqual(path.name, "episode_001001.npy")
        self.assertIn("chunk-001", path.parts)
        self.assertEqual(pointmap_sidecar_path(path).suffixes[-2:], [".pointmap", ".json"])

    def test_array_audit_uses_mmap_and_rejects_range_or_dtype(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            good = root / "good.npy"
            np.save(good, np.zeros((2, 3, 2, 3), dtype=np.float16))
            report = audit_pointmap_cache_array(
                good, expected_shape=(2, 3, 2, 3)
            )
            self.assertTrue(report["ok"])
            bad_range = root / "bad-range.npy"
            value = np.zeros((2, 3, 2, 3), dtype=np.float16)
            value[1, 0, 0, 0] = 1.5
            np.save(bad_range, value)
            report = audit_pointmap_cache_array(
                bad_range, expected_shape=(2, 3, 2, 3)
            )
            self.assertFalse(report["ok"])
            self.assertIn("超出", " ".join(report["errors"]))
            bad_dtype = root / "bad-dtype.npy"
            np.save(bad_dtype, value.astype(np.float32))
            self.assertFalse(
                audit_pointmap_cache_array(
                    bad_dtype, expected_shape=(2, 3, 2, 3)
                )["ok"]
            )

    def test_resume_requires_exact_sidecar_and_array_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            array_path = Path(temporary) / "episode_000000.npy"
            np.save(array_path, np.zeros((2, 3, 2, 3), dtype=np.float16))
            contract = self._contract()
            expected = {
                "schema_version": 1,
                "task_name": "CloseFridge",
                "scope": "atomic_only",
                "episode_index": 0,
                "camera_key": "observation.images.robot0_eye_in_hand",
                "camera_name": "robot0_eye_in_hand",
                "contract_name": POINTMAP_CONTRACT_NAME,
                "contract_sha256": contract["contract_sha256"],
                "render_policy": POINTMAP_RENDER_POLICY,
                "source_identity": {"states": {"sha256": "a" * 64}},
                "frame_count": 2,
                "shape": [2, 3, 2, 3],
                "dtype": "float16",
                "layout": "TCHW",
                "array_path": str(array_path.resolve()),
                "sidecar_path": str(pointmap_sidecar_path(array_path.resolve())),
            }
            audit = audit_pointmap_cache_array(
                array_path, expected_shape=(2, 3, 2, 3)
            )
            sidecar = {
                **expected,
                "array_sha256": audit["sha256"],
                "array_audit": audit,
                "ok": True,
            }
            pointmap_sidecar_path(array_path).write_text(
                json.dumps(sidecar), encoding="utf-8"
            )
            loaded, reasons = validate_resume_pair(array_path, expected=expected)
            self.assertEqual(loaded, sidecar)
            self.assertEqual(reasons, [])
            value = np.load(array_path)
            value[0, 0, 0, 0] = 0.5
            np.save(array_path, value)
            loaded, reasons = validate_resume_pair(array_path, expected=expected)
            self.assertIsNone(loaded)
            self.assertTrue(any("sha256" in reason for reason in reasons))

    def test_manifest_artifact_reopens_array_and_matches_sidecar(self):
        with tempfile.TemporaryDirectory() as temporary:
            array_path = Path(temporary) / "episode_000000.npy"
            shape = expected_array_shape(1, target_height=2, target_width=3)
            np.save(array_path, np.zeros(shape, dtype=np.float16))
            contract = self._contract()
            audit = audit_pointmap_cache_array(array_path, expected_shape=shape)
            record = {
                "contract_name": POINTMAP_CONTRACT_NAME,
                "contract_sha256": contract["contract_sha256"],
                "array_path": str(array_path.resolve()),
                "shape": list(shape),
                "array_sha256": audit["sha256"],
                "array_audit": audit,
            }
            pointmap_sidecar_path(array_path).write_text(
                json.dumps(record), encoding="utf-8"
            )
            result = validate_manifest_artifact(
                record,
                expected_contract_sha256=contract["contract_sha256"],
            )
            self.assertTrue(result["ok"], result["errors"])

    def test_training_validator_binds_manifest_audit_sidecar_and_npy_header(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            camera_key = "observation.images.robot0_agentview_left"
            array_path = pointmap_cache_path(
                root,
                task_name="CloseFridge",
                episode_index=0,
                camera_key=camera_key,
                chunks_size=1000,
            )
            array_path.parent.mkdir(parents=True)
            np.save(
                array_path,
                np.zeros((1, 3, 256, 320), dtype=np.float16),
            )
            record = {
                "camera_key": camera_key,
                "array_path": str(array_path.resolve()),
                "shape": [1, 3, 256, 320],
            }
            pointmap_sidecar_path(array_path).write_text(
                json.dumps(record), encoding="utf-8"
            )
            manifest_path = Path(temporary) / "manifest.json"
            audit_path = Path(temporary) / "audit.json"
            contract_sha256 = "1" * 64
            manifest = {
                "ok": True,
                "result": "pass",
                "scope": "atomic_only",
                "task_name": "CloseFridge",
                "cache_root": str(root.resolve()),
                "contract_sha256": contract_sha256,
                "render_policy": POINTMAP_RENDER_POLICY,
                "contract": {"storage": {"shape_per_frame": [3, 256, 320]}},
                "episodes": [
                    {
                        "episode_index": 0,
                        "episode_length": 1,
                        "cameras": [record],
                    }
                ],
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            audit = {
                "ok": True,
                "result": "pass",
                "scope": "atomic_only",
                "task_name": "CloseFridge",
                "cache_root": str(root.resolve()),
                "manifest_path": str(manifest_path.resolve()),
                "contract_sha256": contract_sha256,
                "render_policy": POINTMAP_RENDER_POLICY,
                "checks": {"all_artifacts": True},
            }
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            result = validate_task_pointmap_cache(
                cache_root=root,
                manifest_path=manifest_path,
                audit_path=audit_path,
                task_name="CloseFridge",
                episode_lengths={0: 1},
                camera_keys=(camera_key,),
                chunks_size=1000,
            )
            self.assertEqual(result["episode_count"], 1)
            self.assertEqual(result["array_count"], 1)
            self.assertEqual(
                result["startup_validation"],
                "manifest_audit_sidecar_and_npy_header_no_full_rehash",
            )


if __name__ == "__main__":
    unittest.main()
