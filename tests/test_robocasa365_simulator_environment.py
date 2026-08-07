from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from project_tools.robocasa365_simulator_environment import (
    PROBE_PREFIX,
    _registry_probe_code,
    _runtime_probe_code,
    classify_reuse,
    evaluate_packages,
    load_manifest,
    parse_probe_payload,
    version_satisfies,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "configs" / "environment" / "robocasa365_simulator.json"


class RoboCasa365SimulatorEnvironmentTest(unittest.TestCase):
    def test_manifest_freezes_robocasa365_atomic_contract(self) -> None:
        manifest = load_manifest(MANIFEST, REPO_ROOT)
        self.assertEqual(manifest["scope"], "atomic_only")
        self.assertEqual(len(manifest["atomic_seen_tasks"]), 18)
        self.assertEqual(manifest["runtime_smoke"]["task"], "CloseFridge")
        self.assertEqual(sum(manifest["runtime_smoke"]["observation_components"].values()), 16)
        self.assertEqual(sum(manifest["runtime_smoke"]["action_components"].values()), 12)
        self.assertEqual(len(manifest["runtime_smoke"]["camera_keys"]), 3)

    def test_exact_versions_reject_old_robocasa(self) -> None:
        self.assertTrue(version_satisfies("1.0.1", {"exact": "1.0.1"})[0])
        self.assertFalse(version_satisfies("0.2.0", {"exact": "1.0.1"})[0])

    def test_package_evaluation_reports_version_and_missing_package(self) -> None:
        versions = {"robocasa": "0.2.0", "robosuite": None}
        _, errors = evaluate_packages(
            [
                {"distribution": "robocasa", "exact": "1.0.1"},
                {"distribution": "robosuite", "min": "1.5.2"},
            ],
            version_lookup=versions.get,
        )
        self.assertTrue(any("robocasa" in error for error in errors))
        self.assertTrue(any("robosuite" in error for error in errors))

    def test_probe_payload_ignores_verbose_output(self) -> None:
        result = {
            "stdout": "Creating CloseFridge\n" + PROBE_PREFIX + json.dumps({"ok": True}) + "\n"
        }
        self.assertEqual(parse_probe_payload(result), {"ok": True})

    def test_generated_probe_programs_compile(self) -> None:
        manifest = load_manifest(MANIFEST, REPO_ROOT)
        compile(_registry_probe_code(manifest["atomic_seen_tasks"]), "<registry-probe>", "exec")
        compile(_runtime_probe_code(manifest["runtime_smoke"]), "<runtime-probe>", "exec")

    def test_classification_separates_legacy_runtime_and_ready(self) -> None:
        legacy_registry = {
            "ok": False,
            "details": {"versions": {"robocasa": "0.2.0"}},
        }
        ok, recommendation, blockers = classify_reuse([], legacy_registry, None)
        self.assertFalse(ok)
        self.assertEqual(recommendation, "legacy_robocasa_only")
        self.assertTrue(blockers)

        registry = {"ok": True, "details": {"versions": {"robocasa": "1.0.1"}}}
        ok, recommendation, _ = classify_reuse([], registry, None)
        self.assertTrue(ok)
        self.assertEqual(recommendation, "runtime_smoke_required")

        ok, recommendation, _ = classify_reuse([], registry, {"ok": True})
        self.assertTrue(ok)
        self.assertEqual(recommendation, "reuse_ready")

    def test_cli_persists_failure_report_without_local_robocasa(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "simulator.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "audit_robocasa365_simulator_environment.py"),
                    "--output",
                    str(output),
                ],
                cwd=REPO_ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertIn(result.returncode, (0, 1), result.stderr)
            self.assertTrue(output.is_file())
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("reuse_recommendation", report)
            self.assertIn(str(output.resolve()), result.stderr)


if __name__ == "__main__":
    unittest.main()
