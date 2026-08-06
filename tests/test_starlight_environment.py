from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from project_tools.starlight_environment import (
    apply_pip_check_policy,
    classify_clone_base,
    evaluate_packages,
    version_key,
    version_satisfies,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class StarlightEnvironmentTest(unittest.TestCase):
    def test_pip_policy_allows_only_explicit_runtime_validated_wheel_tag(self) -> None:
        result = {
            "ok": False,
            "returncode": 1,
            "stdout": "decord 0.6.0 is not supported on this platform\n",
            "stderr": "",
        }
        evaluated = apply_pip_check_policy(result, {"decord": "0.6.0"})
        self.assertTrue(evaluated["ok"])
        self.assertFalse(evaluated["raw_ok"])
        self.assertEqual(evaluated["remaining_lines"], [])

    def test_pip_policy_does_not_hide_dependency_conflicts(self) -> None:
        result = {
            "ok": False,
            "returncode": 1,
            "stdout": (
                "wam 0.1.0 has requirement numpy==1.26.4, but you have numpy 1.23.5.\n"
                "decord 0.6.0 is not supported on this platform\n"
            ),
            "stderr": "",
        }
        evaluated = apply_pip_check_policy(result, {"decord": "0.6.0"})
        self.assertFalse(evaluated["ok"])
        self.assertEqual(len(evaluated["ignored_lines"]), 1)
        self.assertEqual(len(evaluated["remaining_lines"]), 1)

    def test_pip_policy_does_not_allow_unvalidated_wheel_version(self) -> None:
        result = {
            "ok": False,
            "returncode": 1,
            "stdout": "decord 0.7.0 is not supported on this platform\n",
            "stderr": "",
        }
        evaluated = apply_pip_check_policy(result, {"decord": "0.6.0"})
        self.assertFalse(evaluated["ok"])
        self.assertEqual(evaluated["ignored_lines"], [])

    def test_cluster_constraints_protect_audited_cuda_stack(self) -> None:
        constraints = (
            REPO_ROOT / "configs" / "environment" / "xwam_starlight_constraints.txt"
        ).read_text(encoding="utf-8")
        for expected in (
            "torch==2.9.0",
            "torchvision==0.24.0",
            "torchaudio==2.9.0",
            "flash-attn==2.8.3",
            "numpy==1.23.5",
            "transformers==4.51.3",
        ):
            self.assertIn(expected, constraints)

    def test_dependency_installer_refuses_source_environment(self) -> None:
        environment = os.environ.copy()
        environment["CONDA_DEFAULT_ENV"] = "abot_m05"
        result = subprocess.run(
            [
                "bash",
                str(REPO_ROOT / "scripts" / "install_starlight_dependencies.sh"),
                "dry-run",
            ],
            cwd=REPO_ROOT,
            env=environment,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("只读母环境", result.stderr)

    def test_version_key_handles_cuda_and_build_suffixes(self) -> None:
        self.assertEqual(version_key("2.8.0+cu128"), (2, 8, 0, 0))
        self.assertEqual(version_key("4.51.3.dev0"), (4, 51, 3, 0))

    def test_version_bounds(self) -> None:
        spec = {"min": "1.23.5", "max_exclusive": "1.26.0"}
        self.assertTrue(version_satisfies("1.23.5", spec)[0])
        self.assertTrue(version_satisfies("1.25.2", spec)[0])
        self.assertFalse(version_satisfies("1.22.4", spec)[0])
        self.assertFalse(version_satisfies("1.26.0", spec)[0])

    def test_inclusive_upper_bound(self) -> None:
        spec = {"min": "4.49.0", "max_inclusive": "4.51.3"}
        self.assertTrue(version_satisfies("4.51.3", spec)[0])
        self.assertFalse(version_satisfies("4.52.0", spec)[0])

    def test_package_evaluation_separates_errors_and_warnings(self) -> None:
        versions = {"torch": "2.8.0+cu128", "ninja": None, "numpy": "1.26.1"}
        specs = [
            {"distribution": "torch", "module": "torch", "min": "2.4.0", "tested": "2.8.0"},
            {"distribution": "ninja", "module": "ninja", "required": False},
            {"distribution": "numpy", "module": "numpy", "max_exclusive": "1.26.0"},
        ]
        results, errors, warnings = evaluate_packages(
            specs,
            version_lookup=versions.get,
            run_imports=False,
        )
        self.assertEqual(len(results), 3)
        self.assertTrue(any("numpy" in error for error in errors))
        self.assertTrue(any("ninja" in warning for warning in warnings))

    def test_missing_patchable_packages_do_not_reject_clone_base(self) -> None:
        packages = [
            {"distribution": name, "installed": version, "constraint_ok": True, "runtime_import": {"ok": True}}
            for name, version in (
                ("torch", "2.9.0"),
                ("torchvision", "0.24.0"),
                ("torchaudio", "2.9.0"),
            )
        ]
        packages.append({"distribution": "deepspeed", "installed": None, "constraint_ok": False})
        clone_base_ok, blockers = classify_clone_base(
            True,
            packages,
            {"ok": True, "details": {"cuda_available": True}},
            require_gpu=True,
        )
        self.assertTrue(clone_base_ok)
        self.assertEqual(blockers, [])

    def test_broken_torch_rejects_clone_base(self) -> None:
        packages = [
            {"distribution": "torch", "installed": "2.9.0", "constraint_ok": True, "runtime_import": {"ok": False}},
            {"distribution": "torchvision", "installed": "0.24.0", "constraint_ok": True},
            {"distribution": "torchaudio", "installed": "2.9.0", "constraint_ok": True},
        ]
        clone_base_ok, blockers = classify_clone_base(
            True,
            packages,
            {"ok": False},
            require_gpu=True,
        )
        self.assertFalse(clone_base_ok)
        self.assertTrue(any("torch" in blocker.lower() for blocker in blockers))

    def test_cli_persists_report_before_returning_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "cluster" / "environment.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "audit_starlight_environment.py"),
                    "--skip-runtime-imports",
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
            self.assertIn("ok", report)
            self.assertIn(str(output.resolve()), result.stderr)


if __name__ == "__main__":
    unittest.main()
