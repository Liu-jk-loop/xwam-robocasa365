from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "deployment/starlight/eval_atomic9_rgbd_step12000_xwam.sh"
PROBE = REPO_ROOT / "scripts/probe_starlight_eval_runtime.py"


class StarlightEvaluationDeploymentTest(unittest.TestCase):
    def test_shell_entrypoint_is_syntactically_valid(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(SCRIPT)],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_entrypoint_uses_two_conda_environments_and_model_only_gate(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("xwam-robocasa365", source)
        self.assertIn('SIMULATOR_ENV="${XWAM_STL_SIMULATOR_ENV:-robocasa}"', source)
        self.assertIn('REPO="${XWAM_STL_REPO:-$DEFAULT_REPO}"', source)
        self.assertIn('SCRIPT_DIR="$(cd --', source)
        self.assertIn("conda", source.lower())
        self.assertIn("mp_rank_00_model_states.pt", source)
        self.assertIn("26121537691", source)
        self.assertNotIn("optim_states.pt", source)
        self.assertNotIn("/capstor/", source)
        self.assertNotIn("/iopsstor/", source)
        self.assertNotIn("--environment=", source)

    def test_entrypoint_keeps_validated_atomic9_contract(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("robocasa365_atomic9_rgbd_step12000_6server_9client.json", source)
        self.assertIn("--episodes-per-task", source)
        self.assertIn("XWAM_STL_EPISODES:-50", source)
        self.assertIn("CUDA_VISIBLE_DEVICES=0,1,2,3", source)
        self.assertIn("use_depth", source)
        self.assertIn("policy_environment_probe", source)
        self.assertIn("simulator_environment_probe", source)
        self.assertIn('--manifest "$MANIFEST"', source)
        self.assertIn('--statistics-path "$STATS"', source)

    def test_probe_help_does_not_import_runtime_dependencies(self) -> None:
        result = subprocess.run(
            ["python3", str(PROBE), "--help"],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--role", result.stdout)


if __name__ == "__main__":
    unittest.main()
