from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = (
    REPO_ROOT
    / ".agents/skills/xwam-robocasa365-workflow/scripts/check_slurm_env_contract.py"
)
SPEC = importlib.util.spec_from_file_location("check_slurm_env_contract", CHECKER_PATH)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CHECKER
SPEC.loader.exec_module(CHECKER)


class SlurmEnvironmentContractTest(unittest.TestCase):
    def _audit(self, script: str):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "job.sbatch"
            path.write_text(script, encoding="utf-8")
            return CHECKER.audit_path(path)

    def test_repository_clariden_scripts_have_complete_explicit_contracts(self) -> None:
        errors = []
        for path in sorted((REPO_ROOT / "deployment/clariden").glob("*.sbatch")):
            errors.extend(CHECKER.audit_path(path))
        self.assertEqual(errors, [])

    def test_missing_outer_variable_is_rejected(self) -> None:
        errors = self._audit(
            '''#!/bin/bash
MILESTONE_STEP=6855
REPO=/repo
srun env REPO="$REPO" \\
  bash -lc '
set -Eeuo pipefail
cd "$REPO"
test -n "$MILESTONE_STEP"
'
'''
        )
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].missing, ("MILESTONE_STEP",))

    def test_explicitly_forwarded_variable_passes(self) -> None:
        errors = self._audit(
            '''#!/bin/bash
MILESTONE_STEP=6855
REPO=/repo
srun env REPO="$REPO" MILESTONE_STEP="$MILESTONE_STEP" \\
  bash -lc '
set -Eeuo pipefail
cd "$REPO"
test -n "$MILESTONE_STEP"
'
'''
        )
        self.assertEqual(errors, [])

    def test_inner_local_variable_does_not_require_forwarding(self) -> None:
        errors = self._audit(
            '''#!/bin/bash
REPO=/repo
srun env REPO="$REPO" \\
  bash -lc '
set -Eeuo pipefail
LOCAL_REPORT="$REPO/report.json"
test -n "$LOCAL_REPORT"
'
'''
        )
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
