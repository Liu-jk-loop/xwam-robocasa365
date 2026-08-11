#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --job-name=xwam_wandb
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/wandb-overlay-%j.log

set -Eeuo pipefail

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
DEPLOY_IOPS=/iopsstor/scratch/cscs/zjingchen/terry_nys
XWAM_REPO="$DEPLOY_STORE/src/xwam-robocasa365"
EDF="$DEPLOY_STORE/containers/edf/xwam.toml"
REQUIREMENTS="$XWAM_REPO/deployment/clariden/wandb-overlay-clariden.txt"
OVERLAY="$DEPLOY_IOPS/python/xwam-wandb-0.23.1"
MANIFEST="$DEPLOY_STORE/manifests/xwam_wandb_overlay.txt"
MAIN_LOG="$DEPLOY_STORE/logs/xwam/wandb-overlay-${SLURM_JOB_ID}.log"
FAILURE_REPORT="$DEPLOY_STORE/logs/xwam/wandb-overlay-${SLURM_JOB_ID}-failure.txt"

export XWAM_FAILURE_REPORT="$FAILURE_REPORT"
export XWAM_MAIN_LOG="$MAIN_LOG"
export XWAM_TRAIN_LOG=not-applicable
export XWAM_PHASE=wandb_overlay_outer_preflight
source "$XWAM_REPO/deployment/clariden/error_trap.sh"
xwam_install_err_trap

test -s "$EDF"
test -s "$REQUIREMENTS"
mkdir -p "$(dirname "$OVERLAY")" "$(dirname "$MANIFEST")"
echo "failure_report=$FAILURE_REPORT"

XWAM_PHASE=wandb_overlay_srun
srun --kill-on-bad-exit=1 --environment="$EDF" \
  env \
    OVERLAY="$OVERLAY" \
    REQUIREMENTS="$REQUIREMENTS" \
    XWAM_REPO="$XWAM_REPO" \
    XWAM_FAILURE_REPORT="$FAILURE_REPORT" \
    XWAM_MAIN_LOG="$MAIN_LOG" \
    XWAM_TRAIN_LOG=not-applicable \
  bash -lc '
set -Eeuo pipefail
export XWAM_PHASE=wandb_overlay_shell_setup
source "$XWAM_REPO/deployment/clariden/error_trap.sh"
xwam_install_err_trap

verify_overlay() {
  PROBE_DIR="$(mktemp -d)"
  trap '"'"'rm -rf "$PROBE_DIR"'"'"' RETURN
  PYTHONPATH="$OVERLAY:${PYTHONPATH:-}" \
    WANDB_MODE=offline \
    WANDB_SILENT=true \
    WANDB_DIR="$PROBE_DIR" \
    python - <<"PY"
import importlib.metadata
import os
from pathlib import Path

import sentry_sdk
import wandb
from packaging.requirements import Requirement

overlay = Path(os.environ["OVERLAY"]).resolve()
wandb_module = Path(wandb.__file__).resolve()
sentry_module = Path(sentry_sdk.__file__).resolve()
assert wandb_module.is_relative_to(overlay), (wandb_module, overlay)
assert sentry_module.is_relative_to(overlay), (sentry_module, overlay)

pinned_overlay_distributions = {
    "GitPython": "3.1.58",
    "gitdb": "4.0.12",
    "sentry-sdk": "2.58.0",
    "smmap": "5.0.3",
    "wandb": "0.23.1",
}
for distribution_name, expected_version in pinned_overlay_distributions.items():
    distribution = importlib.metadata.distribution(distribution_name)
    distribution_root = Path(distribution.locate_file("")).resolve()
    assert distribution_root.is_relative_to(overlay), (
        distribution_name,
        distribution_root,
        overlay,
    )
    assert distribution.version == expected_version, (
        distribution_name,
        distribution.version,
        expected_version,
    )


def validate_declared_dependencies(distribution):
    resolved = {}
    errors = []
    for raw_requirement in importlib.metadata.requires(distribution) or ():
        requirement = Requirement(raw_requirement)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        try:
            installed = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"{distribution}: missing {raw_requirement}")
            continue
        if not requirement.specifier.contains(installed, prereleases=True):
            errors.append(
                f"{distribution}: {raw_requirement}, installed={installed}"
            )
        resolved[requirement.name] = installed
    return resolved, errors


declared_dependencies = {}
dependency_errors = []
for distribution in pinned_overlay_distributions:
    resolved, errors = validate_declared_dependencies(distribution)
    declared_dependencies[distribution] = resolved
    dependency_errors.extend(errors)
assert not dependency_errors, dependency_errors
run = wandb.init(
    project="xwam-overlay-probe",
    id="xwam-overlay-probe",
    resume="allow",
    mode="offline",
    dir=os.environ["WANDB_DIR"],
)
assert run is not None
run.log({"probe": 1.0}, step=0)
run.finish()
print("wandb declared dependency versions:", declared_dependencies)
print("wandb module:", wandb_module)
print("sentry-sdk module:", sentry_module)
print("wandb offline probe: pass")
PY
  rm -rf "$PROBE_DIR"
  trap - RETURN
}

if [[ -d "$OVERLAY" ]]; then
  XWAM_PHASE=wandb_overlay_existing_probe
  verify_overlay
  echo "[PASS] Existing X-WAM W&B overlay is valid"
  echo "[PASS] X-WAM W&B runtime overlay prepared"
  exit 0
fi

XWAM_PHASE=wandb_overlay_install
STAGING="$(mktemp -d "$(dirname "$OVERLAY")/.xwam-wandb-0.23.1.XXXXXX")"
trap '"'"'rm -rf "$STAGING"'"'"' EXIT
python -m pip install \
  --no-deps \
  --require-hashes \
  --target "$STAGING" \
  --requirement "$REQUIREMENTS"

XWAM_PHASE=wandb_overlay_staging_probe
OVERLAY="$STAGING" verify_overlay
if [[ -e "$OVERLAY" ]]; then
  xwam_fail "W&B overlay appeared during installation: $OVERLAY"
fi
XWAM_PHASE=wandb_overlay_publish
mv "$STAGING" "$OVERLAY"
trap - EXIT
XWAM_PHASE=wandb_overlay_final_probe
verify_overlay
echo "[PASS] X-WAM W&B runtime overlay prepared"
'

XWAM_PHASE=wandb_overlay_manifest
MANIFEST_TMP="$(mktemp "$(dirname "$MANIFEST")/.xwam-wandb-overlay.XXXXXX")"
trap 'rm -f "$MANIFEST_TMP"' EXIT
{
  printf 'overlay=%s\n' "$OVERLAY"
  printf 'wandb=%s\n' '0.23.1'
  printf 'sentry_sdk=%s\n' '2.58.0'
  printf 'gitpython=%s\n' '3.1.58'
  printf 'gitdb=%s\n' '4.0.12'
  printf 'smmap=%s\n' '5.0.3'
  printf 'requirements_sha256=%s\n' "$(sha256sum "$REQUIREMENTS" | cut -d ' ' -f 1)"
  printf 'source_commit=%s\n' "$(git -C "$XWAM_REPO" rev-parse HEAD)"
  printf 'job_id=%s\n' "${SLURM_JOB_ID:-unknown}"
  printf 'created_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$MANIFEST_TMP"
mv "$MANIFEST_TMP" "$MANIFEST"
trap - EXIT

cat "$MANIFEST"
XWAM_PHASE=wandb_overlay_complete
echo "[PASS] X-WAM W&B runtime overlay manifest written"
