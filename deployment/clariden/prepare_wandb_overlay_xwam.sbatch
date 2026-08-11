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

test -s "$EDF"
test -s "$REQUIREMENTS"
mkdir -p "$(dirname "$OVERLAY")" "$(dirname "$MANIFEST")"

srun --kill-on-bad-exit=1 --environment="$EDF" \
  env OVERLAY="$OVERLAY" REQUIREMENTS="$REQUIREMENTS" \
  bash -lc '
set -Eeuo pipefail

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

import wandb

overlay = Path(os.environ["OVERLAY"]).resolve()
module = Path(wandb.__file__).resolve()
assert module.is_relative_to(overlay), (module, overlay)
assert importlib.metadata.version("wandb") == "0.23.1"
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
print("wandb version:", importlib.metadata.version("wandb"))
print("wandb module:", module)
print("wandb offline probe: pass")
PY
  rm -rf "$PROBE_DIR"
  trap - RETURN
}

if [[ -d "$OVERLAY" ]]; then
  verify_overlay
  echo "[PASS] Existing X-WAM W&B overlay is valid"
  echo "[PASS] X-WAM W&B runtime overlay prepared"
  exit 0
fi

STAGING="$(mktemp -d "$(dirname "$OVERLAY")/.xwam-wandb-0.23.1.XXXXXX")"
trap '"'"'rm -rf "$STAGING"'"'"' EXIT
python -m pip install \
  --no-deps \
  --require-hashes \
  --target "$STAGING" \
  --requirement "$REQUIREMENTS"

OVERLAY="$STAGING" verify_overlay
if [[ -e "$OVERLAY" ]]; then
  echo "W&B overlay appeared during installation: $OVERLAY" >&2
  exit 2
fi
mv "$STAGING" "$OVERLAY"
trap - EXIT
verify_overlay
echo "[PASS] X-WAM W&B runtime overlay prepared"
'

MANIFEST_TMP="$(mktemp "$(dirname "$MANIFEST")/.xwam-wandb-overlay.XXXXXX")"
trap 'rm -f "$MANIFEST_TMP"' EXIT
{
  printf 'overlay=%s\n' "$OVERLAY"
  printf 'wandb=%s\n' '0.23.1'
  printf 'requirements_sha256=%s\n' "$(sha256sum "$REQUIREMENTS" | cut -d ' ' -f 1)"
  printf 'source_commit=%s\n' "$(git -C "$XWAM_REPO" rev-parse HEAD)"
  printf 'job_id=%s\n' "${SLURM_JOB_ID:-unknown}"
  printf 'created_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$MANIFEST_TMP"
mv "$MANIFEST_TMP" "$MANIFEST"
trap - EXIT

cat "$MANIFEST"
echo "[PASS] X-WAM W&B runtime overlay manifest written"
