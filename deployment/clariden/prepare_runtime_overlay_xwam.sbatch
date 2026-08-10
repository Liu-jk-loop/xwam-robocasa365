#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --job-name=xwam_overlay
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/overlay-%j.log

set -euo pipefail

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
DEPLOY_IOPS=/iopsstor/scratch/cscs/zjingchen/terry_nys
XWAM_REPO="$DEPLOY_STORE/src/xwam-robocasa365"
EDF="$DEPLOY_STORE/containers/edf/xwam.toml"
REQUIREMENTS="$XWAM_REPO/deployment/clariden/runtime-overlay-clariden.txt"
OVERLAY="$DEPLOY_IOPS/python/xwam-nvtx-0.2.15"
MANIFEST="$DEPLOY_STORE/manifests/xwam_runtime_overlay.txt"

test -s "$EDF"
test -s "$REQUIREMENTS"
mkdir -p "$(dirname "$OVERLAY")" "$(dirname "$MANIFEST")"

srun --environment="$EDF" \
  env OVERLAY="$OVERLAY" REQUIREMENTS="$REQUIREMENTS" \
  bash -lc '
set -euo pipefail

verify_overlay() {
  PYTHONPATH="$OVERLAY:${PYTHONPATH:-}" python - <<"PY"
import importlib.metadata
import os
from pathlib import Path

import nvtx

overlay = Path(os.environ["OVERLAY"]).resolve()
module = Path(nvtx.__file__).resolve()
assert module.is_relative_to(overlay), (module, overlay)
assert importlib.metadata.version("nvtx") == "0.2.15"
assert callable(nvtx.get_domain)
domain = nvtx.get_domain("xwam-clariden-deepspeed-probe")
domain.push_range(message="probe", category=None)
domain.pop_range()
print("nvtx version:", importlib.metadata.version("nvtx"))
print("nvtx module:", module)
print("nvtx get_domain:", callable(nvtx.get_domain))
PY
}

if [[ -d "$OVERLAY" ]]; then
  verify_overlay
  echo "[PASS] Existing X-WAM runtime overlay is valid"
  exit 0
fi

STAGING="$(mktemp -d "$(dirname "$OVERLAY")/.xwam-nvtx-0.2.15.XXXXXX")"
trap '"'"'rm -rf "$STAGING"'"'"' EXIT
python -m pip install \
  --no-deps \
  --require-hashes \
  --target "$STAGING" \
  --requirement "$REQUIREMENTS"

OVERLAY="$STAGING" verify_overlay
if [[ -e "$OVERLAY" ]]; then
  echo "Runtime overlay appeared during installation: $OVERLAY" >&2
  exit 2
fi
mv "$STAGING" "$OVERLAY"
trap - EXIT
verify_overlay
echo "[PASS] X-WAM NVTX runtime overlay prepared"
'

{
  printf 'overlay=%s\n' "$OVERLAY"
  printf 'nvtx=%s\n' '0.2.15'
  printf 'requirements_sha256=%s\n' "$(sha256sum "$REQUIREMENTS" | cut -d ' ' -f 1)"
  printf 'source_commit=%s\n' "$(git -C "$XWAM_REPO" rev-parse HEAD)"
  printf 'job_id=%s\n' "${SLURM_JOB_ID:-unknown}"
  printf 'created_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$MANIFEST"

cat "$MANIFEST"
echo "[PASS] X-WAM runtime overlay manifest written"
