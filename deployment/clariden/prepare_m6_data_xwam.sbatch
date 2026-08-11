#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --job-name=xwam_m6_data
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/m6-data-%j.log

set -euo pipefail

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
DEPLOY_CAPSCR=/capstor/scratch/cscs/zjingchen/terry_nys
EDF="$DEPLOY_STORE/containers/edf/xwam.toml"
DATA_ROOT="$DEPLOY_STORE/datasets/robocasa/v1.0/pretrain/atomic"
EVIDENCE_ROOT="$DEPLOY_STORE/manifests/xwam/m6"
MANIFEST="$EVIDENCE_ROOT/robocasa365_m6_atomic_seen18_manifest.json"
STATS="$EVIDENCE_ROOT/robocasa365_m6_atomic_seen18_global_stats.json"
PREFLIGHT="$EVIDENCE_ROOT/robocasa365_m6_gh200_preflight.json"
STATS_WORK="$DEPLOY_CAPSCR/tmp/xwam-m6-stats-${SLURM_JOB_ID}"

test -s "$EDF"
test -d "$DATA_ROOT"
test -z "$(git status --porcelain)" || {
  echo "[FAIL] Git worktree must be clean before M6 data provenance is frozen" >&2
  git status --short >&2
  exit 2
}
mkdir -p "$EVIDENCE_ROOT" "$STATS_WORK"

srun --kill-on-bad-exit=1 --environment="$EDF" \
  env \
    DATA_ROOT="$DATA_ROOT" \
    MANIFEST="$MANIFEST" \
    STATS="$STATS" \
    PREFLIGHT="$PREFLIGHT" \
    STATS_WORK="$STATS_WORK" \
  bash -lc '
set -euo pipefail
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

git rev-parse HEAD
test -z "$(git status --porcelain)"

python scripts/build_robocasa365_m6_training_manifest.py \
  --dataset-root "$DATA_ROOT" \
  --output "$MANIFEST"

python scripts/compute_robocasa365_global_stats.py \
  --manifest "$MANIFEST" \
  --work-dir "$STATS_WORK" \
  --output "$STATS"

python scripts/audit_robocasa365_m6_preflight.py \
  --manifest "$MANIFEST" \
  --stats "$STATS" \
  --global-batch-size 128 \
  --epochs 5 \
  --output "$PREFLIGHT"
'

for artifact in "$MANIFEST" "$STATS" "$PREFLIGHT"; do
  test -s "$artifact"
  grep -q '"ok": true' "$artifact"
  grep -q '"result": "pass"' "$artifact"
done

echo "manifest=$MANIFEST"
echo "global_stats=$STATS"
echo "preflight=$PREFLIGHT"
echo "[PASS] X-WAM Clariden M6 Atomic-Seen 18 manifest, global stats and schedule"
