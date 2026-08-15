#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --job-name=xwam_a9_data
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/atomic9-ratio00-data-%j.log

set -Eeuo pipefail

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
DEPLOY_CAPSCR=/capstor/scratch/cscs/zjingchen/terry_nys
REPO="$DEPLOY_STORE/src/xwam-robocasa365"
EDF="$DEPLOY_STORE/containers/edf/xwam.toml"
DATA_ROOT="$DEPLOY_STORE/datasets/robocasa/v1.0/pretrain/atomic"
EVIDENCE_ROOT="$DEPLOY_STORE/manifests/xwam/atomic9_ratio00"
MANIFEST="$EVIDENCE_ROOT/robocasa365_atomic9_fastwam_overlap_manifest.json"
STATS="$EVIDENCE_ROOT/robocasa365_atomic9_fastwam_overlap_global_stats.json"
PREFLIGHT="$EVIDENCE_ROOT/robocasa365_atomic9_ratio00_gh200_preflight.json"
TASK_MANIFEST=configs/tasks/robocasa365_atomic9_fastwam_overlap.json
STATS_WORK="$DEPLOY_CAPSCR/tmp/xwam-atomic9-ratio00-stats-${SLURM_JOB_ID}"

cd "$REPO"
test -s "$EDF"
test -d "$DATA_ROOT"
test -z "$(git status --porcelain)" || {
  echo "[FAIL] Git worktree must be clean before Atomic9 data provenance is frozen" >&2
  git status --short >&2
  exit 2
}
mkdir -p "$EVIDENCE_ROOT" "$STATS_WORK"

srun --kill-on-bad-exit=1 --environment="$EDF" \
  env \
    REPO="$REPO" \
    DATA_ROOT="$DATA_ROOT" \
    MANIFEST="$MANIFEST" \
    STATS="$STATS" \
    PREFLIGHT="$PREFLIGHT" \
    TASK_MANIFEST="$TASK_MANIFEST" \
    STATS_WORK="$STATS_WORK" \
  bash -lc '
set -Eeuo pipefail
cd "$REPO"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

git rev-parse HEAD
test -z "$(git status --porcelain)"

python scripts/build_robocasa365_m6_training_manifest.py \
  --dataset-root "$DATA_ROOT" \
  --task-manifest "$TASK_MANIFEST" \
  --output "$MANIFEST"

python scripts/compute_robocasa365_global_stats.py \
  --manifest "$MANIFEST" \
  --task-manifest "$TASK_MANIFEST" \
  --expected-task-count 9 \
  --work-dir "$STATS_WORK" \
  --output "$STATS"

python scripts/audit_robocasa365_m6_preflight.py \
  --manifest "$MANIFEST" \
  --stats "$STATS" \
  --global-batch-size 128 \
  --epochs 5 \
  --expected-task-count 9 \
  --fixed-training-steps 8500 \
  --output "$PREFLIGHT"
'

for artifact in "$MANIFEST" "$STATS" "$PREFLIGHT"; do
  test -s "$artifact"
  grep -q '"ok": true' "$artifact"
  grep -q '"result": "pass"' "$artifact"
done
grep -q '"schedule_mode": "fixed_steps"' "$PREFLIGHT"
grep -q '"fixed_training_steps": 8500' "$PREFLIGHT"
grep -q '"expected_task_count": 9' "$PREFLIGHT"

echo "manifest=$MANIFEST"
echo "global_stats=$STATS"
echo "preflight=$PREFLIGHT"
echo "[PASS] X-WAM Clariden Atomic9 manifest, global stats and fixed 8500-step schedule"
