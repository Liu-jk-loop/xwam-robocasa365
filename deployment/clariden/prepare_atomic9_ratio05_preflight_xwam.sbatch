#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --job-name=xwam_a9_r05_gate
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/atomic9-ratio05-preflight-%j.log
#SBATCH --export=ALL

set -Eeuo pipefail

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
REPO="$DEPLOY_STORE/src/xwam-robocasa365"
EDF="$DEPLOY_STORE/containers/edf/xwam.toml"
RATIO00_EVIDENCE="$DEPLOY_STORE/manifests/xwam/atomic9_ratio00"
RATIO05_EVIDENCE="$DEPLOY_STORE/manifests/xwam/atomic9_ratio05"
MANIFEST="$RATIO00_EVIDENCE/robocasa365_atomic9_fastwam_overlap_manifest.json"
STATS="$RATIO00_EVIDENCE/robocasa365_atomic9_fastwam_overlap_global_stats.json"
PREFLIGHT="$RATIO05_EVIDENCE/robocasa365_atomic9_ratio05_gh200_preflight.json"

cd "$REPO"
test -s "$EDF"
test -z "$(git status --porcelain)" || {
  echo "[FAIL] Git worktree must be clean before Atomic9 ratio05 preflight" >&2
  git status --short >&2
  exit 2
}
for artifact in "$MANIFEST" "$STATS"; do
  test -s "$artifact"
  grep -q '"ok": true' "$artifact"
  grep -q '"result": "pass"' "$artifact"
done
mkdir -p "$RATIO05_EVIDENCE"

srun --kill-on-bad-exit=1 --environment="$EDF" \
  env REPO="$REPO" MANIFEST="$MANIFEST" STATS="$STATS" PREFLIGHT="$PREFLIGHT" \
  bash -lc '
set -Eeuo pipefail
cd "$REPO"
python scripts/audit_robocasa365_m6_preflight.py \
  --manifest "$MANIFEST" \
  --stats "$STATS" \
  --global-batch-size 128 \
  --epochs 5 \
  --expected-task-count 9 \
  --fixed-training-steps 8500 \
  --output "$PREFLIGHT"
'

test -s "$PREFLIGHT"
grep -q '"ok": true' "$PREFLIGHT"
grep -q '"result": "pass"' "$PREFLIGHT"
grep -q '"schedule_mode": "fixed_steps"' "$PREFLIGHT"
grep -q '"fixed_training_steps": 8500' "$PREFLIGHT"
grep -q '"expected_task_count": 9' "$PREFLIGHT"

echo "manifest=$MANIFEST"
echo "global_stats=$STATS"
echo "preflight=$PREFLIGHT"
echo "[PASS] X-WAM Clariden Atomic9 ratio05 reuses the ratio00 8500-step scheduler contract; training target is 7500"
