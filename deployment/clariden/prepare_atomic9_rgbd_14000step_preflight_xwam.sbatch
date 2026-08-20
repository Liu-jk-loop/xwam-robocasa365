#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --job-name=xwam_a9_rgbd_14k
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/atomic9-rgbd-14000step-preflight-%j.log
#SBATCH --export=ALL

set -Eeuo pipefail

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
REPO="$DEPLOY_STORE/src/xwam-robocasa365"
EDF="$DEPLOY_STORE/containers/edf/xwam.toml"
RATIO00_EVIDENCE="$DEPLOY_STORE/manifests/xwam/atomic9_ratio00"
RGBD_EVIDENCE="$DEPLOY_STORE/manifests/xwam/atomic9_rgbd"
MANIFEST="$RATIO00_EVIDENCE/robocasa365_atomic9_fastwam_overlap_manifest.json"
STATS="$RATIO00_EVIDENCE/robocasa365_atomic9_fastwam_overlap_global_stats.json"
PREFLIGHT="$RGBD_EVIDENCE/robocasa365_atomic9_ratio00_rgbd_14000step_preflight.json"
SCHEDULE_ENV="$RGBD_EVIDENCE/robocasa365_atomic9_ratio00_rgbd_14000step_schedule.env"

cd "$REPO"
test -s "$EDF"
test -z "$(git status --porcelain)" || {
  echo "[FAIL] Git worktree must be clean before Atomic9 RGB-D preflight" >&2
  git status --short >&2
  exit 2
}
for artifact in "$MANIFEST" "$STATS"; do
  test -s "$artifact"
  grep -q '"ok": true' "$artifact"
  grep -q '"result": "pass"' "$artifact"
done
mkdir -p "$RGBD_EVIDENCE"

srun --kill-on-bad-exit=1 --environment="$EDF" \
  env REPO="$REPO" MANIFEST="$MANIFEST" STATS="$STATS" PREFLIGHT="$PREFLIGHT" \
    SCHEDULE_ENV="$SCHEDULE_ENV" \
  bash -lc '
set -Eeuo pipefail
cd "$REPO"
python scripts/audit_robocasa365_m6_preflight.py \
  --manifest "$MANIFEST" \
  --stats "$STATS" \
  --global-batch-size 128 \
  --epochs 8 \
  --expected-task-count 9 \
  --fixed-training-steps 14000 \
  --output "$PREFLIGHT"
python scripts/export_robocasa365_epoch_schedule.py \
  --preflight "$PREFLIGHT" \
  --expected-epochs 8 \
  --expected-total-steps 14000 \
  --milestone-epoch 5 \
  --output "$SCHEDULE_ENV"
'

test -s "$PREFLIGHT"
grep -q '"ok": true' "$PREFLIGHT"
grep -q '"result": "pass"' "$PREFLIGHT"
grep -q '"schedule_mode": "fixed_steps"' "$PREFLIGHT"
grep -q '"fixed_training_steps": 14000' "$PREFLIGHT"
grep -q '"num_training_steps": 14000' "$PREFLIGHT"
grep -q '"expected_task_count": 9' "$PREFLIGHT"
test -s "$SCHEDULE_ENV"
# shellcheck disable=SC1090
source "$SCHEDULE_ENV"
[[ "$XWAM_SCHEDULE_NUM_EPOCHS" == 8 ]]
[[ "$XWAM_SCHEDULE_TOTAL_STEPS" == 14000 ]]
[[ "$XWAM_SCHEDULE_MILESTONE_EPOCH" == 5 ]]
[[ "$XWAM_SCHEDULE_MILESTONE_STEP" =~ ^[1-9][0-9]*$ ]]
((XWAM_SCHEDULE_MILESTONE_STEP < XWAM_SCHEDULE_TOTAL_STEPS))

echo "manifest=$MANIFEST"
echo "global_stats=$STATS"
echo "preflight=$PREFLIGHT"
echo "schedule_env=$SCHEDULE_ENV"
echo "total_steps=$XWAM_SCHEDULE_TOTAL_STEPS"
echo "epoch5_step=$XWAM_SCHEDULE_MILESTONE_STEP"
echo "[PASS] X-WAM Clariden Atomic9 RGB-D fixed 14000-step schedule with epoch-5 milestone"
