#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --job-name=xwam_a9_rgb_cont_pre
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/atomic9-rgb-cont12000-preflight-%j.log
#SBATCH --export=ALL

set -Eeuo pipefail

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
DEPLOY_IOPS=/iopsstor/scratch/cscs/zjingchen/terry_nys
REPO="$DEPLOY_STORE/src/xwam-robocasa365"
EDF="$DEPLOY_STORE/containers/edf/xwam.toml"
RATIO00_EVIDENCE="$DEPLOY_STORE/manifests/xwam/atomic9_ratio00"
CONT_EVIDENCE="$DEPLOY_STORE/manifests/xwam/atomic9_rgb_cont7500_12000"
MANIFEST="$RATIO00_EVIDENCE/robocasa365_atomic9_fastwam_overlap_manifest.json"
STATS="$RATIO00_EVIDENCE/robocasa365_atomic9_fastwam_overlap_global_stats.json"
PREFLIGHT="$CONT_EVIDENCE/robocasa365_atomic9_ratio00_rgb_cont7500_12000_preflight.json"
BOOTSTRAP_PLAN="$CONT_EVIDENCE/robocasa365_atomic9_ratio00_rgb_cont7500_12000_bootstrap.json"
BOOTSTRAP_ENV="$CONT_EVIDENCE/robocasa365_atomic9_ratio00_rgb_cont7500_12000_bootstrap.env"
SOURCE_EXP=robocasa365_atomic9_fastwam_overlap_ratio00_rgb_seed42_8gpu
CONT_EXP=robocasa365_atomic9_ratio00_rgb_cont7500_12000_seed42_8gpu
SOURCE_HOT="$DEPLOY_IOPS/xwam_run/$SOURCE_EXP/checkpoints"
SOURCE_DURABLE="$DEPLOY_STORE/checkpoints/xwam/$SOURCE_EXP/checkpoints"
CONT_HOT="$DEPLOY_IOPS/xwam_run/$CONT_EXP/checkpoints"
CONT_DURABLE="$DEPLOY_STORE/checkpoints/xwam/$CONT_EXP/checkpoints"
TMP_ROOT="$DEPLOY_STORE/manifests/xwam/atomic9_rgb_cont7500_12000/tmp-${SLURM_JOB_ID}"

cd "$REPO"
test -s "$EDF"
git rev-parse HEAD
if [[ -n "$(git status --porcelain)" ]]; then
  echo "[WARN] RGB continuation preflight records a dirty worktree but does not block"
  git status --short
fi
for artifact in "$MANIFEST" "$STATS"; do
  test -s "$artifact"
  grep -q '"ok": true' "$artifact"
  grep -q '"result": "pass"' "$artifact"
done
if [[ ! -d "$SOURCE_HOT" && ! -d "$SOURCE_DURABLE" ]]; then
  echo "[FAIL] neither source checkpoint root exists" >&2
  exit 1
fi
mkdir -p "$CONT_EVIDENCE"

srun --kill-on-bad-exit=1 --environment="$EDF" \
  env \
    REPO="$REPO" \
    MANIFEST="$MANIFEST" \
    STATS="$STATS" \
    PREFLIGHT="$PREFLIGHT" \
    BOOTSTRAP_PLAN="$BOOTSTRAP_PLAN" \
    BOOTSTRAP_ENV="$BOOTSTRAP_ENV" \
    SOURCE_HOT="$SOURCE_HOT" \
    SOURCE_DURABLE="$SOURCE_DURABLE" \
    CONT_HOT="$CONT_HOT" \
    CONT_DURABLE="$CONT_DURABLE" \
    TMP_ROOT="$TMP_ROOT" \
  bash -lc '
set -Eeuo pipefail
cd "$REPO"
python scripts/audit_robocasa365_m6_preflight.py \
  --manifest "$MANIFEST" \
  --stats "$STATS" \
  --global-batch-size 128 \
  --epochs 5 \
  --expected-task-count 9 \
  --fixed-training-steps 12000 \
  --output "$PREFLIGHT"
python scripts/plan_robocasa365_m6_formal_chunk.py \
  --checkpoint-root "$CONT_HOT" \
  --checkpoint-root "$CONT_DURABLE" \
  --bootstrap-checkpoint-root "$SOURCE_HOT" \
  --bootstrap-checkpoint-root "$SOURCE_DURABLE" \
  --bootstrap-step 7500 \
  --total-steps 12000 \
  --chunk-steps 12000 \
  --expected-world-size 8 \
  --output-json "$BOOTSTRAP_PLAN" \
  --output-env "$BOOTSTRAP_ENV" \
  --quarantine-incomplete-root "$TMP_ROOT/hot" \
  --quarantine-incomplete-root "$TMP_ROOT/durable"
'

test -s "$PREFLIGHT"
test -s "$BOOTSTRAP_PLAN"
grep -q '"ok": true' "$PREFLIGHT"
grep -q '"result": "pass"' "$PREFLIGHT"
grep -q '"fixed_training_steps": 12000' "$PREFLIGHT"
grep -q '"expected_task_count": 9' "$PREFLIGHT"
grep -q '"bootstrap_step": 7500' "$BOOTSTRAP_PLAN"
grep -q '"completed_step": 7500' "$BOOTSTRAP_PLAN"
grep -q '"target_step": 12000' "$BOOTSTRAP_PLAN"
grep -q 'step=7500.ckpt' "$BOOTSTRAP_PLAN"

echo "preflight=$PREFLIGHT"
echo "bootstrap_plan=$BOOTSTRAP_PLAN"
echo "bootstrap_env=$BOOTSTRAP_ENV"
echo "[PASS] Atomic9 RGB step7500 full-state continuation to step12000 preflight"
