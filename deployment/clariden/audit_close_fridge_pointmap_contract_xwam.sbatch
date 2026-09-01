#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --job-name=xwam_cf_pm_p0
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/close-fridge-pointmap-p0-%j.log
#SBATCH --error=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/close-fridge-pointmap-p0-%j.err

set -Eeuo pipefail

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
REPO="$DEPLOY_STORE/src/xwam-robocasa365"
ROBOCASA_ROOT="$DEPLOY_STORE/src/robocasa"
ROBOSUITE_ROOT="$DEPLOY_STORE/src/robosuite"
ROBOCASA_EDF="$DEPLOY_STORE/containers/edf/robocasa365.toml"
DATA_ROOT="$DEPLOY_STORE/datasets/robocasa/v1.0/pretrain/atomic"
EVIDENCE_ROOT="$DEPLOY_STORE/manifests/xwam/pointmap/close_fridge"
REPORT="$EVIDENCE_ROOT/close_fridge_pointmap_p0_${SLURM_JOB_ID}.json"
ARTIFACTS_DIR="$EVIDENCE_ROOT/artifacts_${SLURM_JOB_ID}"
TASK_MANIFEST=configs/tasks/robocasa365_rgbd_close_fridge.json

cd "$REPO"
test -s "$ROBOCASA_EDF"
test -d "$DATA_ROOT"
test -d "$ROBOCASA_ROOT/robocasa"
test -d "$ROBOSUITE_ROOT/robosuite"
mkdir -p "$EVIDENCE_ROOT"

echo "commit=$(git rev-parse HEAD)"
if test -n "$(git status --porcelain)"; then
  echo "[WARN] Git worktree is dirty; exact status will be recorded in the JSON report" >&2
  git status --short >&2
fi

srun --kill-on-bad-exit=1 --gpus-per-node=1 --environment="$ROBOCASA_EDF" \
  env REPO="$REPO" ROBOCASA_ROOT="$ROBOCASA_ROOT" ROBOSUITE_ROOT="$ROBOSUITE_ROOT" \
    DATA_ROOT="$DATA_ROOT" REPORT="$REPORT" ARTIFACTS_DIR="$ARTIFACTS_DIR" \
    TASK_MANIFEST="$TASK_MANIFEST" \
  bash -lc '
set -Eeuo pipefail
cd "$REPO"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export EGL_DEVICE_ID=0
export ROBOSUITE_RENDER_GPU_DEVICE_ID=0
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$ROBOCASA_ROOT:$ROBOSUITE_ROOT:$REPO${PYTHONPATH:+:$PYTHONPATH}"

python scripts/audit_robocasa365_pointmap_contract.py \
  --dataset-root "$DATA_ROOT" \
  --task-manifest "$TASK_MANIFEST" \
  --episodes-per-task 3 \
  --frame-fractions 0,0.5,1 \
  --height 256 \
  --width 256 \
  --target-height 256 \
  --target-width 320 \
  --max-rgb-mae 12 \
  --max-reprojection-error-px 0.001 \
  --max-float16-roundtrip-error-mm 2 \
  --max-intrinsics-drift 0.000001 \
  --artifacts-dir "$ARTIFACTS_DIR" \
  --output "$REPORT"
'

test -s "$REPORT"
grep -q '"ok": true' "$REPORT"
grep -q '"result": "pass"' "$REPORT"
grep -q '"passed_tasks": 1' "$REPORT"
grep -q '"camera_count": 3' "$REPORT"
grep -q '"float16_roundtrip": true' "$REPORT"
grep -q '"reprojection": true' "$REPORT"

echo "pointmap_p0_report=$REPORT"
echo "pointmap_p0_artifacts=$ARTIFACTS_DIR"
echo "[PASS] CloseFridge three-camera PointMap numerical contract"
