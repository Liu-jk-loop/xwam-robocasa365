#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --job-name=xwam_a9_rgbd_p1
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/atomic9-rgbd-p1-%j.log
#SBATCH --error=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/atomic9-rgbd-p1-%j.err

set -Eeuo pipefail

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
REPO="$DEPLOY_STORE/src/xwam-robocasa365"
ROBOCASA_ROOT="$DEPLOY_STORE/src/robocasa"
ROBOSUITE_ROOT="$DEPLOY_STORE/src/robosuite"
ROBOCASA_EDF="$DEPLOY_STORE/containers/edf/robocasa365.toml"
DATA_ROOT="$DEPLOY_STORE/datasets/robocasa/v1.0/pretrain/atomic"
EVIDENCE_ROOT="$DEPLOY_STORE/manifests/xwam/atomic9_rgbd"
STRUCTURE_REPORT="$EVIDENCE_ROOT/robocasa365_atomic9_rgbd_replay_inputs.json"
RENDER_REPORT="$EVIDENCE_ROOT/robocasa365_atomic9_rgbd_render_probe.json"
ARTIFACTS_DIR="$EVIDENCE_ROOT/render_artifacts_job_${SLURM_JOB_ID}"
TASK_MANIFEST=configs/tasks/robocasa365_atomic9_fastwam_overlap.json
MAX_RGB_MAE="${XWAM_RGBD_MAX_RGB_MAE:-12.0}"

cd "$REPO"
test -s "$ROBOCASA_EDF"
test -d "$DATA_ROOT"
test -d "$ROBOCASA_ROOT/robocasa"
test -d "$ROBOSUITE_ROOT/robosuite"
test -z "$(git status --porcelain)" || {
  echo "[FAIL] Git worktree must be clean before RGB-D provenance audit" >&2
  git status --short >&2
  exit 2
}
mkdir -p "$EVIDENCE_ROOT"

srun --kill-on-bad-exit=1 --gpus-per-node=1 --environment="$ROBOCASA_EDF" \
  env REPO="$REPO" ROBOCASA_ROOT="$ROBOCASA_ROOT" ROBOSUITE_ROOT="$ROBOSUITE_ROOT" \
    DATA_ROOT="$DATA_ROOT" STRUCTURE_REPORT="$STRUCTURE_REPORT" \
    RENDER_REPORT="$RENDER_REPORT" ARTIFACTS_DIR="$ARTIFACTS_DIR" \
    TASK_MANIFEST="$TASK_MANIFEST" MAX_RGB_MAE="$MAX_RGB_MAE" \
  bash -lc '
set -Eeuo pipefail
cd "$REPO"
git rev-parse HEAD
test -z "$(git status --porcelain)"
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

python scripts/audit_robocasa365_depth_replay_inputs.py \
  --dataset-root "$DATA_ROOT" \
  --task-manifest "$TASK_MANIFEST" \
  --episodes-to-decode 3 \
  --output "$STRUCTURE_REPORT"

python scripts/probe_robocasa365_depth_render.py \
  --dataset-root "$DATA_ROOT" \
  --task-manifest "$TASK_MANIFEST" \
  --episodes-per-task 3 \
  --frame-fractions 0,0.5,1 \
  --height 256 \
  --width 256 \
  --max-rgb-mae "$MAX_RGB_MAE" \
  --artifacts-dir "$ARTIFACTS_DIR" \
  --output "$RENDER_REPORT"
'

test -s "$STRUCTURE_REPORT"
grep -q '"ok": true' "$STRUCTURE_REPORT"
grep -q '"passed_tasks": 9' "$STRUCTURE_REPORT"
test -s "$RENDER_REPORT"
grep -q '"ok": true' "$RENDER_REPORT"
grep -q '"result": "pass"' "$RENDER_REPORT"
grep -q '"passed_tasks": 9' "$RENDER_REPORT"

echo "rgbd_replay_inputs=$STRUCTURE_REPORT"
echo "rgbd_render_probe=$RENDER_REPORT"
echo "rgbd_render_artifacts=$ARTIFACTS_DIR"
echo "[PASS] Atomic9 per-episode MJCF/state RGB-D replay and alignment ready"
