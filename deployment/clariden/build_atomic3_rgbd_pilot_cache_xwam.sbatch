#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --job-name=xwam_a3_rgbd_p2
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/atomic3-rgbd-p2-%j.log
#SBATCH --error=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/atomic3-rgbd-p2-%j.err

set -Eeuo pipefail

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
DEPLOY_IOPS=/iopsstor/scratch/cscs/zjingchen/terry_nys
REPO="$DEPLOY_STORE/src/xwam-robocasa365"
ROBOCASA_ROOT="$DEPLOY_STORE/src/robocasa"
ROBOSUITE_ROOT="$DEPLOY_STORE/src/robosuite"
ROBOCASA_EDF="$DEPLOY_STORE/containers/edf/robocasa365.toml"
DATA_ROOT="$DEPLOY_STORE/datasets/robocasa/v1.0/pretrain/atomic"
EVIDENCE_ROOT="$DEPLOY_STORE/manifests/xwam/atomic9_rgbd"
CACHE_ROOT="$DEPLOY_IOPS/x-wam-depth-cache/robocasa365_inverse_metric_global_q_v1/pilot3"
ENCODING_OUTPUT="$EVIDENCE_ROOT/robocasa365_inverse_metric_global_q_v1.json"
MANIFEST_OUTPUT="$EVIDENCE_ROOT/robocasa365_atomic3_rgbd_pilot_cache_manifest.json"
AUDIT_OUTPUT="$EVIDENCE_ROOT/robocasa365_atomic3_rgbd_pilot_cache_audit.json"
TASK_MANIFEST=configs/tasks/robocasa365_rgbd_pilot3.json

cd "$REPO"
test -s "$ROBOCASA_EDF"
test -d "$DATA_ROOT"
test -d "$ROBOCASA_ROOT/robocasa"
test -d "$ROBOSUITE_ROOT/robosuite"
test -z "$(git status --porcelain)" || {
  echo "[FAIL] Git worktree must be clean before RGB-D P2" >&2
  git status --short >&2
  exit 2
}
mkdir -p "$EVIDENCE_ROOT" "$CACHE_ROOT"

srun --kill-on-bad-exit=1 --gpus-per-node=1 --environment="$ROBOCASA_EDF" \
  env REPO="$REPO" ROBOCASA_ROOT="$ROBOCASA_ROOT" ROBOSUITE_ROOT="$ROBOSUITE_ROOT" \
    DATA_ROOT="$DATA_ROOT" CACHE_ROOT="$CACHE_ROOT" \
    ENCODING_OUTPUT="$ENCODING_OUTPUT" MANIFEST_OUTPUT="$MANIFEST_OUTPUT" \
    AUDIT_OUTPUT="$AUDIT_OUTPUT" TASK_MANIFEST="$TASK_MANIFEST" \
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

python scripts/build_and_audit_robocasa365_depth_cache.py \
  --dataset-root "$DATA_ROOT" \
  --task-manifest "$TASK_MANIFEST" \
  --episodes-per-task 3 \
  --calibration-frame-stride 10 \
  --calibration-pixels-per-frame 4096 \
  --quantile-low 0.01 \
  --quantile-high 0.99 \
  --seed 42 \
  --height 256 \
  --width 256 \
  --fps 20 \
  --max-roundtrip-mae 3 \
  --max-channel-delta 3 \
  --cache-root "$CACHE_ROOT" \
  --encoding-output "$ENCODING_OUTPUT" \
  --manifest-output "$MANIFEST_OUTPUT" \
  --audit-output "$AUDIT_OUTPUT"
'

test -s "$ENCODING_OUTPUT"
grep -q '"encoding_name": "robocasa365_inverse_metric_global_q_v1"' "$ENCODING_OUTPUT"
test -s "$MANIFEST_OUTPUT"
grep -q '"task_count": 3' "$MANIFEST_OUTPUT"
grep -q '"episode_count": 9' "$MANIFEST_OUTPUT"
grep -q '"video_count": 27' "$MANIFEST_OUTPUT"
test -s "$AUDIT_OUTPUT"
grep -q '"ok": true' "$AUDIT_OUTPUT"
grep -q '"result": "pass"' "$AUDIT_OUTPUT"

echo "depth_encoding=$ENCODING_OUTPUT"
echo "depth_cache=$CACHE_ROOT"
echo "depth_cache_manifest=$MANIFEST_OUTPUT"
echo "depth_cache_audit=$AUDIT_OUTPUT"
echo "[PASS] Atomic RGB-D pilot encoding, cache, timing and storage audit ready"
