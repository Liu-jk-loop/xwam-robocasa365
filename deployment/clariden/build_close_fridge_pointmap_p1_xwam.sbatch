#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --job-name=xwam_cf_pm_p1
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/close-fridge-pointmap-p1-%j.log
#SBATCH --error=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/close-fridge-pointmap-p1-%j.err

set -Eeuo pipefail

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
DEPLOY_IOPS=/iopsstor/scratch/cscs/zjingchen/terry_nys
REPO="$DEPLOY_STORE/src/xwam-robocasa365"
ROBOCASA_ROOT="$DEPLOY_STORE/src/robocasa"
ROBOSUITE_ROOT="$DEPLOY_STORE/src/robosuite"
ROBOCASA_EDF="$DEPLOY_STORE/containers/edf/robocasa365.toml"
DATA_ROOT="$DEPLOY_STORE/datasets/robocasa/v1.0/pretrain/atomic"
EVIDENCE_ROOT="$DEPLOY_STORE/manifests/xwam/pointmap/close_fridge_p1"
CACHE_ROOT="$DEPLOY_IOPS/x-wam-pointmap-cache/robocasa365_camera_xyz_flexpi_v1/atomic"
MANIFEST_OUTPUT="$EVIDENCE_ROOT/close_fridge_pointmap_p1_manifest.json"
AUDIT_OUTPUT="$EVIDENCE_ROOT/close_fridge_pointmap_p1_audit.json"
TASK_MANIFEST=configs/tasks/robocasa365_rgbd_close_fridge.json
TRANSPARENT_AUDIT="$DEPLOY_STORE/manifests/xwam/pointmap/close_blender_lid_transparent/close_blender_lid_transparent_pointmap_3259701.json"

cd "$REPO"
test -s "$ROBOCASA_EDF"
test -d "$DATA_ROOT/CloseFridge/20250819/lerobot"
test -d "$ROBOCASA_ROOT/robocasa"
test -d "$ROBOSUITE_ROOT/robosuite"
test -s "$TRANSPARENT_AUDIT"
mkdir -p "$EVIDENCE_ROOT" "$CACHE_ROOT"

echo "commit=$(git rev-parse HEAD)"
if test -n "$(git status --porcelain)"; then
  echo "[WARN] Git worktree is dirty; provenance will be recorded but generation is allowed" >&2
  git status --short >&2
fi

srun --kill-on-bad-exit=1 --gpus-per-node=1 --environment="$ROBOCASA_EDF" \
  env REPO="$REPO" ROBOCASA_ROOT="$ROBOCASA_ROOT" ROBOSUITE_ROOT="$ROBOSUITE_ROOT" \
    DATA_ROOT="$DATA_ROOT" CACHE_ROOT="$CACHE_ROOT" \
    MANIFEST_OUTPUT="$MANIFEST_OUTPUT" AUDIT_OUTPUT="$AUDIT_OUTPUT" \
    TASK_MANIFEST="$TASK_MANIFEST" TRANSPARENT_AUDIT="$TRANSPARENT_AUDIT" \
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

python scripts/build_and_audit_robocasa365_pointmap_cache.py \
  --dataset-root "$DATA_ROOT" \
  --task-manifest "$TASK_MANIFEST" \
  --task-name CloseFridge \
  --transparent-audit "$TRANSPARENT_AUDIT" \
  --cache-root "$CACHE_ROOT" \
  --source-height 256 \
  --source-width 256 \
  --target-height 256 \
  --target-width 320 \
  --episodes-per-task 0 \
  --expected-episode-count 106 \
  --expected-array-count 318 \
  --progress-interval 25 \
  --manifest-output "$MANIFEST_OUTPUT" \
  --audit-output "$AUDIT_OUTPUT"
'

python - "$MANIFEST_OUTPUT" "$AUDIT_OUTPUT" <<'PY'
import json
import sys

manifest_path, audit_path = sys.argv[1:]
with open(manifest_path, encoding="utf-8") as handle:
    manifest = json.load(handle)
with open(audit_path, encoding="utf-8") as handle:
    audit = json.load(handle)
assert manifest["ok"] is True and manifest["result"] == "pass", manifest["errors"]
assert manifest["scope"] == "atomic_only"
assert manifest["task_name"] == "CloseFridge"
assert manifest["episode_count"] == 106
assert manifest["array_count"] == 318
assert manifest["render_policy"] == "use_forced_opaque_depth_for_pointmap_keep_original_rgb"
assert manifest["transparent_policy_evidence"]["conclusion"] == "forced_opaque_required"
assert audit["ok"] is True and audit["result"] == "pass", audit["errors"]
assert audit["episode_count"] == 106
assert audit["array_count"] == 318
assert all(audit["checks"].values()), audit["checks"]
print("resumed_camera_count=" + str(manifest["resumed_camera_count"]))
print("regenerated_camera_count=" + str(manifest["regenerated_camera_count"]))
print("total_cache_bytes=" + str(audit["total_cache_bytes"]))
PY

echo "pointmap_cache=$CACHE_ROOT"
echo "pointmap_cache_manifest=$MANIFEST_OUTPUT"
echo "pointmap_cache_audit=$AUDIT_OUTPUT"
echo "[PASS] CloseFridge resumable forced-opaque PointMap P1 cache"
