#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --job-name=xwam_a9_pm_cache
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/atomic9-pointmap-cache-%j.log
#SBATCH --error=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/atomic9-pointmap-cache-%j.err
#SBATCH --export=ALL

set -Eeuo pipefail

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
DEPLOY_IOPS=/iopsstor/scratch/cscs/zjingchen/terry_nys
REPO="$DEPLOY_STORE/src/xwam-robocasa365"
ROBOCASA_ROOT="$DEPLOY_STORE/src/robocasa"
ROBOSUITE_ROOT="$DEPLOY_STORE/src/robosuite"
ROBOCASA_EDF="$DEPLOY_STORE/containers/edf/robocasa365.toml"
DATA_ROOT="$DEPLOY_STORE/datasets/robocasa/v1.0/pretrain/atomic"
RATIO00_EVIDENCE="$DEPLOY_STORE/manifests/xwam/atomic9_ratio00"
EVIDENCE_ROOT="$DEPLOY_STORE/manifests/xwam/pointmap/atomic9"
CACHE_ROOT="$DEPLOY_IOPS/x-wam-pointmap-cache/robocasa365_camera_xyz_flexpi_v1/atomic"
TASK_MANIFEST=configs/tasks/robocasa365_atomic9_fastwam_overlap.json
TRAINING_MANIFEST="$RATIO00_EVIDENCE/robocasa365_atomic9_fastwam_overlap_manifest.json"
TRANSPARENT_AUDIT="$DEPLOY_STORE/manifests/xwam/pointmap/close_blender_lid_transparent/close_blender_lid_transparent_pointmap_3259701.json"
MANIFEST_OUTPUT="$EVIDENCE_ROOT/robocasa365_atomic9_pointmap_cache_manifest.json"
AUDIT_OUTPUT="$EVIDENCE_ROOT/robocasa365_atomic9_pointmap_cache_audit.json"

cd "$REPO"
test -s "$ROBOCASA_EDF"
test -s "$TASK_MANIFEST"
test -s "$TRAINING_MANIFEST"
test -s "$TRANSPARENT_AUDIT"
test -d "$DATA_ROOT"
test -d "$ROBOCASA_ROOT/robocasa"
test -d "$ROBOSUITE_ROOT/robosuite"
mkdir -p "$EVIDENCE_ROOT/tasks" "$CACHE_ROOT"
exec 9>"$EVIDENCE_ROOT/.atomic9-pointmap-cache.lock"
flock -n 9 || {
  echo "[FAIL] Another Atomic9 PointMap cache job is still running" >&2
  exit 2
}

echo "commit=$(git rev-parse HEAD)"
if test -n "$(git status --porcelain)"; then
  echo "[WARN] Git worktree is dirty; provenance is recorded but cache generation is allowed" >&2
  git status --short >&2
fi

srun --kill-on-bad-exit=1 --ntasks=4 --gpus-per-task=1 --cpus-per-task=12 \
  --gpu-bind=single:1 --environment="$ROBOCASA_EDF" \
  env REPO="$REPO" ROBOCASA_ROOT="$ROBOCASA_ROOT" ROBOSUITE_ROOT="$ROBOSUITE_ROOT" \
    DATA_ROOT="$DATA_ROOT" CACHE_ROOT="$CACHE_ROOT" EVIDENCE_ROOT="$EVIDENCE_ROOT" \
    TASK_MANIFEST="$TASK_MANIFEST" TRAINING_MANIFEST="$TRAINING_MANIFEST" \
    TRANSPARENT_AUDIT="$TRANSPARENT_AUDIT" MANIFEST_OUTPUT="$MANIFEST_OUTPUT" \
    AUDIT_OUTPUT="$AUDIT_OUTPUT" \
  bash -lc '
set -Eeuo pipefail
SHARD_INDEX="${SLURM_PROCID:?SLURM_PROCID is required}"
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

python scripts/build_and_audit_robocasa365_pointmap_atomic_cache.py \
  --dataset-root "$DATA_ROOT" \
  --task-manifest "$TASK_MANIFEST" \
  --training-manifest "$TRAINING_MANIFEST" \
  --expected-task-count 9 \
  --transparent-audit "$TRANSPARENT_AUDIT" \
  --cache-root "$CACHE_ROOT" \
  --artifact-root "$EVIDENCE_ROOT" \
  --manifest-output "$MANIFEST_OUTPUT" \
  --audit-output "$AUDIT_OUTPUT" \
  --num-shards 4 \
  --shard-index "$SHARD_INDEX" \
  --generate-only
'

srun --kill-on-bad-exit=1 --ntasks=1 --cpus-per-task=1 \
  --environment="$ROBOCASA_EDF" \
  env REPO="$REPO" DATA_ROOT="$DATA_ROOT" CACHE_ROOT="$CACHE_ROOT" \
    EVIDENCE_ROOT="$EVIDENCE_ROOT" TASK_MANIFEST="$TASK_MANIFEST" \
    TRAINING_MANIFEST="$TRAINING_MANIFEST" TRANSPARENT_AUDIT="$TRANSPARENT_AUDIT" \
    MANIFEST_OUTPUT="$MANIFEST_OUTPUT" AUDIT_OUTPUT="$AUDIT_OUTPUT" \
  bash -lc '
set -Eeuo pipefail
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
python scripts/build_and_audit_robocasa365_pointmap_atomic_cache.py \
  --dataset-root "$DATA_ROOT" \
  --task-manifest "$TASK_MANIFEST" \
  --training-manifest "$TRAINING_MANIFEST" \
  --expected-task-count 9 \
  --transparent-audit "$TRANSPARENT_AUDIT" \
  --cache-root "$CACHE_ROOT" \
  --artifact-root "$EVIDENCE_ROOT" \
  --manifest-output "$MANIFEST_OUTPUT" \
  --audit-output "$AUDIT_OUTPUT" \
  --index-only
'

for artifact in "$MANIFEST_OUTPUT" "$AUDIT_OUTPUT"; do
  test -s "$artifact"
  grep -q '"task_count": 9' "$artifact"
  grep -q '"ok": true' "$artifact"
  grep -q '"result": "pass"' "$artifact"
done

echo "pointmap_cache=$CACHE_ROOT"
echo "pointmap_cache_manifest=$MANIFEST_OUTPUT"
echo "pointmap_cache_audit=$AUDIT_OUTPUT"
echo "[PASS] Full Atomic9 PointMap cache ready; resubmit after timeout to resume four shards"
