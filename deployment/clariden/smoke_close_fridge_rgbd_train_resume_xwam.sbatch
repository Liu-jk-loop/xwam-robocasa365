#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --job-name=xwam_cf_rgbd_gate
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/close-fridge-rgbd-gate-%j.log
#SBATCH --error=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/close-fridge-rgbd-gate-%j.err

set -Eeuo pipefail

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
DEPLOY_CAPSCR=/capstor/scratch/cscs/zjingchen/terry_nys
DEPLOY_IOPS=/iopsstor/scratch/cscs/zjingchen/terry_nys
REPO="$DEPLOY_STORE/src/xwam-robocasa365"
EDF="$DEPLOY_STORE/containers/edf/xwam.toml"
RUNTIME_OVERLAY="$DEPLOY_IOPS/python/xwam-nvtx-0.2.15"
DATASET="$DEPLOY_STORE/datasets/robocasa/v1.0/pretrain/atomic/CloseFridge/20250819/lerobot"
DEPTH_CACHE_ROOT="$DEPLOY_IOPS/x-wam-depth-cache/robocasa365_inverse_metric_global_q_v1/atomic"
DEPTH_ENCODING="$DEPLOY_STORE/manifests/xwam/atomic9_rgbd/robocasa365_inverse_metric_global_q_v1.json"
DEPTH_MANIFEST="$DEPLOY_STORE/manifests/xwam/atomic9_rgbd/robocasa365_close_fridge_rgbd_cache_manifest.json"
WAN_DIR="$DEPLOY_STORE/checkpoints/xwam/wan22_5b"
PRETRAINED="$DEPLOY_STORE/checkpoints/xwam/pretrained/checkpoints/last.ckpt"
EXP_ROOT="$DEPLOY_CAPSCR/experiments/xwam"
EXP_NAME="close_fridge_rgbd_resume_gate_${SLURM_JOB_ID}"
EXP_DIR="$EXP_ROOT/$EXP_NAME"
RUN_DIR="$EXP_DIR/runs"
INITIAL_RUN_ID="close-fridge-rgbd-${SLURM_JOB_ID}-initial"
RESUMED_RUN_ID="close-fridge-rgbd-${SLURM_JOB_ID}-resumed"
BATCH_REPORT="$DEPLOY_STORE/manifests/xwam/atomic9_rgbd/close_fridge_rgbd_batch_${SLURM_JOB_ID}.json"
INITIAL_LOG="$DEPLOY_STORE/logs/xwam/close-fridge-rgbd-${SLURM_JOB_ID}-initial.log"
RESUMED_LOG="$DEPLOY_STORE/logs/xwam/close-fridge-rgbd-${SLURM_JOB_ID}-resumed.log"
AUDIT="$DEPLOY_STORE/manifests/xwam/atomic9_rgbd/close_fridge_rgbd_resume_${SLURM_JOB_ID}.json"
CHECKPOINT_RECORD="$RUN_DIR/${INITIAL_RUN_ID}_checkpoint_path.txt"

INITIAL_METADATA="$RUN_DIR/${INITIAL_RUN_ID}_metadata.json"
INITIAL_RESULT="$RUN_DIR/${INITIAL_RUN_ID}_result.json"
INITIAL_EVENTS="$RUN_DIR/${INITIAL_RUN_ID}_checkpoint_events.jsonl"
INITIAL_OPTIMIZER="$RUN_DIR/${INITIAL_RUN_ID}_optimizer_state"
RESUMED_METADATA="$RUN_DIR/${RESUMED_RUN_ID}_metadata.json"
RESUMED_RESULT="$RUN_DIR/${RESUMED_RUN_ID}_result.json"
RESUMED_EVENTS="$RUN_DIR/${RESUMED_RUN_ID}_checkpoint_events.jsonl"
RESUMED_OPTIMIZER="$RUN_DIR/${RESUMED_RUN_ID}_optimizer_state"

cd "$REPO"
test -s "$EDF"
test -d "$RUNTIME_OVERLAY"
test -s "$DATASET/meta/info.json"
test -s "$DEPTH_ENCODING"
test -s "$DEPTH_MANIFEST"
grep -q '"ok": true' "$DEPTH_MANIFEST"
grep -q '"episode_count": 106' "$DEPTH_MANIFEST"
grep -q '"video_count": 318' "$DEPTH_MANIFEST"
test -s "$WAN_DIR/config.json"
test -s "$WAN_DIR/models_t5_umt5-xxl-enc-bf16.pth"
test -s "$WAN_DIR/Wan2.2_VAE.pth"
test -s "$PRETRAINED/checkpoint/mp_rank_00_model_states.pt"
test -z "$(git status --porcelain)" || {
  echo "[FAIL] Git worktree must be clean before RGB-D batch/train gate" >&2
  git status --short >&2
  exit 2
}
mkdir -p "$EXP_ROOT" "$RUN_DIR" "$(dirname "$BATCH_REPORT")" "$(dirname "$INITIAL_LOG")"

echo "[RGBD-P3] phase=batch_audit"
srun --kill-on-bad-exit=1 --environment="$EDF" \
  env REPO="$REPO" DATASET="$DATASET" DEPTH_CACHE_ROOT="$DEPTH_CACHE_ROOT" \
    DEPTH_ENCODING="$DEPTH_ENCODING" DEPTH_MANIFEST="$DEPTH_MANIFEST" \
    BATCH_REPORT="$BATCH_REPORT" \
  bash -lc '
set -Eeuo pipefail
cd "$REPO"
export PYTHONUNBUFFERED=1
python scripts/audit_robocasa365_batch.py \
  --dataset "$DATASET" \
  --task-name CloseFridge \
  --task-manifest configs/tasks/robocasa365_atomic_seen.json \
  --index 0 \
  --sequence-length 9 \
  --frame-skip 4 \
  --action-skip 1 \
  --video-height 256 \
  --video-width 320 \
  --depth-cache-root "$DEPTH_CACHE_ROOT" \
  --depth-encoding-path "$DEPTH_ENCODING" \
  --depth-cache-manifest "$DEPTH_MANIFEST" \
  --num-workers 0 \
  --output "$BATCH_REPORT"
'
test -s "$BATCH_REPORT"
grep -q '"ok": true' "$BATCH_REPORT"
grep -q '"result": "pass"' "$BATCH_REPORT"

echo "[RGBD-P3] phase=initial_step_0_to_2"
srun --kill-on-bad-exit=1 --environment="$EDF" \
  env REPO="$REPO" RUNTIME_OVERLAY="$RUNTIME_OVERLAY" DATASET="$DATASET" \
    DEPTH_CACHE_ROOT="$DEPTH_CACHE_ROOT" DEPTH_ENCODING="$DEPTH_ENCODING" \
    DEPTH_MANIFEST="$DEPTH_MANIFEST" WAN_DIR="$WAN_DIR" PRETRAINED="$PRETRAINED" \
    EXP_ROOT="$EXP_ROOT" EXP_NAME="$EXP_NAME" RUN_ID="$INITIAL_RUN_ID" \
  bash -lc '
set -Eeuo pipefail
cd "$REPO"
export MAX_JOBS=8
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1
export SLURM_JOB_NAME=interactive
unset SLURM_NTASKS SLURM_NTASKS_PER_NODE SLURM_PROCID SLURM_LOCALID SLURM_NODEID
export XWAM_RUN_ID="$RUN_ID"
export PYTHONPATH="$RUNTIME_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"

python - <<"PY"
import importlib.metadata
import os
from pathlib import Path
import nvtx
import torch

overlay = Path(os.environ["RUNTIME_OVERLAY"]).resolve()
assert Path(nvtx.__file__).resolve().is_relative_to(overlay)
assert importlib.metadata.version("nvtx") == "0.2.15"
assert torch.cuda.is_available() and torch.cuda.device_count() == 4
names = [torch.cuda.get_device_name(index) for index in range(4)]
assert all("GH200" in name.upper() for name in names), names
print("RGB-D 4-GPU preflight:", names)
PY

git rev-parse HEAD
test -z "$(git status --porcelain)"
python scripts/train_sft.py \
  model_config=configs/model/wan22_5b_robocasa365_atomic_rgbd.yaml \
  data_config=configs/data/robocasa365_close_fridge_rgbd.yaml \
  hardware_config=configs/hardware/gh200x4_96gb_resume_debug.yaml \
  experiment_config=configs/experiment/robocasa365_close_fridge_rgbd_resume_gate.yaml \
  "wan_checkpoint_dir=$WAN_DIR" \
  "pretrained_checkpoint=$PRETRAINED" \
  "dataset.dataset_path=$DATASET" \
  "dataset.depth_cache_root=$DEPTH_CACHE_ROOT" \
  "dataset.depth_encoding_path=$DEPTH_ENCODING" \
  "dataset.depth_cache_manifest=$DEPTH_MANIFEST" \
  "exp_root=$EXP_ROOT" \
  "exp_name=$EXP_NAME"
' 2>&1 | tee "$INITIAL_LOG"

test -s "$INITIAL_METADATA"
test -s "$INITIAL_RESULT"
test -s "$INITIAL_EVENTS"
srun --kill-on-bad-exit=1 --environment="$EDF" \
  env INITIAL_RESULT="$INITIAL_RESULT" CHECKPOINT_RECORD="$CHECKPOINT_RECORD" \
  bash -lc '
set -Eeuo pipefail
python scripts/resolve_clariden_initial_checkpoint.py \
  --result "$INITIAL_RESULT" \
  --output "$CHECKPOINT_RECORD"
'
test -s "$CHECKPOINT_RECORD"
IFS= read -r CHECKPOINT < "$CHECKPOINT_RECORD"
test -d "$CHECKPOINT"

echo "[RGBD-P3] phase=resume_step_2_to_4 checkpoint=$CHECKPOINT"
srun --kill-on-bad-exit=1 --environment="$EDF" \
  env REPO="$REPO" RUNTIME_OVERLAY="$RUNTIME_OVERLAY" DATASET="$DATASET" \
    DEPTH_CACHE_ROOT="$DEPTH_CACHE_ROOT" DEPTH_ENCODING="$DEPTH_ENCODING" \
    DEPTH_MANIFEST="$DEPTH_MANIFEST" WAN_DIR="$WAN_DIR" PRETRAINED="$PRETRAINED" \
    EXP_ROOT="$EXP_ROOT" EXP_NAME="$EXP_NAME" RUN_ID="$RESUMED_RUN_ID" \
    CHECKPOINT="$CHECKPOINT" \
  bash -lc '
set -Eeuo pipefail
cd "$REPO"
export MAX_JOBS=8
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1
export SLURM_JOB_NAME=interactive
unset SLURM_NTASKS SLURM_NTASKS_PER_NODE SLURM_PROCID SLURM_LOCALID SLURM_NODEID
export XWAM_RUN_ID="$RUN_ID"
export PYTHONPATH="$RUNTIME_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"

python scripts/train_sft.py \
  model_config=configs/model/wan22_5b_robocasa365_atomic_rgbd.yaml \
  data_config=configs/data/robocasa365_close_fridge_rgbd.yaml \
  hardware_config=configs/hardware/gh200x4_96gb_resume_debug.yaml \
  experiment_config=configs/experiment/robocasa365_close_fridge_rgbd_resume_gate.yaml \
  "wan_checkpoint_dir=$WAN_DIR" \
  "pretrained_checkpoint=$PRETRAINED" \
  "dataset.dataset_path=$DATASET" \
  "dataset.depth_cache_root=$DEPTH_CACHE_ROOT" \
  "dataset.depth_encoding_path=$DEPTH_ENCODING" \
  "dataset.depth_cache_manifest=$DEPTH_MANIFEST" \
  "exp_root=$EXP_ROOT" \
  "exp_name=$EXP_NAME" \
  trainer_max_steps=4 \
  "resume_checkpoint=$CHECKPOINT"
' 2>&1 | tee "$RESUMED_LOG"

test -s "$RESUMED_METADATA"
test -s "$RESUMED_RESULT"
test -s "$RESUMED_EVENTS"

echo "[RGBD-P3] phase=resume_audit"
srun --kill-on-bad-exit=1 --environment="$EDF" \
  env REPO="$REPO" INITIAL_METADATA="$INITIAL_METADATA" INITIAL_RESULT="$INITIAL_RESULT" \
    INITIAL_EVENTS="$INITIAL_EVENTS" INITIAL_OPTIMIZER="$INITIAL_OPTIMIZER" \
    INITIAL_LOG="$INITIAL_LOG" RESUMED_METADATA="$RESUMED_METADATA" \
    RESUMED_RESULT="$RESUMED_RESULT" RESUMED_EVENTS="$RESUMED_EVENTS" \
    RESUMED_OPTIMIZER="$RESUMED_OPTIMIZER" RESUMED_LOG="$RESUMED_LOG" AUDIT="$AUDIT" \
  bash -lc '
set -Eeuo pipefail
cd "$REPO"
python scripts/audit_clariden_4gpu_resume.py \
  --initial-metadata "$INITIAL_METADATA" \
  --initial-result "$INITIAL_RESULT" \
  --initial-events "$INITIAL_EVENTS" \
  --initial-optimizer-dir "$INITIAL_OPTIMIZER" \
  --initial-log "$INITIAL_LOG" \
  --resumed-metadata "$RESUMED_METADATA" \
  --resumed-result "$RESUMED_RESULT" \
  --resumed-events "$RESUMED_EVENTS" \
  --resumed-optimizer-dir "$RESUMED_OPTIMIZER" \
  --resumed-log "$RESUMED_LOG" \
  --expect-depth \
  --output "$AUDIT"
'

test -s "$AUDIT"
grep -q '"ok": true' "$AUDIT"
grep -q '"result": "pass"' "$AUDIT"
echo "batch_report=$BATCH_REPORT"
echo "initial_result=$INITIAL_RESULT"
echo "resume_checkpoint=$CHECKPOINT"
echo "resumed_result=$RESUMED_RESULT"
echo "resume_audit=$AUDIT"
echo "[PASS] CloseFridge RGB-D loader, depth loss, optimizer update and resume gate"
