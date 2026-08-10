#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --job-name=xwam_train4_resume
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/train4-resume-%j.log

set -euo pipefail

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
DEPLOY_CAPSCR=/capstor/scratch/cscs/zjingchen/terry_nys
DEPLOY_IOPS=/iopsstor/scratch/cscs/zjingchen/terry_nys
EDF="$DEPLOY_STORE/containers/edf/xwam.toml"
RUNTIME_OVERLAY="$DEPLOY_IOPS/python/xwam-nvtx-0.2.15"
DATASET="$DEPLOY_STORE/datasets/robocasa/v1.0/pretrain/atomic/CloseFridge/20250819/lerobot"
WAN_DIR="$DEPLOY_STORE/checkpoints/xwam/wan22_5b"
PRETRAINED="$DEPLOY_STORE/checkpoints/xwam/pretrained/checkpoints/last.ckpt"
EXP_ROOT="$DEPLOY_CAPSCR/experiments/xwam"
SOURCE_JOB_ID="${XWAM_INITIAL_JOB_ID:-$SLURM_JOB_ID}"
[[ "$SOURCE_JOB_ID" =~ ^[0-9]+$ ]] || {
  echo "XWAM_INITIAL_JOB_ID must be a numeric Slurm job ID" >&2
  exit 2
}
REUSE_INITIAL=false
if [[ -n "${XWAM_INITIAL_JOB_ID:-}" ]]; then
  REUSE_INITIAL=true
fi
EXP_NAME="clariden_close_fridge_4gpu_resume_${SOURCE_JOB_ID}"
EXP_DIR="$EXP_ROOT/$EXP_NAME"
RUN_DIR="$EXP_DIR/runs"
INITIAL_RUN_ID="clariden4-${SOURCE_JOB_ID}-initial"
RESUMED_RUN_ID="clariden4-${SLURM_JOB_ID}-resumed"
INITIAL_LOG="$DEPLOY_STORE/logs/xwam/train4-resume-${SOURCE_JOB_ID}-initial.log"
RESUMED_LOG="$DEPLOY_STORE/logs/xwam/train4-resume-${SLURM_JOB_ID}-resumed.log"
AUDIT="$DEPLOY_STORE/logs/xwam/train4-resume-${SLURM_JOB_ID}-audit.json"
CHECKPOINT_RECORD="$RUN_DIR/${INITIAL_RUN_ID}_checkpoint_path.txt"

INITIAL_METADATA="$RUN_DIR/${INITIAL_RUN_ID}_metadata.json"
INITIAL_RESULT="$RUN_DIR/${INITIAL_RUN_ID}_result.json"
INITIAL_EVENTS="$RUN_DIR/${INITIAL_RUN_ID}_checkpoint_events.jsonl"
INITIAL_OPTIMIZER="$RUN_DIR/${INITIAL_RUN_ID}_optimizer_state"
RESUMED_METADATA="$RUN_DIR/${RESUMED_RUN_ID}_metadata.json"
RESUMED_RESULT="$RUN_DIR/${RESUMED_RUN_ID}_result.json"
RESUMED_EVENTS="$RUN_DIR/${RESUMED_RUN_ID}_checkpoint_events.jsonl"
RESUMED_OPTIMIZER="$RUN_DIR/${RESUMED_RUN_ID}_optimizer_state"

test -s "$EDF"
test -d "$RUNTIME_OVERLAY"
test -s "$DATASET/meta/info.json"
test -s "$WAN_DIR/config.json"
test -s "$WAN_DIR/models_t5_umt5-xxl-enc-bf16.pth"
test -s "$WAN_DIR/Wan2.2_VAE.pth"
test -s "$PRETRAINED/checkpoint/mp_rank_00_model_states.pt"
mkdir -p "$EXP_ROOT" "$RUN_DIR" "$(dirname "$AUDIT")"

if [[ "$REUSE_INITIAL" == true ]]; then
  echo "[INFO] Reusing initial phase artifacts from Slurm job $SOURCE_JOB_ID"
else
  srun --kill-on-bad-exit=1 --environment="$EDF" bash -lc "
set -euo pipefail
export MAX_JOBS=8
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# srun只启动一个容器任务；四个本地rank由Lightning按devices=4派生。
# 清除step级单任务拓扑，避免Lightning把SLURM_NTASKS=1误当训练world size。
export SLURM_JOB_NAME=interactive
unset SLURM_NTASKS SLURM_NTASKS_PER_NODE SLURM_PROCID SLURM_LOCALID SLURM_NODEID
export XWAM_RUN_ID='$INITIAL_RUN_ID'
export PYTHONPATH='$RUNTIME_OVERLAY':\${PYTHONPATH:-}

python - <<'PY'
import importlib.metadata
from pathlib import Path

import nvtx
import torch

overlay = Path('$RUNTIME_OVERLAY').resolve()
module = Path(nvtx.__file__).resolve()
assert module.is_relative_to(overlay), (module, overlay)
assert importlib.metadata.version('nvtx') == '0.2.15'
domain = nvtx.get_domain('xwam-clariden-deepspeed-probe')
domain.push_range(message='probe', category=None)
domain.pop_range()
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 4
names = [torch.cuda.get_device_name(index) for index in range(4)]
capabilities = [torch.cuda.get_device_capability(index) for index in range(4)]
assert all('GH200' in name.upper() for name in names), names
assert capabilities == [(9, 0)] * 4, capabilities
print('Clariden 4-GPU preflight:', names, capabilities)
PY

nvidia-smi
free -h
git rev-parse HEAD
git status --short

python scripts/train_sft.py \\
  model_config=configs/model/wan22_5b_robocasa365_atomic.yaml \\
  data_config=configs/data/robocasa365.yaml \\
  hardware_config=configs/hardware/gh200x4_96gb_resume_debug.yaml \\
  experiment_config=configs/experiment/robocasa365_clariden_4gpu_resume_gate.yaml \\
  wan_checkpoint_dir='$WAN_DIR' \\
  pretrained_checkpoint='$PRETRAINED' \\
  dataset.dataset_path='$DATASET' \\
  exp_root='$EXP_ROOT' \\
  exp_name='$EXP_NAME'
" 2>&1 | tee "$INITIAL_LOG"
fi

test -s "$INITIAL_METADATA"
test -s "$INITIAL_RESULT"
test -s "$INITIAL_EVENTS"

srun --kill-on-bad-exit=1 --environment="$EDF" \
  env INITIAL_RESULT="$INITIAL_RESULT" CHECKPOINT_RECORD="$CHECKPOINT_RECORD" \
  bash -lc '
set -euo pipefail
python scripts/resolve_clariden_initial_checkpoint.py \
  --result "$INITIAL_RESULT" \
  --output "$CHECKPOINT_RECORD"
'
test -s "$CHECKPOINT_RECORD"
IFS= read -r CHECKPOINT < "$CHECKPOINT_RECORD"
test -d "$CHECKPOINT"

srun --kill-on-bad-exit=1 --environment="$EDF" bash -lc "
set -euo pipefail
export MAX_JOBS=8
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SLURM_JOB_NAME=interactive
unset SLURM_NTASKS SLURM_NTASKS_PER_NODE SLURM_PROCID SLURM_LOCALID SLURM_NODEID
export XWAM_RUN_ID='$RESUMED_RUN_ID'
export PYTHONPATH='$RUNTIME_OVERLAY':\${PYTHONPATH:-}

python scripts/train_sft.py \\
  model_config=configs/model/wan22_5b_robocasa365_atomic.yaml \\
  data_config=configs/data/robocasa365.yaml \\
  hardware_config=configs/hardware/gh200x4_96gb_resume_debug.yaml \\
  experiment_config=configs/experiment/robocasa365_clariden_4gpu_resume_gate.yaml \\
  wan_checkpoint_dir='$WAN_DIR' \\
  pretrained_checkpoint='$PRETRAINED' \\
  dataset.dataset_path='$DATASET' \\
  exp_root='$EXP_ROOT' \\
  exp_name='$EXP_NAME' \\
  trainer_max_steps=4 \\
  resume_checkpoint='$CHECKPOINT'
" 2>&1 | tee "$RESUMED_LOG"

test -s "$RESUMED_METADATA"
test -s "$RESUMED_RESULT"
test -s "$RESUMED_EVENTS"

srun --kill-on-bad-exit=1 --environment="$EDF" \
  env \
    INITIAL_METADATA="$INITIAL_METADATA" \
    INITIAL_RESULT="$INITIAL_RESULT" \
    INITIAL_EVENTS="$INITIAL_EVENTS" \
    INITIAL_OPTIMIZER="$INITIAL_OPTIMIZER" \
    INITIAL_LOG="$INITIAL_LOG" \
    RESUMED_METADATA="$RESUMED_METADATA" \
    RESUMED_RESULT="$RESUMED_RESULT" \
    RESUMED_EVENTS="$RESUMED_EVENTS" \
    RESUMED_OPTIMIZER="$RESUMED_OPTIMIZER" \
    RESUMED_LOG="$RESUMED_LOG" \
    AUDIT="$AUDIT" \
    ALLOW_ORCHESTRATION_COMMIT_DELTA="$REUSE_INITIAL" \
  bash -lc '
set -euo pipefail
AUDIT_COMMIT_ARGS=()
if [[ "$ALLOW_ORCHESTRATION_COMMIT_DELTA" == true ]]; then
  AUDIT_COMMIT_ARGS+=(--allow-orchestration-only-commit-delta)
fi
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
  --output "$AUDIT" \
  "${AUDIT_COMMIT_ARGS[@]}"
'

echo "initial_job_id=$SOURCE_JOB_ID"
echo "initial_result=$INITIAL_RESULT"
echo "resume_checkpoint=$CHECKPOINT"
echo "resumed_result=$RESUMED_RESULT"
echo "audit=$AUDIT"
echo "[PASS] X-WAM Clariden 4xGH200 step 2 to 4 resume gate"
