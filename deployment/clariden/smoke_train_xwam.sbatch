#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --job-name=xwam_train1
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/train1-%j.log

set -euo pipefail

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
DEPLOY_CAPSCR=/capstor/scratch/cscs/zjingchen/terry_nys
EDF="$DEPLOY_STORE/containers/edf/xwam.toml"
DATASET="$DEPLOY_STORE/datasets/robocasa/v1.0/pretrain/atomic/CloseFridge/20250819/lerobot"
WAN_DIR="$DEPLOY_STORE/checkpoints/xwam/wan22_5b"
PRETRAINED="$DEPLOY_STORE/checkpoints/xwam/pretrained/checkpoints/last.ckpt"
EXP_ROOT="$DEPLOY_CAPSCR/experiments/xwam"
EXP_NAME="clariden_close_fridge_single_step_${SLURM_JOB_ID}"
RUN_ID="clariden-${SLURM_JOB_ID}"
EXP_DIR="$EXP_ROOT/$EXP_NAME"
RESULT="$EXP_DIR/runs/${RUN_ID}_result.json"
INITIALIZATION_REPORT="$EXP_DIR/runs/${RUN_ID}_checkpoint_initialization_rank_000.json"

test -s "$EDF"
test -s "$DATASET/meta/info.json"
test -s "$WAN_DIR/config.json"
test -s "$WAN_DIR/models_t5_umt5-xxl-enc-bf16.pth"
test -s "$WAN_DIR/Wan2.2_VAE.pth"
test -s "$PRETRAINED/checkpoint/mp_rank_00_model_states.pt"
mkdir -p "$EXP_ROOT"

srun --environment="$EDF" bash -lc "
set -euo pipefail
export MAX_JOBS=8
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export XWAM_RUN_ID='$RUN_ID'

nvidia-smi
free -h
git rev-parse HEAD
git status --short

python - <<'PY'
import torch

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
name = torch.cuda.get_device_name(0)
assert 'GH200' in name.upper(), name
assert torch.cuda.get_device_capability(0) == (9, 0)
print('Clariden smoke GPU:', name)
PY

python scripts/train_sft.py \\
  model_config=configs/model/wan22_5b_robocasa365_atomic.yaml \\
  data_config=configs/data/robocasa365.yaml \\
  hardware_config=configs/hardware/gh200_96gb_debug.yaml \\
  experiment_config=configs/experiment/robocasa365_clariden_single_step.yaml \\
  wan_checkpoint_dir='$WAN_DIR' \\
  pretrained_checkpoint='$PRETRAINED' \\
  dataset.dataset_path='$DATASET' \\
  exp_root='$EXP_ROOT' \\
  exp_name='$EXP_NAME'
"

test -s "$RESULT"
test -s "$INITIALIZATION_REPORT"
grep -q '"result": "pass"' "$RESULT"
grep -q '"global_step": 1' "$RESULT"
grep -q '"error": null' "$RESULT"
grep -q '"result": "pass"' "$INITIALIZATION_REPORT"

echo "result=$RESULT"
echo "initialization_report=$INITIALIZATION_REPORT"
echo "[PASS] X-WAM Clariden one-step train"
