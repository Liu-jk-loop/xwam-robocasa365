#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --job-name=xwam_cf_ab
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/close-fridge-ab-%j.log
#SBATCH --export=ALL

set -Eeuo pipefail

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
DEPLOY_CAPSCR=/capstor/scratch/cscs/zjingchen/terry_nys
DEPLOY_IOPS=/iopsstor/scratch/cscs/zjingchen/terry_nys
REPO="$DEPLOY_STORE/src/xwam-robocasa365"
EDF="$DEPLOY_STORE/containers/edf/xwam.toml"
RUNTIME_OVERLAY="$DEPLOY_IOPS/python/xwam-nvtx-0.2.15"
WANDB_OVERLAY="$DEPLOY_IOPS/python/xwam-wandb-0.23.1"
DATASET="$DEPLOY_STORE/datasets/robocasa/v1.0/pretrain/atomic/CloseFridge/20250819/lerobot"
WAN_DIR="$DEPLOY_STORE/checkpoints/xwam/wan22_5b"
PRETRAINED="$DEPLOY_STORE/checkpoints/xwam/pretrained/checkpoints/last.ckpt"
HARDWARE_CONFIG=configs/hardware/gh200x4_96gb_gbs128_single_task.yaml
DATA_CONFIG=configs/data/robocasa365_close_fridge_rgb.yaml

VARIANT="${XWAM_CF_VARIANT:-}"
case "$VARIANT" in
  ratio05)
    CLEAN_ACTION_RATIO=0.5
    EXPERIMENT_CONFIG=configs/experiment/robocasa365_close_fridge_ratio05.yaml
    EXP_NAME=close_fridge_rgb_ratio05_seed42_4gpu
    ;;
  ratio00)
    CLEAN_ACTION_RATIO=0.0
    EXPERIMENT_CONFIG=configs/experiment/robocasa365_close_fridge_ratio00.yaml
    EXP_NAME=close_fridge_rgb_ratio00_seed42_4gpu
    ;;
  *)
    echo "XWAM_CF_VARIANT must be ratio05 or ratio00" >&2
    exit 2
    ;;
esac

# 两组先训练到1000步；选出更优组后可把目标扩展到3000步，scheduler始终为3000步。
TARGET_STEPS="${XWAM_CF_TARGET_STEPS:-1000}"
case "$TARGET_STEPS" in
  1000|3000) ;;
  *)
    echo "XWAM_CF_TARGET_STEPS must be 1000 or 3000" >&2
    exit 2
    ;;
esac

EXP_ROOT="$DEPLOY_CAPSCR/experiments/xwam"
EXP_DIR="$EXP_ROOT/$EXP_NAME"
RUN_DIR="$EXP_DIR/runs"
HOT_CHECKPOINT_ROOT="$DEPLOY_IOPS/xwam_run/$EXP_NAME/checkpoints"
FINAL_CHECKPOINT_ROOT="$DEPLOY_STORE/checkpoints/xwam/$EXP_NAME/checkpoints"
HOT_QUARANTINE_ROOT="$DEPLOY_IOPS/xwam_run/$EXP_NAME/incomplete-checkpoints/close-fridge-ab-${SLURM_JOB_ID}"
FINAL_QUARANTINE_ROOT="$DEPLOY_STORE/checkpoints/xwam/$EXP_NAME/incomplete-checkpoints/close-fridge-ab-${SLURM_JOB_ID}"
WANDB_RUN_ID_FILE="$EXP_DIR/.wandb_run_id"
WANDB_PROJECT="${WANDB_PROJECT:-xwam-robocasa365}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_ROOT="$DEPLOY_IOPS/wandb/xwam/$EXP_NAME"
WANDB_CACHE_DIR="$DEPLOY_IOPS/cache/wandb"
PLAN_JSON="$RUN_DIR/close-fridge-ab-${SLURM_JOB_ID}-plan.json"
PLAN_ENV="$RUN_DIR/close-fridge-ab-${SLURM_JOB_ID}-plan.env"
RUN_ID="close-fridge-ab-${VARIANT}-${SLURM_JOB_ID}"
MAIN_LOG="$DEPLOY_STORE/logs/xwam/close-fridge-ab-${SLURM_JOB_ID}.log"
TRAIN_LOG="$DEPLOY_STORE/logs/xwam/close-fridge-ab-${VARIANT}-${SLURM_JOB_ID}-train.log"
FAILURE_REPORT="$DEPLOY_STORE/logs/xwam/close-fridge-ab-${VARIANT}-${SLURM_JOB_ID}-failure.txt"

export XWAM_FAILURE_REPORT="$FAILURE_REPORT"
export XWAM_MAIN_LOG="$MAIN_LOG"
export XWAM_TRAIN_LOG="$TRAIN_LOG"
export XWAM_PHASE=close_fridge_ab_preflight
# shellcheck source=deployment/clariden/error_trap.sh
source "$REPO/deployment/clariden/error_trap.sh"
xwam_install_err_trap

cd "$REPO"
test "${SLURM_NNODES:-0}" == 1
test -s "$EDF"
test -d "$RUNTIME_OVERLAY"
test -d "$WANDB_OVERLAY"
test -s "$DATASET/meta/info.json"
test -s "$DATASET/meta/stats.json"
test -s "$WAN_DIR/config.json"
test -s "$WAN_DIR/models_t5_umt5-xxl-enc-bf16.pth"
test -s "$WAN_DIR/Wan2.2_VAE.pth"
test -s "$PRETRAINED/checkpoint/mp_rank_00_model_states.pt"
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  xwam_fail "WANDB_API_KEY must be exported before sbatch submission"
fi
test -z "$(git status --porcelain)" || {
  git status --short >&2
  xwam_fail "Git worktree must be clean before CloseFridge A/B training"
}

mkdir -p \
  "$EXP_ROOT" \
  "$RUN_DIR" \
  "$HOT_CHECKPOINT_ROOT" \
  "$FINAL_CHECKPOINT_ROOT" \
  "$WANDB_ROOT" \
  "$WANDB_CACHE_DIR"
exec 9>"$EXP_DIR/.close-fridge-ab.lock"
flock -n 9 || {
  xwam_fail "Another job is already writing $EXP_DIR"
}

if [[ ! -s "$WANDB_RUN_ID_FILE" ]]; then
  WANDB_RUN_ID_TMP="$WANDB_RUN_ID_FILE.${SLURM_JOB_ID}.tmp"
  umask 077
  printf 'xwam-cf-%s-seed42-%s-%s\n' \
    "$VARIANT" "$(date -u +%Y%m%dT%H%M%SZ)" "$SLURM_JOB_ID" > "$WANDB_RUN_ID_TMP"
  mv "$WANDB_RUN_ID_TMP" "$WANDB_RUN_ID_FILE"
fi
read -r WANDB_RUN_ID < "$WANDB_RUN_ID_FILE"
[[ "$WANDB_RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || {
  xwam_fail "Invalid persistent W&B run ID: $WANDB_RUN_ID"
}

echo "variant=$VARIANT clean_action_ratio=$CLEAN_ACTION_RATIO"
echo "target_steps=$TARGET_STEPS scheduler_steps=3000"
echo "experiment=$EXP_NAME"
echo "rolling_checkpoints=$HOT_CHECKPOINT_ROOT (every 250 steps, max 5)"
echo "final_checkpoints=$FINAL_CHECKPOINT_ROOT"
echo "wandb_project=$WANDB_PROJECT"
echo "wandb_entity=$WANDB_ENTITY"
echo "wandb_run_id=$WANDB_RUN_ID"
echo "failure_report=$FAILURE_REPORT"

XWAM_PHASE=close_fridge_ab_planner
srun --nodes=1 --ntasks=1 --kill-on-bad-exit=1 --environment="$EDF" \
  env \
    REPO="$REPO" \
    RUNTIME_OVERLAY="$RUNTIME_OVERLAY" \
    HOT_CHECKPOINT_ROOT="$HOT_CHECKPOINT_ROOT" \
    FINAL_CHECKPOINT_ROOT="$FINAL_CHECKPOINT_ROOT" \
    HOT_QUARANTINE_ROOT="$HOT_QUARANTINE_ROOT" \
    FINAL_QUARANTINE_ROOT="$FINAL_QUARANTINE_ROOT" \
    PLAN_JSON="$PLAN_JSON" \
    PLAN_ENV="$PLAN_ENV" \
    TARGET_STEPS="$TARGET_STEPS" \
    XWAM_FAILURE_REPORT="$FAILURE_REPORT" \
    XWAM_MAIN_LOG="$MAIN_LOG" \
    XWAM_TRAIN_LOG="$TRAIN_LOG" \
  bash -lc '
set -Eeuo pipefail
export XWAM_PHASE=close_fridge_ab_planner_container
source "$REPO/deployment/clariden/error_trap.sh"
xwam_install_err_trap
cd "$REPO"
export PYTHONPATH="$RUNTIME_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
python scripts/plan_robocasa365_m6_formal_chunk.py \
  --checkpoint-root "$HOT_CHECKPOINT_ROOT" \
  --checkpoint-root "$FINAL_CHECKPOINT_ROOT" \
  --total-steps "$TARGET_STEPS" \
  --chunk-steps "$TARGET_STEPS" \
  --expected-world-size 4 \
  --output-json "$PLAN_JSON" \
  --output-env "$PLAN_ENV" \
  --quarantine-incomplete-root "$HOT_QUARANTINE_ROOT" \
  --quarantine-incomplete-root "$FINAL_QUARANTINE_ROOT"
'

test -s "$PLAN_JSON"
test -s "$PLAN_ENV"
# shellcheck disable=SC1090
source "$PLAN_ENV"
echo "completed_step=$XWAM_PLAN_COMPLETED_STEP"
echo "target_step=$XWAM_PLAN_TARGET_STEP"
echo "resume_checkpoint=$XWAM_PLAN_RESUME_CHECKPOINT"
if [[ "$XWAM_PLAN_ALREADY_COMPLETE" == 1 ]]; then
  echo "[PASS] CloseFridge $VARIANT already has a complete step $TARGET_STEPS checkpoint"
  exit 0
fi

METADATA="$RUN_DIR/${RUN_ID}_metadata.json"
RESULT="$RUN_DIR/${RUN_ID}_result.json"
EVENTS="$RUN_DIR/${RUN_ID}_checkpoint_events.jsonl"
OPTIMIZER_DIR="$RUN_DIR/${RUN_ID}_optimizer_state"
XWAM_PHASE=close_fridge_ab_training_srun
srun --nodes=1 \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --kill-on-bad-exit=1 \
  --environment="$EDF" \
  env \
    REPO="$REPO" \
    RUNTIME_OVERLAY="$RUNTIME_OVERLAY" \
    WANDB_OVERLAY="$WANDB_OVERLAY" \
    DATASET="$DATASET" \
    WAN_DIR="$WAN_DIR" \
    PRETRAINED="$PRETRAINED" \
    DATA_CONFIG="$DATA_CONFIG" \
    HARDWARE_CONFIG="$HARDWARE_CONFIG" \
    EXPERIMENT_CONFIG="$EXPERIMENT_CONFIG" \
    EXP_ROOT="$EXP_ROOT" \
    EXP_NAME="$EXP_NAME" \
    HOT_CHECKPOINT_ROOT="$HOT_CHECKPOINT_ROOT" \
    FINAL_CHECKPOINT_ROOT="$FINAL_CHECKPOINT_ROOT" \
    RUN_ID="$RUN_ID" \
    TARGET_STEP="$XWAM_PLAN_TARGET_STEP" \
    RESUME_CHECKPOINT="$XWAM_PLAN_RESUME_CHECKPOINT" \
    TRAIN_LOG="$TRAIN_LOG" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_ENTITY="$WANDB_ENTITY" \
    WANDB_RUN_ID="$WANDB_RUN_ID" \
    WANDB_DIR="$WANDB_ROOT" \
    WANDB_CACHE_DIR="$WANDB_CACHE_DIR" \
    WANDB_MODE=online \
    WANDB_RESUME=allow \
    XWAM_FAILURE_REPORT="$FAILURE_REPORT" \
    XWAM_MAIN_LOG="$MAIN_LOG" \
    XWAM_TRAIN_LOG="$TRAIN_LOG" \
  bash -lc '
set -Eeuo pipefail
NODE_TRAIN_LOG="$TRAIN_LOG"
NODE_FAILURE_REPORT="$XWAM_FAILURE_REPORT"
export XWAM_FAILURE_REPORT="$NODE_FAILURE_REPORT"
export XWAM_TRAIN_LOG="$NODE_TRAIN_LOG"
export XWAM_PHASE=close_fridge_ab_training_shell
source "$REPO/deployment/clariden/error_trap.sh"
xwam_install_err_trap
cd "$REPO"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1
unset WANDB_IDENTITY_TOKEN_FILE
export PYTHONPATH="$WANDB_OVERLAY:$RUNTIME_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"

python - <<"PY"
import torch

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 4, torch.cuda.device_count()
names = [torch.cuda.get_device_name(index) for index in range(4)]
assert all("GH200" in name.upper() for name in names), names
print("CloseFridge A/B node GPU preflight:", names)
PY

XWAM_PHASE=close_fridge_ab_wandb_preflight
python - <<"PY"
import importlib.metadata
import os
from pathlib import Path

import wandb

overlay = Path(os.environ["WANDB_OVERLAY"]).resolve()
module = Path(wandb.__file__).resolve()
assert module.is_relative_to(overlay), (module, overlay)
assert importlib.metadata.version("wandb") == "0.23.1"
assert os.environ.get("WANDB_API_KEY")
assert wandb.login(verify=True)
print("wandb authentication: api_key_env pass")
PY

export SLURM_JOB_NAME=interactive
unset SLURM_NTASKS SLURM_NTASKS_PER_NODE SLURM_PROCID SLURM_LOCALID SLURM_NODEID
export XWAM_RUN_ID="$RUN_ID"

TRAIN_ARGS=(
  scripts/train_sft.py
  model_config=configs/model/wan22_5b_robocasa365_atomic.yaml
  "data_config=$DATA_CONFIG"
  "hardware_config=$HARDWARE_CONFIG"
  "experiment_config=$EXPERIMENT_CONFIG"
  "wan_checkpoint_dir=$WAN_DIR"
  "pretrained_checkpoint=$PRETRAINED"
  "dataset.dataset_path=$DATASET"
  "exp_root=$EXP_ROOT"
  "exp_name=$EXP_NAME"
  "checkpoint_dir=$HOT_CHECKPOINT_ROOT"
  "final_checkpoint_dir=$FINAL_CHECKPOINT_ROOT"
  "trainer_max_steps=$TARGET_STEP"
  save_final_checkpoint=true
)
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  TRAIN_ARGS+=("resume_checkpoint=$RESUME_CHECKPOINT")
fi

git rev-parse HEAD
test -z "$(git status --porcelain)"
nvidia-smi
free -h
python "${TRAIN_ARGS[@]}" 2>&1 | tee "$NODE_TRAIN_LOG"
'

XWAM_PHASE=close_fridge_ab_artifact_check
test -s "$METADATA"
test -s "$RESULT"
test -s "$EVENTS"
test -d "$OPTIMIZER_DIR"
srun --nodes=1 --ntasks=1 --kill-on-bad-exit=1 --environment="$EDF" \
  env \
    RESULT="$RESULT" \
    METADATA="$METADATA" \
    EXPECTED_STEP="$XWAM_PLAN_TARGET_STEP" \
    EXPECTED_RATIO="$CLEAN_ACTION_RATIO" \
    EXPECTED_WANDB_RUN_ID="$WANDB_RUN_ID" \
  bash -lc '
set -Eeuo pipefail
python - <<"PY"
import json
import os
from pathlib import Path

result = json.loads(Path(os.environ["RESULT"]).read_text())
metadata = json.loads(Path(os.environ["METADATA"]).read_text())
expected_step = int(os.environ["EXPECTED_STEP"])
assert result["result"] == "pass", result
assert int(result["global_step"]) == expected_step, result
assert metadata["git"]["dirty"] is False, metadata["git"]
assert metadata["dataset"]["task_name"] == "CloseFridge", metadata["dataset"]
assert int(metadata["topology"]["world_size"]) == 4, metadata["topology"]
assert int(metadata["training"]["num_training_steps"]) == 3000, metadata["training"]
wandb = metadata["tracking"]["wandb"]
assert wandb["run_id"] == os.environ["EXPECTED_WANDB_RUN_ID"], wandb
assert wandb["auth"] == "api_key_env", wandb
resolved = Path(metadata["resolved_config"]).read_text()
expected_ratio = os.environ["EXPECTED_RATIO"]
assert f"clean_action_ratio: {expected_ratio}" in resolved
assert "global_batch_size: 128" in resolved
assert "deepspeed_stage: 1" in resolved
assert "use_gradient_checkpointing: true" in resolved
assert "save_interval: 250" in resolved
checkpoint = Path(result["last_checkpoint"])
assert checkpoint.is_dir(), checkpoint
print("CloseFridge A/B artifact check: pass")
print("checkpoint:", checkpoint)
PY
'

echo "metadata=$METADATA"
echo "result=$RESULT"
echo "train_log=$TRAIN_LOG"
echo "[PASS] CloseFridge $VARIANT reached step $XWAM_PLAN_TARGET_STEP"
