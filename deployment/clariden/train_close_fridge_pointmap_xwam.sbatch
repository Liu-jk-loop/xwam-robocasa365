#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --job-name=xwam_cf_pm8
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/close-fridge-pointmap-formal-8gpu-%j.log
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
POINTMAP_CACHE_ROOT="$DEPLOY_IOPS/x-wam-pointmap-cache/robocasa365_camera_xyz_flexpi_v1/atomic"
POINTMAP_MANIFEST="$DEPLOY_STORE/manifests/xwam/pointmap/close_fridge_p1/close_fridge_pointmap_p1_manifest.json"
POINTMAP_AUDIT="$DEPLOY_STORE/manifests/xwam/pointmap/close_fridge_p1/close_fridge_pointmap_p1_audit.json"
POINTMAP_GATE_AUDIT="${XWAM_POINTMAP_GATE_AUDIT:-$DEPLOY_STORE/manifests/xwam/pointmap/close_fridge_training/close_fridge_pointmap_resume_gate.json}"
WAN_DIR="$DEPLOY_STORE/checkpoints/xwam/wan22_5b"
PRETRAINED="$DEPLOY_STORE/checkpoints/xwam/pretrained/checkpoints/last.ckpt"
MODEL_CONFIG=configs/model/wan22_5b_robocasa365_atomic_pointmap.yaml
DATA_CONFIG=configs/data/robocasa365_close_fridge_pointmap_formal.yaml
HARDWARE_CONFIG=configs/hardware/gh200x8_96gb_gbs128_pointmap_single_task.yaml
EXPERIMENT_CONFIG=configs/experiment/robocasa365_close_fridge_pointmap_formal.yaml

TOTAL_STEPS=1500
EXP_ROOT="$DEPLOY_CAPSCR/experiments/xwam"
EXP_NAME=close_fridge_pointmap_aux_ratio00_seed42_8gpu
EXP_DIR="$EXP_ROOT/$EXP_NAME"
RUN_DIR="$EXP_DIR/runs"
HOT_CHECKPOINT_ROOT="$DEPLOY_IOPS/xwam_run/$EXP_NAME/checkpoints"
DURABLE_CHECKPOINT_ROOT="$DEPLOY_STORE/checkpoints/xwam/$EXP_NAME/checkpoints"
HOT_QUARANTINE_ROOT="$DEPLOY_IOPS/xwam_run/$EXP_NAME/incomplete-checkpoints/formal-${SLURM_JOB_ID}"
DURABLE_QUARANTINE_ROOT="$DEPLOY_STORE/checkpoints/xwam/$EXP_NAME/incomplete-checkpoints/formal-${SLURM_JOB_ID}"
WANDB_RUN_ID_FILE="$EXP_DIR/.wandb_run_id"
WANDB_PROJECT="${WANDB_PROJECT:-xwam-robocasa365}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_ROOT="$DEPLOY_IOPS/wandb/xwam/$EXP_NAME"
WANDB_CACHE_DIR="$DEPLOY_IOPS/cache/wandb"
PLAN_JSON="$RUN_DIR/close-fridge-pointmap-formal-8gpu-${SLURM_JOB_ID}-plan.json"
PLAN_ENV="$RUN_DIR/close-fridge-pointmap-formal-8gpu-${SLURM_JOB_ID}-plan.env"
RUN_ID="close-fridge-pointmap-formal-8gpu-${SLURM_JOB_ID}"
MAIN_LOG="$DEPLOY_STORE/logs/xwam/close-fridge-pointmap-formal-8gpu-${SLURM_JOB_ID}.log"
TRAIN_LOG="$DEPLOY_STORE/logs/xwam/close-fridge-pointmap-formal-8gpu-${SLURM_JOB_ID}-train.log"
FAILURE_REPORT="$DEPLOY_STORE/logs/xwam/close-fridge-pointmap-formal-8gpu-${SLURM_JOB_ID}-failure.txt"
FINAL_AUDIT="$DEPLOY_STORE/manifests/xwam/pointmap/close_fridge_training/close_fridge_pointmap_formal_8gpu_${SLURM_JOB_ID}.json"

export XWAM_FAILURE_REPORT="$FAILURE_REPORT"
export XWAM_MAIN_LOG="$MAIN_LOG"
export XWAM_TRAIN_LOG="$TRAIN_LOG"
export XWAM_PHASE=close_fridge_pointmap_formal_preflight
# shellcheck source=deployment/clariden/error_trap.sh
source "$REPO/deployment/clariden/error_trap.sh"
xwam_install_err_trap

cd "$REPO"
test "${SLURM_NNODES:-0}" == 2
test -s "$EDF"
test -d "$RUNTIME_OVERLAY"
test -d "$WANDB_OVERLAY"
test -s "$DATASET/meta/info.json"
test -s "$DATASET/meta/stats.json"
test -d "$POINTMAP_CACHE_ROOT/CloseFridge"
test -s "$POINTMAP_MANIFEST"
test -s "$POINTMAP_AUDIT"
test -s "$POINTMAP_GATE_AUDIT"
for artifact in "$POINTMAP_AUDIT" "$POINTMAP_GATE_AUDIT"; do
  grep -q '"ok": true' "$artifact"
  grep -q '"result": "pass"' "$artifact"
done
CURRENT_COMMIT="$(git rev-parse HEAD)"
grep -q '"expect_pointmap": true' "$POINTMAP_GATE_AUDIT"
grep -q "\"git_commit\": \"$CURRENT_COMMIT\"" "$POINTMAP_GATE_AUDIT"
grep -q '"episode_count": 106' "$POINTMAP_AUDIT"
grep -q '"array_count": 318' "$POINTMAP_AUDIT"
test -s "$WAN_DIR/config.json"
test -s "$WAN_DIR/models_t5_umt5-xxl-enc-bf16.pth"
test -s "$WAN_DIR/Wan2.2_VAE.pth"
test -s "$PRETRAINED/checkpoint/mp_rank_00_model_states.pt"
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  xwam_fail "WANDB_API_KEY must be exported before sbatch submission"
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "[WARN] Git worktree is dirty; commit and status are recorded in run metadata" >&2
  git status --short >&2
fi

mkdir -p \
  "$EXP_ROOT" \
  "$RUN_DIR" \
  "$HOT_CHECKPOINT_ROOT" \
  "$DURABLE_CHECKPOINT_ROOT" \
  "$WANDB_ROOT" \
  "$WANDB_CACHE_DIR" \
  "$(dirname "$FINAL_AUDIT")"
exec 9>"$EXP_DIR/.close-fridge-pointmap-formal.lock"
flock -n 9 || {
  xwam_fail "Another job is already writing $EXP_DIR"
}

if [[ ! -s "$WANDB_RUN_ID_FILE" ]]; then
  WANDB_RUN_ID_TMP="$WANDB_RUN_ID_FILE.${SLURM_JOB_ID}.tmp"
  umask 077
  printf 'xwam-cf-pointmap-ratio00-seed42-%s-%s\n' \
    "$(date -u +%Y%m%dT%H%M%SZ)" "$SLURM_JOB_ID" > "$WANDB_RUN_ID_TMP"
  mv "$WANDB_RUN_ID_TMP" "$WANDB_RUN_ID_FILE"
fi
read -r WANDB_RUN_ID < "$WANDB_RUN_ID_FILE"
[[ "$WANDB_RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || {
  xwam_fail "Invalid persistent W&B run ID: $WANDB_RUN_ID"
}

echo "experiment=$EXP_NAME"
echo "model_config=$MODEL_CONFIG"
echo "data_config=$DATA_CONFIG"
echo "hardware_config=$HARDWARE_CONFIG"
echo "experiment_config=$EXPERIMENT_CONFIG"
echo "scheduler_steps=$TOTAL_STEPS target_steps=$TOTAL_STEPS"
echo "global_batch_size=128 (8 GPUs x batch 4 x accumulation 4)"
echo "rolling_checkpoints=$HOT_CHECKPOINT_ROOT (every 250 steps, max 5)"
echo "durable_checkpoints=$DURABLE_CHECKPOINT_ROOT (every 500 steps, unlimited)"
echo "experiment_artifacts=$EXP_DIR (not deleted by this job)"
echo "pointmap_gate_audit=$POINTMAP_GATE_AUDIT"
echo "wandb_run_id=$WANDB_RUN_ID"
echo "failure_report=$FAILURE_REPORT"

XWAM_PHASE=close_fridge_pointmap_formal_planner
srun --nodes=1 --ntasks=1 --kill-on-bad-exit=1 --environment="$EDF" \
  env \
    REPO="$REPO" \
    RUNTIME_OVERLAY="$RUNTIME_OVERLAY" \
    HOT_CHECKPOINT_ROOT="$HOT_CHECKPOINT_ROOT" \
    DURABLE_CHECKPOINT_ROOT="$DURABLE_CHECKPOINT_ROOT" \
    HOT_QUARANTINE_ROOT="$HOT_QUARANTINE_ROOT" \
    DURABLE_QUARANTINE_ROOT="$DURABLE_QUARANTINE_ROOT" \
    PLAN_JSON="$PLAN_JSON" \
    PLAN_ENV="$PLAN_ENV" \
    TOTAL_STEPS="$TOTAL_STEPS" \
    XWAM_FAILURE_REPORT="$FAILURE_REPORT" \
    XWAM_MAIN_LOG="$MAIN_LOG" \
    XWAM_TRAIN_LOG="$TRAIN_LOG" \
  bash -lc '
set -Eeuo pipefail
export XWAM_PHASE=close_fridge_pointmap_formal_planner_container
source "$REPO/deployment/clariden/error_trap.sh"
xwam_install_err_trap
cd "$REPO"
export PYTHONPATH="$RUNTIME_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
python scripts/plan_robocasa365_m6_formal_chunk.py \
  --checkpoint-root "$HOT_CHECKPOINT_ROOT" \
  --checkpoint-root "$DURABLE_CHECKPOINT_ROOT" \
  --total-steps "$TOTAL_STEPS" \
  --chunk-steps "$TOTAL_STEPS" \
  --expected-world-size 8 \
  --output-json "$PLAN_JSON" \
  --output-env "$PLAN_ENV" \
  --quarantine-incomplete-root "$HOT_QUARANTINE_ROOT" \
  --quarantine-incomplete-root "$DURABLE_QUARANTINE_ROOT"
'

test -s "$PLAN_JSON"
test -s "$PLAN_ENV"
# shellcheck disable=SC1090
source "$PLAN_ENV"
echo "completed_step=$XWAM_PLAN_COMPLETED_STEP"
echo "target_step=$XWAM_PLAN_TARGET_STEP"
echo "resume_checkpoint=$XWAM_PLAN_RESUME_CHECKPOINT"
if [[ "$XWAM_PLAN_ALREADY_COMPLETE" == 1 ]]; then
  echo "[PASS] CloseFridge PointMap-Aux formal training already has a complete step $TOTAL_STEPS checkpoint"
  exit 0
fi

METADATA="$RUN_DIR/${RUN_ID}_metadata.json"
RESULT="$RUN_DIR/${RUN_ID}_result.json"
EVENTS="$RUN_DIR/${RUN_ID}_checkpoint_events.jsonl"
OPTIMIZER_DIR="$RUN_DIR/${RUN_ID}_optimizer_state"

XWAM_PHASE=close_fridge_pointmap_formal_training
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')"
MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
test -n "$MASTER_ADDR"
srun --nodes=2 \
  --ntasks=2 \
  --ntasks-per-node=1 \
  --kill-on-bad-exit=1 \
  --environment="$EDF" \
  env \
    REPO="$REPO" \
    RUNTIME_OVERLAY="$RUNTIME_OVERLAY" \
    WANDB_OVERLAY="$WANDB_OVERLAY" \
    DATASET="$DATASET" \
    POINTMAP_CACHE_ROOT="$POINTMAP_CACHE_ROOT" \
    POINTMAP_MANIFEST="$POINTMAP_MANIFEST" \
    POINTMAP_AUDIT="$POINTMAP_AUDIT" \
    WAN_DIR="$WAN_DIR" \
    PRETRAINED="$PRETRAINED" \
    MODEL_CONFIG="$MODEL_CONFIG" \
    DATA_CONFIG="$DATA_CONFIG" \
    HARDWARE_CONFIG="$HARDWARE_CONFIG" \
    EXPERIMENT_CONFIG="$EXPERIMENT_CONFIG" \
    EXP_ROOT="$EXP_ROOT" \
    EXP_NAME="$EXP_NAME" \
    HOT_CHECKPOINT_ROOT="$HOT_CHECKPOINT_ROOT" \
    DURABLE_CHECKPOINT_ROOT="$DURABLE_CHECKPOINT_ROOT" \
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
    MASTER_ADDR="$MASTER_ADDR" \
    MASTER_PORT="$MASTER_PORT" \
    XWAM_FAILURE_REPORT="$FAILURE_REPORT" \
    XWAM_MAIN_LOG="$MAIN_LOG" \
    XWAM_TRAIN_LOG="$TRAIN_LOG" \
  bash -lc '
set -Eeuo pipefail
NODE_RANK="${SLURM_NODEID:?SLURM_NODEID is required}"
NODE_TRAIN_LOG="$TRAIN_LOG"
NODE_FAILURE_REPORT="$XWAM_FAILURE_REPORT"
if [[ "$NODE_RANK" != 0 ]]; then
  NODE_TRAIN_LOG="${TRAIN_LOG%.log}-node${NODE_RANK}.log"
fi
NODE_FAILURE_REPORT="${XWAM_FAILURE_REPORT%.txt}-node${NODE_RANK}.txt"
export XWAM_FAILURE_REPORT="$NODE_FAILURE_REPORT"
export XWAM_TRAIN_LOG="$NODE_TRAIN_LOG"
export XWAM_PHASE=close_fridge_pointmap_formal_training_container
source "$REPO/deployment/clariden/error_trap.sh"
xwam_install_err_trap
cd "$REPO"
export MAX_JOBS=8
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
print("CloseFridge PointMap-Aux formal GPU preflight:", names)
PY

if [[ "$NODE_RANK" == 0 ]]; then
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
fi

export SLURM_JOB_NAME=interactive
unset SLURM_NTASKS SLURM_NTASKS_PER_NODE SLURM_PROCID SLURM_LOCALID SLURM_NODEID
export XWAM_RUN_ID="$RUN_ID"

TRAIN_ARGS=(
  scripts/train_sft.py
  "model_config=$MODEL_CONFIG"
  "data_config=$DATA_CONFIG"
  "hardware_config=$HARDWARE_CONFIG"
  "experiment_config=$EXPERIMENT_CONFIG"
  "wan_checkpoint_dir=$WAN_DIR"
  "pretrained_checkpoint=$PRETRAINED"
  "dataset.dataset_path=$DATASET"
  "dataset.pointmap_cache_root=$POINTMAP_CACHE_ROOT"
  "dataset.pointmap_cache_manifest=$POINTMAP_MANIFEST"
  "dataset.pointmap_cache_audit=$POINTMAP_AUDIT"
  "exp_root=$EXP_ROOT"
  "exp_name=$EXP_NAME"
  "checkpoint_dir=$HOT_CHECKPOINT_ROOT"
  "durable_checkpoint_dir=$DURABLE_CHECKPOINT_ROOT"
  "final_checkpoint_dir=$DURABLE_CHECKPOINT_ROOT"
  "trainer_max_steps=$TARGET_STEP"
  save_final_checkpoint=true
)
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  TRAIN_ARGS+=("resume_checkpoint=$RESUME_CHECKPOINT")
fi

git rev-parse HEAD
if [[ -n "$(git status --porcelain)" ]]; then git status --short >&2; fi
nvidia-smi
free -h
python -m torch.distributed.run \
  --nnodes 2 \
  --nproc-per-node 4 \
  --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" \
  --master-port "$MASTER_PORT" \
  "${TRAIN_ARGS[@]}" 2>&1 | tee "$NODE_TRAIN_LOG"
'

XWAM_PHASE=close_fridge_pointmap_formal_artifact_audit
test -s "$METADATA"
test -s "$RESULT"
test -s "$EVENTS"
test -d "$OPTIMIZER_DIR"
srun --nodes=1 --ntasks=1 --kill-on-bad-exit=1 --environment="$EDF" \
  env \
    METADATA="$METADATA" \
    RESULT="$RESULT" \
    EVENTS="$EVENTS" \
    OPTIMIZER_DIR="$OPTIMIZER_DIR" \
    TRAIN_LOG="$TRAIN_LOG" \
    EXPECTED_STEP="$XWAM_PLAN_TARGET_STEP" \
    EXPECTED_WANDB_RUN_ID="$WANDB_RUN_ID" \
    FINAL_AUDIT="$FINAL_AUDIT" \
  bash -lc '
set -Eeuo pipefail
python - <<"PY"
import json
import math
import os
import re
from pathlib import Path

metadata = json.loads(Path(os.environ["METADATA"]).read_text())
result = json.loads(Path(os.environ["RESULT"]).read_text())
expected_step = int(os.environ["EXPECTED_STEP"])
resolved_path = Path(metadata["resolved_config"])
resolved = resolved_path.read_text()
log_text = Path(os.environ["TRAIN_LOG"]).read_text(errors="replace")
pointmap_losses = [
    float(value)
    for value in re.findall(r"train/pointmap_loss:\s*([0-9.eE+-]+)", log_text)
]

checks = {
    "result_pass": result.get("result") == "pass",
    "target_step": int(result.get("global_step", -1)) == expected_step == 1500,
    "git_commit_recorded": bool(metadata["git"]["commit"]),
    "close_fridge": metadata["dataset"]["task_name"] == "CloseFridge",
    "world_size_eight": int(metadata["topology"]["world_size"]) == 8,
    "two_node_topology": int(metadata["topology"]["num_nodes"]) == 2 and int(metadata["topology"]["devices_per_node"]) == 4,
    "scheduler_1500": int(metadata["training"]["num_training_steps"]) == 1500,
    "wandb_identity": metadata["tracking"]["wandb"]["run_id"] == os.environ["EXPECTED_WANDB_RUN_ID"],
    "wandb_auth": metadata["tracking"]["wandb"]["auth"] == "api_key_env",
    "pointmap_enabled": "use_pointmap: true" in resolved and "pointmap_loss_weight: 1.0" in resolved and "use_depth: false" in resolved,
    "ratio_zero": "clean_action_ratio: 0.0" in resolved,
    "augmentation_enabled": "augment: true" in resolved,
    "global_batch_128": "global_batch_size: 128" in resolved,
    "microbatch_contract": "batch_size_per_gpu: 4" in resolved and "accumulate_grad_batches: 4" in resolved,
    "configured_topology": "formal_world_size: 8" in resolved and "formal_num_nodes: 2" in resolved and "formal_devices_per_node: 4" in resolved,
    "zero_one_fp32": "deepspeed_stage: 1" in resolved and "deepspeed_fp32_optimizer_states: true" in resolved,
    "gradient_checkpointing": "use_gradient_checkpointing: true" in resolved,
    "rolling_checkpoint_contract": "save_interval: 250" in resolved and "save_top_k: 5" in resolved,
    "durable_checkpoint_contract": "durable_save_interval: 500" in resolved and "durable_save_top_k: -1" in resolved,
    "pointmap_loss_positive": bool(pointmap_losses) and all(math.isfinite(value) and value > 0 for value in pointmap_losses),
}
checkpoint = Path(result["last_checkpoint"])
checks["complete_checkpoint"] = checkpoint.is_dir()
model_states = sorted(checkpoint.glob("**/mp_rank_*_model_states.pt"))
checks["model_state_present"] = bool(model_states) and all(
    path.stat().st_size > 0 for path in model_states
)
optimizer_reports = sorted(Path(os.environ["OPTIMIZER_DIR"]).glob("optimizer_state_rank_*.json"))
optimizer_ranks = sorted(
    int(json.loads(path.read_text())["global_rank"]) for path in optimizer_reports
)
checks["eight_optimizer_reports"] = optimizer_ranks == list(range(8))
optimizer_pattern = re.compile(
    r"^(?:[^/]+_)?zero_pp_rank_(\d+)_mp_rank_\d+_optim_states\.pt$"
)
optimizer_shards = [
    path
    for path in checkpoint.glob("**/*zero_pp_rank_*_optim_states.pt")
    if optimizer_pattern.fullmatch(path.name)
]
checkpoint_ranks = sorted(
    int(optimizer_pattern.fullmatch(path.name).group(1)) for path in optimizer_shards
)
checks["eight_checkpoint_shards"] = checkpoint_ranks == list(range(8)) and all(
    path.stat().st_size > 0 for path in optimizer_shards
)
errors = [name for name, passed in checks.items() if not passed]
payload = {
    "schema_version": 1,
    "result": "pass" if not errors else "fail",
    "ok": not errors,
    "checks": checks,
    "errors": errors,
    "metadata": str(Path(os.environ["METADATA"]).resolve()),
    "result_path": str(Path(os.environ["RESULT"]).resolve()),
    "resolved_config": str(resolved_path.resolve()),
    "checkpoint": str(checkpoint.resolve()),
    "pointmap_loss_records": len(pointmap_losses),
}
output = Path(os.environ["FINAL_AUDIT"])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
if errors:
    raise SystemExit(2)
PY
'

test -s "$FINAL_AUDIT"
grep -q '"ok": true' "$FINAL_AUDIT"
echo "metadata=$METADATA"
echo "result=$RESULT"
echo "train_log=$TRAIN_LOG"
echo "formal_audit=$FINAL_AUDIT"
echo "rolling_checkpoints=$HOT_CHECKPOINT_ROOT"
echo "durable_checkpoints=$DURABLE_CHECKPOINT_ROOT"
echo "[PASS] CloseFridge PointMap-Aux formal training reached step $XWAM_PLAN_TARGET_STEP"
