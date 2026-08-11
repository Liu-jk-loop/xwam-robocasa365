#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --job-name=xwam_m6_formal
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/m6-formal-%j.log
#SBATCH --export=ALL

set -Eeuo pipefail

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
DEPLOY_CAPSCR=/capstor/scratch/cscs/zjingchen/terry_nys
DEPLOY_IOPS=/iopsstor/scratch/cscs/zjingchen/terry_nys
REPO="$DEPLOY_STORE/src/xwam-robocasa365"
EDF="$DEPLOY_STORE/containers/edf/xwam.toml"
RUNTIME_OVERLAY="$DEPLOY_IOPS/python/xwam-nvtx-0.2.15"
WANDB_OVERLAY="$DEPLOY_IOPS/python/xwam-wandb-0.23.1"
WAN_DIR="$DEPLOY_STORE/checkpoints/xwam/wan22_5b"
PRETRAINED="$DEPLOY_STORE/checkpoints/xwam/pretrained/checkpoints/last.ckpt"
MANIFEST="$DEPLOY_STORE/manifests/xwam/m6/robocasa365_m6_atomic_seen18_manifest.json"
STATS="$DEPLOY_STORE/manifests/xwam/m6/robocasa365_m6_atomic_seen18_global_stats.json"
PREFLIGHT="$DEPLOY_STORE/manifests/xwam/m6/robocasa365_m6_gh200_preflight.json"
GATE_AUDIT="${XWAM_M6_GATE_AUDIT:-$DEPLOY_STORE/logs/xwam/debug/m6-gate-3053436-audit.json}"

TOTAL_STEPS=16390
# Do not stop at an artificial intermediate step boundary.  The 12-hour Slurm
# allocation may end first; the next submission resumes the newest complete
# rolling (500-step) or durable (3000-step) checkpoint.
CHUNK_STEPS="$TOTAL_STEPS"
EXP_ROOT="$DEPLOY_CAPSCR/experiments/xwam"
EXP_NAME="${XWAM_M6_FORMAL_NAME:-robocasa365_m6_atomic_seen18_rgb_seed42}"
EXP_DIR="$EXP_ROOT/$EXP_NAME"
HOT_CHECKPOINT_ROOT="$DEPLOY_IOPS/xwam_run/$EXP_NAME/checkpoints"
DURABLE_CHECKPOINT_ROOT="$DEPLOY_STORE/checkpoints/xwam/$EXP_NAME/checkpoints"
RUN_DIR="$EXP_DIR/runs"
WANDB_RUN_ID_FILE="$EXP_DIR/.wandb_run_id"
WANDB_PROJECT="${WANDB_PROJECT:-xwam-robocasa365}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_ROOT="$DEPLOY_IOPS/wandb/xwam/$EXP_NAME"
WANDB_CACHE_DIR="$DEPLOY_IOPS/cache/wandb"
PLAN_JSON="$RUN_DIR/m6-formal-${SLURM_JOB_ID}-plan.json"
PLAN_ENV="$RUN_DIR/m6-formal-${SLURM_JOB_ID}-plan.env"
HOT_QUARANTINE_ROOT="$DEPLOY_IOPS/xwam_run/$EXP_NAME/incomplete-checkpoints/m6-formal-${SLURM_JOB_ID}"
DURABLE_QUARANTINE_ROOT="$DEPLOY_STORE/checkpoints/xwam/$EXP_NAME/incomplete-checkpoints/m6-formal-${SLURM_JOB_ID}"
RUN_ID="m6-formal-${SLURM_JOB_ID}"
TRAIN_LOG="$DEPLOY_STORE/logs/xwam/m6-formal-${SLURM_JOB_ID}-train.log"
CHUNK_AUDIT="$DEPLOY_STORE/logs/xwam/m6-formal-${SLURM_JOB_ID}-chunk-audit.json"
FINAL_AUDIT="$DEPLOY_STORE/logs/xwam/m6-formal-${SLURM_JOB_ID}-final-audit.json"
MAIN_LOG="$DEPLOY_STORE/logs/xwam/m6-formal-${SLURM_JOB_ID}.log"
FAILURE_REPORT="$DEPLOY_STORE/logs/xwam/m6-formal-${SLURM_JOB_ID}-failure.txt"

export XWAM_FAILURE_REPORT="$FAILURE_REPORT"
export XWAM_MAIN_LOG="$MAIN_LOG"
export XWAM_TRAIN_LOG="$TRAIN_LOG"
export XWAM_PHASE=outer_preflight
# shellcheck source=deployment/clariden/error_trap.sh
source "$REPO/deployment/clariden/error_trap.sh"
xwam_install_err_trap

cd "$REPO"
test -s "$EDF"
test -d "$RUNTIME_OVERLAY"
test -d "$WANDB_OVERLAY"
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  xwam_fail "WANDB_API_KEY must be exported before sbatch submission"
fi
test -s "$WAN_DIR/config.json"
test -s "$WAN_DIR/models_t5_umt5-xxl-enc-bf16.pth"
test -s "$WAN_DIR/Wan2.2_VAE.pth"
test -s "$PRETRAINED/checkpoint/mp_rank_00_model_states.pt"
for artifact in "$MANIFEST" "$STATS" "$PREFLIGHT" "$GATE_AUDIT"; do
  test -s "$artifact"
  grep -q '"ok": true' "$artifact"
  grep -q '"result": "pass"' "$artifact"
done
grep -q '"same_zero_stage": true' "$GATE_AUDIT"
grep -q '"zero_stage": 1' "$GATE_AUDIT"
test -z "$(git status --porcelain)" || {
  git status --short >&2
  xwam_fail "Git worktree must be clean before M6 formal training"
}
mkdir -p \
  "$EXP_ROOT" \
  "$RUN_DIR" \
  "$WANDB_ROOT" \
  "$WANDB_CACHE_DIR" \
  "$HOT_CHECKPOINT_ROOT" \
  "$DURABLE_CHECKPOINT_ROOT"
exec 9>"$EXP_DIR/.formal-training.lock"
flock -n 9 || {
  xwam_fail "Another formal job is already writing $EXP_DIR"
}

if [[ ! -s "$WANDB_RUN_ID_FILE" ]]; then
  WANDB_RUN_ID_TMP="$WANDB_RUN_ID_FILE.${SLURM_JOB_ID}.tmp"
  umask 077
  printf 'xwam-m6-a18-seed42-%s-%s\n' \
    "$(date -u +%Y%m%dT%H%M%SZ)" "$SLURM_JOB_ID" > "$WANDB_RUN_ID_TMP"
  mv "$WANDB_RUN_ID_TMP" "$WANDB_RUN_ID_FILE"
fi
read -r WANDB_RUN_ID < "$WANDB_RUN_ID_FILE"
[[ "$WANDB_RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || {
  xwam_fail "Invalid persistent W&B run ID: $WANDB_RUN_ID"
}
echo "wandb_project=$WANDB_PROJECT"
echo "wandb_entity=$WANDB_ENTITY"
echo "wandb_run_id=$WANDB_RUN_ID"
echo "rolling_checkpoints=$HOT_CHECKPOINT_ROOT (every 500 steps, max 5)"
echo "durable_checkpoints=$DURABLE_CHECKPOINT_ROOT (every 3000 steps, unlimited)"
echo "failure_report=$FAILURE_REPORT"

XWAM_PHASE=planner_srun
srun --kill-on-bad-exit=1 --environment="$EDF" \
  env \
    REPO="$REPO" \
    RUNTIME_OVERLAY="$RUNTIME_OVERLAY" \
    WANDB_OVERLAY="$WANDB_OVERLAY" \
    HOT_CHECKPOINT_ROOT="$HOT_CHECKPOINT_ROOT" \
    DURABLE_CHECKPOINT_ROOT="$DURABLE_CHECKPOINT_ROOT" \
    PLAN_JSON="$PLAN_JSON" \
    PLAN_ENV="$PLAN_ENV" \
    HOT_QUARANTINE_ROOT="$HOT_QUARANTINE_ROOT" \
    DURABLE_QUARANTINE_ROOT="$DURABLE_QUARANTINE_ROOT" \
    TOTAL_STEPS="$TOTAL_STEPS" \
    CHUNK_STEPS="$CHUNK_STEPS" \
    XWAM_FAILURE_REPORT="$FAILURE_REPORT" \
    XWAM_MAIN_LOG="$MAIN_LOG" \
    XWAM_TRAIN_LOG="$TRAIN_LOG" \
  bash -lc '
set -Eeuo pipefail
export XWAM_PHASE=chunk_planner
source "$REPO/deployment/clariden/error_trap.sh"
xwam_install_err_trap
cd "$REPO"
export PYTHONPATH="$RUNTIME_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
python scripts/plan_robocasa365_m6_formal_chunk.py \
  --checkpoint-root "$HOT_CHECKPOINT_ROOT" \
  --checkpoint-root "$DURABLE_CHECKPOINT_ROOT" \
  --total-steps "$TOTAL_STEPS" \
  --chunk-steps "$CHUNK_STEPS" \
  --output-json "$PLAN_JSON" \
  --output-env "$PLAN_ENV" \
  --quarantine-incomplete-root "$HOT_QUARANTINE_ROOT" \
  --quarantine-incomplete-root "$DURABLE_QUARANTINE_ROOT"
'

XWAM_PHASE=plan_validation
test -s "$PLAN_JSON"
test -s "$PLAN_ENV"
# shellcheck disable=SC1090
source "$PLAN_ENV"

echo "formal_name=$EXP_NAME"
echo "completed_step=$XWAM_PLAN_COMPLETED_STEP"
echo "target_step=$XWAM_PLAN_TARGET_STEP"
echo "resume_checkpoint=$XWAM_PLAN_RESUME_CHECKPOINT"
echo "final_chunk=$XWAM_PLAN_FINAL_CHUNK"

if [[ "$XWAM_PLAN_ALREADY_COMPLETE" == 1 ]]; then
  echo "[PASS] X-WAM M6 formal training already has a complete step $TOTAL_STEPS checkpoint"
  exit 0
fi

SAVE_FINAL=false
if [[ "$XWAM_PLAN_FINAL_CHUNK" == 1 ]]; then
  SAVE_FINAL=true
fi

METADATA="$RUN_DIR/${RUN_ID}_metadata.json"
RESULT="$RUN_DIR/${RUN_ID}_result.json"
EVENTS="$RUN_DIR/${RUN_ID}_checkpoint_events.jsonl"
OPTIMIZER_DIR="$RUN_DIR/${RUN_ID}_optimizer_state"

XWAM_PHASE=training_srun
srun --kill-on-bad-exit=1 --environment="$EDF" \
  env \
    REPO="$REPO" \
    RUNTIME_OVERLAY="$RUNTIME_OVERLAY" \
    WANDB_OVERLAY="$WANDB_OVERLAY" \
    WAN_DIR="$WAN_DIR" \
    PRETRAINED="$PRETRAINED" \
    MANIFEST="$MANIFEST" \
    STATS="$STATS" \
    EXP_ROOT="$EXP_ROOT" \
    EXP_NAME="$EXP_NAME" \
    HOT_CHECKPOINT_ROOT="$HOT_CHECKPOINT_ROOT" \
    DURABLE_CHECKPOINT_ROOT="$DURABLE_CHECKPOINT_ROOT" \
    RUN_ID="$RUN_ID" \
    TARGET_STEP="$XWAM_PLAN_TARGET_STEP" \
    RESUME_CHECKPOINT="$XWAM_PLAN_RESUME_CHECKPOINT" \
    SAVE_FINAL="$SAVE_FINAL" \
    TRAIN_LOG="$TRAIN_LOG" \
    METADATA="$METADATA" \
    RESULT="$RESULT" \
    EVENTS="$EVENTS" \
    OPTIMIZER_DIR="$OPTIMIZER_DIR" \
    CHUNK_AUDIT="$CHUNK_AUDIT" \
    FINAL_AUDIT="$FINAL_AUDIT" \
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
export XWAM_PHASE=training_shell_setup
source "$REPO/deployment/clariden/error_trap.sh"
xwam_install_err_trap
cd "$REPO"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1
unset WANDB_IDENTITY_TOKEN_FILE
export SLURM_JOB_NAME=interactive
unset SLURM_NTASKS SLURM_NTASKS_PER_NODE SLURM_PROCID SLURM_LOCALID SLURM_NODEID
export XWAM_RUN_ID="$RUN_ID"
export PYTHONPATH="$WANDB_OVERLAY:$RUNTIME_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"

XWAM_PHASE=wandb_preflight
python - <<"PY"
import importlib.metadata
import os
from pathlib import Path

import wandb

overlay = Path(os.environ["WANDB_OVERLAY"]).resolve()
module = Path(wandb.__file__).resolve()
assert module.is_relative_to(overlay), (module, overlay)
assert importlib.metadata.version("wandb") == "0.23.1"
assert os.environ["WANDB_MODE"] == "online"
assert os.environ["WANDB_RESUME"] == "allow"
assert os.environ["WANDB_RUN_ID"]
assert os.environ.get("WANDB_API_KEY")
assert wandb.login(verify=True)
print("wandb version:", importlib.metadata.version("wandb"))
print("wandb module:", module)
print("wandb project:", os.environ["WANDB_PROJECT"])
print("wandb run id:", os.environ["WANDB_RUN_ID"])
print("wandb authentication: api_key_env pass")
PY

TRAIN_ARGS=(
  python scripts/train_sft.py
  model_config=configs/model/wan22_5b_robocasa365_atomic.yaml
  data_config=configs/data/robocasa365_m6_atomic_seen18.yaml
  hardware_config=configs/hardware/gh200x4_96gb_gbs128.yaml
  experiment_config=configs/experiment/robocasa365_m6_gh200_rgb_formal.yaml
  "wan_checkpoint_dir=$WAN_DIR"
  "pretrained_checkpoint=$PRETRAINED"
  "dataset.multitask_manifest=$MANIFEST"
  "dataset.statistics_path=$STATS"
  "exp_root=$EXP_ROOT"
  "exp_name=$EXP_NAME"
  "checkpoint_dir=$HOT_CHECKPOINT_ROOT"
  "durable_checkpoint_dir=$DURABLE_CHECKPOINT_ROOT"
  "final_checkpoint_dir=$DURABLE_CHECKPOINT_ROOT"
  "trainer_max_steps=$TARGET_STEP"
  "save_final_checkpoint=$SAVE_FINAL"
)
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  TRAIN_ARGS+=("resume_checkpoint=$RESUME_CHECKPOINT")
fi

XWAM_PHASE=training
git rev-parse HEAD
test -z "$(git status --porcelain)"
nvidia-smi
free -h
"${TRAIN_ARGS[@]}" 2>&1 | tee "$TRAIN_LOG"

XWAM_PHASE=training_artifact_validation
test -s "$METADATA"
test -s "$RESULT"
test -s "$EVENTS"
test -d "$OPTIMIZER_DIR"

AUDIT_ARGS=(
  python scripts/audit_robocasa365_m6_formal_chunk.py
  --metadata "$METADATA"
  --result "$RESULT"
  --events "$EVENTS"
  --optimizer-dir "$OPTIMIZER_DIR"
  --log "$TRAIN_LOG"
  --expected-step "$TARGET_STEP"
  --expected-wandb-run-id "$WANDB_RUN_ID"
  --expected-rolling-checkpoint-root "$HOT_CHECKPOINT_ROOT"
  --expected-durable-checkpoint-root "$DURABLE_CHECKPOINT_ROOT"
  --output "$CHUNK_AUDIT"
)
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  AUDIT_ARGS+=(--expected-resume-checkpoint "$RESUME_CHECKPOINT")
fi
XWAM_PHASE=chunk_audit
"${AUDIT_ARGS[@]}"

if [[ "$SAVE_FINAL" == true ]]; then
  XWAM_PHASE=final_audit
  python scripts/audit_robocasa365_m6_formal_training.py \
    --metadata "$METADATA" \
    --result "$RESULT" \
    --events "$EVENTS" \
    --optimizer-dir "$OPTIMIZER_DIR" \
    --log "$TRAIN_LOG" \
    --expected-wandb-run-id "$WANDB_RUN_ID" \
    --expected-rolling-checkpoint-root "$HOT_CHECKPOINT_ROOT" \
    --expected-durable-checkpoint-root "$DURABLE_CHECKPOINT_ROOT" \
    --output "$FINAL_AUDIT"
fi
'

XWAM_PHASE=completion_reporting
echo "plan=$PLAN_JSON"
echo "train_log=$TRAIN_LOG"
echo "chunk_audit=$CHUNK_AUDIT"
if [[ "$SAVE_FINAL" == true ]]; then
  echo "final_audit=$FINAL_AUDIT"
  echo "[PASS] X-WAM Clariden M6 formal 5-epoch training completed at step $TOTAL_STEPS"
else
  echo "[PASS] X-WAM Clariden M6 formal chunk completed at step $XWAM_PLAN_TARGET_STEP"
  echo "Submit this same sbatch again to continue from the newest complete checkpoint."
fi
