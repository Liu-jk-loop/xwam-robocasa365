#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --job-name=xwam_m6_gate
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/m6-gate-%j.log

set -euo pipefail

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
DEPLOY_CAPSCR=/capstor/scratch/cscs/zjingchen/terry_nys
DEPLOY_IOPS=/iopsstor/scratch/cscs/zjingchen/terry_nys
EDF="$DEPLOY_STORE/containers/edf/xwam.toml"
RUNTIME_OVERLAY="$DEPLOY_IOPS/python/xwam-nvtx-0.2.15"
WAN_DIR="$DEPLOY_STORE/checkpoints/xwam/wan22_5b"
PRETRAINED="$DEPLOY_STORE/checkpoints/xwam/pretrained/checkpoints/last.ckpt"
MANIFEST="$DEPLOY_STORE/manifests/xwam/m6/robocasa365_m6_atomic_seen18_manifest.json"
STATS="$DEPLOY_STORE/manifests/xwam/m6/robocasa365_m6_atomic_seen18_global_stats.json"
PREFLIGHT="$DEPLOY_STORE/manifests/xwam/m6/robocasa365_m6_gh200_preflight.json"
HARDWARE_CONFIG="${XWAM_M6_HARDWARE_CONFIG:-configs/hardware/gh200x4_96gb_gbs128.yaml}"
case "$HARDWARE_CONFIG" in
  configs/hardware/gh200x4_96gb_gbs128.yaml | \
    configs/hardware/gh200x4_96gb_gbs128_safe.yaml) ;;
  *)
    echo "[FAIL] Unsupported XWAM_M6_HARDWARE_CONFIG=$HARDWARE_CONFIG" >&2
    exit 2
    ;;
esac
EXP_ROOT="$DEPLOY_CAPSCR/experiments/xwam"
EXP_NAME="robocasa365_m6_gh200_gate_${SLURM_JOB_ID}"
EXP_DIR="$EXP_ROOT/$EXP_NAME"
RUN_DIR="$EXP_DIR/runs"
INITIAL_RUN_ID="m6-gh200-${SLURM_JOB_ID}-initial"
RESUMED_RUN_ID="m6-gh200-${SLURM_JOB_ID}-resumed"
INITIAL_LOG="$DEPLOY_STORE/logs/xwam/m6-gate-${SLURM_JOB_ID}-initial.log"
RESUMED_LOG="$DEPLOY_STORE/logs/xwam/m6-gate-${SLURM_JOB_ID}-resumed.log"
AUDIT="$DEPLOY_STORE/logs/xwam/m6-gate-${SLURM_JOB_ID}-audit.json"
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
test -s "$WAN_DIR/config.json"
test -s "$WAN_DIR/models_t5_umt5-xxl-enc-bf16.pth"
test -s "$WAN_DIR/Wan2.2_VAE.pth"
test -s "$PRETRAINED/checkpoint/mp_rank_00_model_states.pt"
for artifact in "$MANIFEST" "$STATS" "$PREFLIGHT"; do
  test -s "$artifact"
  grep -q '"ok": true' "$artifact"
  grep -q '"result": "pass"' "$artifact"
done
test -z "$(git status --porcelain)" || {
  echo "[FAIL] Git worktree must be clean before the M6 gate" >&2
  git status --short >&2
  exit 2
}
mkdir -p "$EXP_ROOT" "$RUN_DIR"

srun --kill-on-bad-exit=1 --environment="$EDF" bash -lc "
set -euo pipefail
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SLURM_JOB_NAME=interactive
unset SLURM_NTASKS SLURM_NTASKS_PER_NODE SLURM_PROCID SLURM_LOCALID SLURM_NODEID
export XWAM_RUN_ID='$INITIAL_RUN_ID'
export PYTHONPATH='$RUNTIME_OVERLAY':\${PYTHONPATH:-}

git rev-parse HEAD
test -z \"\$(git status --porcelain)\"
nvidia-smi
free -h

python scripts/train_sft.py \\
  model_config=configs/model/wan22_5b_robocasa365_atomic.yaml \\
  data_config=configs/data/robocasa365_m6_atomic_seen18.yaml \\
  hardware_config='$HARDWARE_CONFIG' \\
  experiment_config=configs/experiment/robocasa365_m6_gh200_gate.yaml \\
  wan_checkpoint_dir='$WAN_DIR' \\
  pretrained_checkpoint='$PRETRAINED' \\
  dataset.multitask_manifest='$MANIFEST' \\
  dataset.statistics_path='$STATS' \\
  exp_root='$EXP_ROOT' \\
  exp_name='$EXP_NAME'
" 2>&1 | tee "$INITIAL_LOG"

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
  data_config=configs/data/robocasa365_m6_atomic_seen18.yaml \\
  hardware_config='$HARDWARE_CONFIG' \\
  experiment_config=configs/experiment/robocasa365_m6_gh200_gate.yaml \\
  wan_checkpoint_dir='$WAN_DIR' \\
  pretrained_checkpoint='$PRETRAINED' \\
  dataset.multitask_manifest='$MANIFEST' \\
  dataset.statistics_path='$STATS' \\
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
  bash -lc '
set -euo pipefail
python scripts/audit_robocasa365_m6_gate.py \
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
  --output "$AUDIT"
'

echo "initial_result=$INITIAL_RESULT"
echo "resume_checkpoint=$CHECKPOINT"
echo "resumed_result=$RESUMED_RESULT"
echo "audit=$AUDIT"
echo "hardware_config=$HARDWARE_CONFIG"
echo "[PASS] X-WAM Clariden M6 4xGH200 formal-profile step 2 to 4 gate"
