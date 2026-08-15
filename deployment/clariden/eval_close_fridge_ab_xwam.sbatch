#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=220G
#SBATCH --time=12:00:00
#SBATCH --job-name=xwam_cf_eval
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/close-fridge-eval-%j.log
#SBATCH --error=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/close-fridge-eval-%j.err
#SBATCH --export=ALL

set -Eeuo pipefail

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
DEPLOY_CAPSCR=/capstor/scratch/cscs/zjingchen/terry_nys
DEPLOY_IOPS=/iopsstor/scratch/cscs/zjingchen/terry_nys
# 评测默认使用独立clone；不要在训练作业使用的主仓库中pull或checkout。
REPO="${XWAM_EVAL_REPO:-$DEPLOY_STORE/src/xwam-robocasa365-eval}"
ROBOCASA_ROOT="$DEPLOY_STORE/src/robocasa"
ROBOSUITE_ROOT="$DEPLOY_STORE/src/robosuite"
XWAM_EDF="$DEPLOY_STORE/containers/edf/xwam.toml"
ROBOCASA_EDF="$DEPLOY_STORE/containers/edf/robocasa365.toml"
RUNTIME_OVERLAY="$DEPLOY_IOPS/python/xwam-nvtx-0.2.15"
WAN_DIR="$DEPLOY_STORE/checkpoints/xwam/wan22_5b"
TOPOLOGY="$REPO/configs/evaluation/robocasa365_m6_atomic18_8server_16client.json"
EXPERIMENT_DIR="${XWAM_EVAL_EXPERIMENT_DIR:-$DEPLOY_CAPSCR/experiments/xwam/close_fridge_rgb_ratio05_seed42_4gpu}"
CHECKPOINT="${XWAM_EVAL_CHECKPOINT:-$DEPLOY_STORE/checkpoints/xwam/close_fridge_rgb_ratio05_seed42_4gpu/checkpoints/final-step=1000.ckpt}"
EVAL_ID="${XWAM_EVAL_ID:-}"
EPISODES="${XWAM_EVAL_EPISODES:-50}"
CUDA_DEVICE="${XWAM_EVAL_CUDA_DEVICE:-0}"
STEP_GPUS="${XWAM_EVAL_STEP_GPUS:-1}"

if [[ -z "$EVAL_ID" || ! "$EVAL_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[FAIL] XWAM_EVAL_ID must be nonempty and contain only A-Za-z0-9._-" >&2
  exit 2
fi
if [[ ! "$CUDA_DEVICE" =~ ^[0-9]+$ ]]; then
  echo "[FAIL] XWAM_EVAL_CUDA_DEVICE must be a nonnegative integer" >&2
  exit 2
fi
if [[ ! "$STEP_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[FAIL] XWAM_EVAL_STEP_GPUS must be a positive integer" >&2
  exit 2
fi
if (( CUDA_DEVICE >= STEP_GPUS )); then
  echo "[FAIL] XWAM_EVAL_CUDA_DEVICE must be smaller than XWAM_EVAL_STEP_GPUS" >&2
  exit 2
fi
EVAL_ROOT="$DEPLOY_IOPS/x-wam-eval/close-fridge-ab/$EVAL_ID"
RESULT_ROOT="$EVAL_ROOT/results"
LOG_ROOT="$EVAL_ROOT/logs"
CONTROL_ROOT="$EVAL_ROOT/.runtime/$SLURM_JOB_ID"
READY_FILE="$CONTROL_ROOT/policy-ready.json"
STOP_FILE="$CONTROL_ROOT/stop-policy"
RESULT_JSON="$RESULT_ROOT/CloseFridge/result.json"
SUMMARY_JSON="$EVAL_ROOT/summary.json"
mkdir -p "$RESULT_ROOT" "$LOG_ROOT/clients" "$LOG_ROOT/servers" "$CONTROL_ROOT"
exec > >(tee -a "$LOG_ROOT/job.log") 2> >(tee -a "$LOG_ROOT/job.err" >&2)

if [[ ! -s "$REPO/deployment/clariden/error_trap.sh" ]]; then
  echo "[FAIL] Independent evaluation repo is missing or incomplete: $REPO" >&2
  exit 2
fi
export XWAM_FAILURE_REPORT="$LOG_ROOT/failure.txt"
export XWAM_MAIN_LOG="$LOG_ROOT/job.log"
export XWAM_TRAIN_LOG=not-applicable
export XWAM_PHASE=close_fridge_eval_preflight
source "$REPO/deployment/clariden/error_trap.sh"
xwam_install_err_trap

cd "$REPO"
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || xwam_fail "XWAM_EVAL_EPISODES must be positive"
test -s "$XWAM_EDF"
test -s "$ROBOCASA_EDF"
test -d "$RUNTIME_OVERLAY"
test -d "$EXPERIMENT_DIR"
test -s "$EXPERIMENT_DIR/config.yaml"
test -d "$CHECKPOINT"
test -s "$CHECKPOINT/checkpoint/mp_rank_00_model_states.pt"
test -s "$WAN_DIR/config.json"
test -s "$TOPOLOGY"
test -s "$ROBOCASA_ROOT/robocasa/models/assets/fixtures/sinks/Sink025/model.xml"
test -z "$(git status --porcelain)" || xwam_fail "Independent evaluation repo must be clean"

mkdir -p "$DEPLOY_IOPS/x-wam-eval/.locks"
exec 9>"$DEPLOY_IOPS/x-wam-eval/.locks/close-fridge-$EVAL_ID.lock"
flock -n 9 || xwam_fail "Another evaluation job is already writing $EVAL_ROOT"

CONTRACT="$LOG_ROOT/evaluation_contract.txt"
EXPECTED_CONTRACT="$CONTROL_ROOT/evaluation_contract.expected.txt"
{
  printf 'eval_id=%s\n' "$EVAL_ID"
  printf 'task=CloseFridge\n'
  printf 'experiment_dir=%s\n' "$(realpath "$EXPERIMENT_DIR")"
  printf 'checkpoint=%s\n' "$(realpath "$CHECKPOINT")"
  printf 'episodes=%s\n' "$EPISODES"
  printf 'model_seed=42\nseed_start=42\nreplan_steps=20\naction_denoise_steps=10\nmax_steps=1000\n'
  printf 'repo_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'experiment_config_sha256=%s\n' "$(sha256sum "$EXPERIMENT_DIR/config.yaml" | awk '{print $1}')"
  printf 'topology_sha256=%s\n' "$(sha256sum "$TOPOLOGY" | awk '{print $1}')"
  printf 'robocasa_commit=%s\n' "$(git -C "$ROBOCASA_ROOT" rev-parse HEAD)"
  printf 'robosuite_commit=%s\n' "$(git -C "$ROBOSUITE_ROOT" rev-parse HEAD)"
} >"$EXPECTED_CONTRACT"
if [[ -e "$CONTRACT" ]]; then
  cmp -s "$EXPECTED_CONTRACT" "$CONTRACT" || {
    diff -u "$CONTRACT" "$EXPECTED_CONTRACT" >&2 || true
    xwam_fail "EVAL_ID already belongs to another immutable evaluation contract"
  }
else
  cp "$EXPECTED_CONTRACT" "$CONTRACT.tmp.$SLURM_JOB_ID"
  mv "$CONTRACT.tmp.$SLURM_JOB_ID" "$CONTRACT"
fi

echo "eval_id=$EVAL_ID"
echo "repo=$REPO commit=$(git rev-parse HEAD)"
echo "experiment_dir=$EXPERIMENT_DIR"
echo "checkpoint=$CHECKPOINT"
echo "task=CloseFridge episodes=$EPISODES seed_start=42 model_seed=42"
echo "replan_steps=20 action_denoise_steps=10 max_steps=1000"
echo "gpu_binding=CUDA_VISIBLE_DEVICES=$CUDA_DEVICE step_gpus=$STEP_GPUS"
echo "result_root=$RESULT_ROOT"

SIMULATOR_PROBE="$LOG_ROOT/robocasa_eval_runtime_probe.json"
export XWAM_PHASE=close_fridge_eval_simulator_probe
srun --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=1 \
  --environment="$ROBOCASA_EDF" \
  env REPO="$REPO" ROBOCASA_ROOT="$ROBOCASA_ROOT" \
    ROBOSUITE_ROOT="$ROBOSUITE_ROOT" SIMULATOR_PROBE="$SIMULATOR_PROBE" \
  bash -lc '
set -Eeuo pipefail
cd "$REPO"
export PYTHONPATH="$ROBOCASA_ROOT:$ROBOSUITE_ROOT:$REPO${PYTHONPATH:+:$PYTHONPATH}"
python scripts/probe_robocasa365_eval_runtime.py \
  --robocasa-root "$ROBOCASA_ROOT" \
  --robosuite-root "$ROBOSUITE_ROOT" \
  --output "$SIMULATOR_PROBE"
'
grep -q '"ok": true' "$SIMULATOR_PROBE"

POLICY_STEP_PID=""
cleanup_policy() {
  touch "$STOP_FILE" 2>/dev/null || true
  if [[ -n "$POLICY_STEP_PID" ]] && kill -0 "$POLICY_STEP_PID" 2>/dev/null; then
    wait "$POLICY_STEP_PID" 2>/dev/null || true
  fi
  rm -f "$STOP_FILE" "$READY_FILE" "$EXPECTED_CONTRACT"
}
trap cleanup_policy EXIT INT TERM

export XWAM_PHASE=close_fridge_eval_policy_start
srun --overlap --exact --nodes=1 --ntasks=1 --gpus-per-node="$STEP_GPUS" --cpus-per-task=12 \
  --environment="$XWAM_EDF" \
  env REPO="$REPO" RUNTIME_OVERLAY="$RUNTIME_OVERLAY" TOPOLOGY="$TOPOLOGY" \
    EXPERIMENT_DIR="$EXPERIMENT_DIR" CHECKPOINT="$CHECKPOINT" WAN_DIR="$WAN_DIR" \
    LOG_ROOT="$LOG_ROOT" READY_FILE="$READY_FILE" STOP_FILE="$STOP_FILE" \
    CUDA_DEVICE="$CUDA_DEVICE" \
  bash -lc '
set -Eeuo pipefail
cd "$REPO"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$RUNTIME_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
python evaluation/launch_robocasa365_m6_policy_pool.py \
  --topology "$TOPOLOGY" \
  --experiment-dir "$EXPERIMENT_DIR" \
  --checkpoint "$CHECKPOINT" \
  --wan-checkpoint-dir "$WAN_DIR" \
  --log-root "$LOG_ROOT" \
  --ready-file "$READY_FILE" \
  --stop-file "$STOP_FILE" \
  --server-id 5 \
  --single-task-checkpoint \
  --single-task-name CloseFridge \
  --cuda-device "$CUDA_DEVICE"
' >"$LOG_ROOT/server_launcher.log" 2>&1 &
POLICY_STEP_PID=$!

export XWAM_PHASE=close_fridge_eval_policy_ready
for _ in $(seq 1 900); do
  if [[ -s "$READY_FILE" ]] && grep -q '"result": "ready"' "$READY_FILE"; then
    break
  fi
  if ! kill -0 "$POLICY_STEP_PID" 2>/dev/null; then
    wait "$POLICY_STEP_PID"
    xwam_fail "CloseFridge policy server exited before READY"
  fi
  sleep 2
done
test -s "$READY_FILE"
grep -q '"ok": true' "$READY_FILE"

export XWAM_PHASE=close_fridge_eval_client
set +e
srun --overlap --exact --nodes=1 --ntasks=1 --gpus-per-node="$STEP_GPUS" --cpus-per-task=8 \
  --environment="$ROBOCASA_EDF" \
  env REPO="$REPO" ROBOCASA_ROOT="$ROBOCASA_ROOT" \
    ROBOSUITE_ROOT="$ROBOSUITE_ROOT" TOPOLOGY="$TOPOLOGY" \
    RESULT_ROOT="$RESULT_ROOT" LOG_ROOT="$LOG_ROOT" EPISODES="$EPISODES" \
    CUDA_DEVICE="$CUDA_DEVICE" \
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
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTHONPATH="$ROBOCASA_ROOT:$ROBOSUITE_ROOT:$REPO${PYTHONPATH:+:$PYTHONPATH}"
python evaluation/launch_robocasa365_m6_client_pool.py \
  --topology "$TOPOLOGY" \
  --output-root "$RESULT_ROOT" \
  --log-root "$LOG_ROOT" \
  --episodes-per-task "$EPISODES" \
  --client-id 5 \
  --cuda-visible-devices "$CUDA_DEVICE"
'
CLIENT_RC=$?
set -e

touch "$STOP_FILE"
export XWAM_PHASE=close_fridge_eval_policy_shutdown
set +e
wait "$POLICY_STEP_PID"
POLICY_RC=$?
set -e
POLICY_STEP_PID=""

[[ "$CLIENT_RC" -eq 0 ]] || xwam_fail "CloseFridge evaluation client failed"
[[ "$POLICY_RC" -eq 0 ]] || xwam_fail "CloseFridge evaluation policy failed"
test -s "$RESULT_JSON"

export XWAM_PHASE=close_fridge_eval_summary
srun --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=1 \
  --environment="$XWAM_EDF" \
  env REPO="$REPO" RESULT_JSON="$RESULT_JSON" SUMMARY_JSON="$SUMMARY_JSON" \
    EPISODES="$EPISODES" EVAL_ID="$EVAL_ID" CHECKPOINT="$CHECKPOINT" \
  bash -lc '
set -Eeuo pipefail
cd "$REPO"
python scripts/summarize_robocasa365_single_task_evaluation.py \
  --result "$RESULT_JSON" \
  --task CloseFridge \
  --episodes "$EPISODES" \
  --eval-id "$EVAL_ID" \
  --checkpoint "$CHECKPOINT" \
  --output "$SUMMARY_JSON"
'
grep -q '"ok": true' "$SUMMARY_JSON"

rm -f "$STOP_FILE" "$READY_FILE" "$EXPECTED_CONTRACT"
trap - EXIT INT TERM
echo "[PASS] CloseFridge A/B evaluation completed: $SUMMARY_JSON"
