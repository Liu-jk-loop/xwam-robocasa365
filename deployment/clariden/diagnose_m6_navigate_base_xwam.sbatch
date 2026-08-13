#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=220G
#SBATCH --time=02:00:00
#SBATCH --job-name=xwam_nav_base_diag
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/nav-base-diag-%j.log
#SBATCH --error=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/nav-base-diag-%j.err
#SBATCH --export=ALL

set -Eeuo pipefail

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
DEPLOY_IOPS=/iopsstor/scratch/cscs/zjingchen/terry_nys
REPO="$DEPLOY_STORE/src/xwam-robocasa365"
ROBOCASA_ROOT="$DEPLOY_STORE/src/robocasa"
ROBOSUITE_ROOT="$DEPLOY_STORE/src/robosuite"
XWAM_EDF="$DEPLOY_STORE/containers/edf/xwam.toml"
ROBOCASA_EDF="$DEPLOY_STORE/containers/edf/robocasa365.toml"
RUNTIME_OVERLAY="$DEPLOY_IOPS/python/xwam-nvtx-0.2.15"
WAN_DIR="$DEPLOY_STORE/checkpoints/xwam/wan22_5b"
MANIFEST="$DEPLOY_STORE/manifests/xwam/m6/robocasa365_m6_atomic_seen18_manifest.json"
STATS="$DEPLOY_STORE/manifests/xwam/m6/robocasa365_m6_atomic_seen18_global_stats.json"
TOPOLOGY="$REPO/configs/evaluation/robocasa365_m6_atomic18_8server_16client.json"
DIAG_ID="${XWAM_BASE_DIAG_ID:-}"
EXPERIMENT_DIR="${XWAM_EVAL_EXPERIMENT_DIR:-}"
CHECKPOINT="${XWAM_EVAL_CHECKPOINT:-}"
EPISODES="${XWAM_BASE_DIAG_EPISODES:-1}"

if [[ -z "$DIAG_ID" || ! "$DIAG_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[FAIL] XWAM_BASE_DIAG_ID must be nonempty and contain only A-Za-z0-9._-" >&2
  exit 2
fi
DIAG_ROOT="$DEPLOY_IOPS/x-wam-eval/diagnostics/$DIAG_ID"
RESULT_ROOT="$DIAG_ROOT/results"
LOG_ROOT="$DIAG_ROOT/logs"
CONTROL_ROOT="$DIAG_ROOT/.runtime/$SLURM_JOB_ID"
READY_FILE="$CONTROL_ROOT/policy-ready.json"
STOP_FILE="$CONTROL_ROOT/stop-policy"
RESULT_JSON="$RESULT_ROOT/NavigateKitchen/result.json"
SUMMARY_JSON="$DIAG_ROOT/base_diagnostic_summary.json"
mkdir -p "$RESULT_ROOT" "$LOG_ROOT/clients" "$LOG_ROOT/servers" "$CONTROL_ROOT"
exec > >(tee -a "$LOG_ROOT/job.log") 2> >(tee -a "$LOG_ROOT/job.err" >&2)

export XWAM_FAILURE_REPORT="$LOG_ROOT/failure.txt"
export XWAM_MAIN_LOG="$LOG_ROOT/job.log"
export XWAM_TRAIN_LOG=not-applicable
export XWAM_PHASE=base_diagnostic_preflight
source "$REPO/deployment/clariden/error_trap.sh"
xwam_install_err_trap

cd "$REPO"
[[ -n "$EXPERIMENT_DIR" ]] || xwam_fail "XWAM_EVAL_EXPERIMENT_DIR is required"
[[ -n "$CHECKPOINT" ]] || xwam_fail "XWAM_EVAL_CHECKPOINT is required"
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || xwam_fail "XWAM_BASE_DIAG_EPISODES must be positive"
test -s "$XWAM_EDF"
test -s "$ROBOCASA_EDF"
test -d "$RUNTIME_OVERLAY"
test -d "$EXPERIMENT_DIR"
test -d "$CHECKPOINT"
test -s "$CHECKPOINT/checkpoint/mp_rank_00_model_states.pt"
test -s "$WAN_DIR/config.json"
test -s "$MANIFEST"
test -s "$STATS"
test -s "$ROBOCASA_ROOT/robocasa/models/assets/fixtures/sinks/Sink025/model.xml"
test -z "$(git status --porcelain)" || xwam_fail "Git worktree must be clean"

POLICY_STEP_PID=""
cleanup_policy() {
  touch "$STOP_FILE" 2>/dev/null || true
  if [[ -n "$POLICY_STEP_PID" ]] && kill -0 "$POLICY_STEP_PID" 2>/dev/null; then
    wait "$POLICY_STEP_PID" 2>/dev/null || true
  fi
  rm -f "$STOP_FILE" "$READY_FILE"
}
trap cleanup_policy EXIT INT TERM

export XWAM_PHASE=base_diagnostic_policy_start
srun --overlap --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=12 \
  --environment="$XWAM_EDF" \
  env REPO="$REPO" RUNTIME_OVERLAY="$RUNTIME_OVERLAY" TOPOLOGY="$TOPOLOGY" \
    EXPERIMENT_DIR="$EXPERIMENT_DIR" CHECKPOINT="$CHECKPOINT" WAN_DIR="$WAN_DIR" \
    MANIFEST="$MANIFEST" STATS="$STATS" LOG_ROOT="$LOG_ROOT" \
    READY_FILE="$READY_FILE" STOP_FILE="$STOP_FILE" \
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
  --multitask-manifest "$MANIFEST" \
  --statistics-path "$STATS" \
  --log-root "$LOG_ROOT" \
  --ready-file "$READY_FILE" \
  --stop-file "$STOP_FILE" \
  --server-id 0
' >"$LOG_ROOT/server_launcher.log" 2>&1 &
POLICY_STEP_PID=$!

export XWAM_PHASE=base_diagnostic_policy_ready
for _ in $(seq 1 900); do
  if [[ -s "$READY_FILE" ]] && grep -q '"result": "ready"' "$READY_FILE"; then
    break
  fi
  if ! kill -0 "$POLICY_STEP_PID" 2>/dev/null; then
    wait "$POLICY_STEP_PID"
    xwam_fail "NavigateKitchen policy server exited before READY"
  fi
  sleep 2
done
test -s "$READY_FILE"
grep -q '"ok": true' "$READY_FILE"

export XWAM_PHASE=base_diagnostic_client
set +e
srun --overlap --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=8 \
  --environment="$ROBOCASA_EDF" \
  env REPO="$REPO" ROBOCASA_ROOT="$ROBOCASA_ROOT" \
    ROBOSUITE_ROOT="$ROBOSUITE_ROOT" TOPOLOGY="$TOPOLOGY" \
    RESULT_ROOT="$RESULT_ROOT" LOG_ROOT="$LOG_ROOT" EPISODES="$EPISODES" \
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
python evaluation/launch_robocasa365_m6_client_pool.py \
  --topology "$TOPOLOGY" \
  --output-root "$RESULT_ROOT" \
  --log-root "$LOG_ROOT" \
  --episodes-per-task "$EPISODES" \
  --client-id 8
'
CLIENT_RC=$?
set -e

touch "$STOP_FILE"
export XWAM_PHASE=base_diagnostic_policy_shutdown
set +e
wait "$POLICY_STEP_PID"
POLICY_RC=$?
set -e
POLICY_STEP_PID=""

[[ "$CLIENT_RC" -eq 0 ]] || xwam_fail "NavigateKitchen diagnostic client failed"
[[ "$POLICY_RC" -eq 0 ]] || xwam_fail "NavigateKitchen diagnostic policy failed"
test -s "$RESULT_JSON"

export XWAM_PHASE=base_diagnostic_summary
srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=1 \
  --environment="$XWAM_EDF" \
  env REPO="$REPO" RESULT_JSON="$RESULT_JSON" SUMMARY_JSON="$SUMMARY_JSON" \
  bash -lc '
set -Eeuo pipefail
cd "$REPO"
python scripts/summarize_robocasa365_base_diagnostics.py \
  --result "$RESULT_JSON" \
  --output "$SUMMARY_JSON"
'
grep -q '"verdict":' "$SUMMARY_JSON"

rm -f "$STOP_FILE" "$READY_FILE"
trap - EXIT INT TERM
echo "[PASS] NavigateKitchen base diagnostic: $SUMMARY_JSON"
