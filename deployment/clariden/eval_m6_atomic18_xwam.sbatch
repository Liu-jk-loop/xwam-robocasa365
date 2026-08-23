#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=450G
#SBATCH --time=12:00:00
#SBATCH --job-name=xwam_m6_eval
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/m6-eval-%j.log
#SBATCH --error=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/m6-eval-%j.err
#SBATCH --export=ALL

set -Eeuo pipefail

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
DEPLOY_IOPS=/iopsstor/scratch/cscs/zjingchen/terry_nys
REPO="${XWAM_EVAL_REPO:-$DEPLOY_STORE/src/xwam-robocasa365}"
ROBOCASA_ROOT="$DEPLOY_STORE/src/robocasa"
ROBOSUITE_ROOT="$DEPLOY_STORE/src/robosuite"
XWAM_EDF="$DEPLOY_STORE/containers/edf/xwam.toml"
ROBOCASA_EDF="$DEPLOY_STORE/containers/edf/robocasa365.toml"
RUNTIME_OVERLAY="$DEPLOY_IOPS/python/xwam-nvtx-0.2.15"
WAN_DIR="$DEPLOY_STORE/checkpoints/xwam/wan22_5b"
MANIFEST="${XWAM_EVAL_MANIFEST:-$DEPLOY_STORE/manifests/xwam/m6/robocasa365_m6_atomic_seen18_manifest.json}"
STATS="${XWAM_EVAL_STATS:-$DEPLOY_STORE/manifests/xwam/m6/robocasa365_m6_atomic_seen18_global_stats.json}"
TOPOLOGY="${XWAM_EVAL_TOPOLOGY:-$REPO/configs/evaluation/robocasa365_m6_atomic18_8server_16client.json}"
EVAL_ID="${XWAM_EVAL_ID:-}"
EXPERIMENT_DIR="${XWAM_EVAL_EXPERIMENT_DIR:-}"
CHECKPOINT="${XWAM_EVAL_CHECKPOINT:-}"
EPISODES_PER_TASK="${XWAM_EVAL_EPISODES:-50}"
EVAL_NAMESPACE="${XWAM_EVAL_NAMESPACE:-atomic18}"
SUMMARY_STEM="${XWAM_EVAL_SUMMARY_STEM:-atomic18}"
REQUIRE_DEPTH="${XWAM_EVAL_REQUIRE_DEPTH:-false}"

if [[ -z "$EVAL_ID" || ! "$EVAL_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[FAIL] XWAM_EVAL_ID must be nonempty and contain only A-Za-z0-9._-" >&2
  exit 2
fi
if [[ -z "$EVAL_NAMESPACE" || ! "$EVAL_NAMESPACE" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[FAIL] XWAM_EVAL_NAMESPACE must contain only A-Za-z0-9._-" >&2
  exit 2
fi
if [[ -z "$SUMMARY_STEM" || ! "$SUMMARY_STEM" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[FAIL] XWAM_EVAL_SUMMARY_STEM must contain only A-Za-z0-9._-" >&2
  exit 2
fi
if [[ "$REQUIRE_DEPTH" != true && "$REQUIRE_DEPTH" != false ]]; then
  echo "[FAIL] XWAM_EVAL_REQUIRE_DEPTH must be true or false" >&2
  exit 2
fi
EVAL_ROOT="$DEPLOY_IOPS/x-wam-eval/$EVAL_NAMESPACE/$EVAL_ID"
RESULT_ROOT="$EVAL_ROOT/results"
LOG_ROOT="$EVAL_ROOT/logs"
CONTROL_ROOT="$EVAL_ROOT/.runtime/$SLURM_JOB_ID"
READY_FILE="$CONTROL_ROOT/policy-ready.json"
STOP_FILE="$CONTROL_ROOT/stop-policy"
SUMMARY_JSON="$EVAL_ROOT/aggregate.json"
SUMMARY_CSV="$EVAL_ROOT/summary_${SUMMARY_STEM}.csv"
mkdir -p "$RESULT_ROOT" "$LOG_ROOT/clients" "$LOG_ROOT/servers"
exec > >(tee -a "$LOG_ROOT/job.log") 2> >(tee -a "$LOG_ROOT/job.err" >&2)

MAIN_LOG="$LOG_ROOT/job.log"
FAILURE_REPORT="$LOG_ROOT/failure.txt"
export XWAM_FAILURE_REPORT="$FAILURE_REPORT"
export XWAM_MAIN_LOG="$MAIN_LOG"
export XWAM_TRAIN_LOG=not-applicable
export XWAM_PHASE=eval_outer_preflight
# shellcheck source=deployment/clariden/error_trap.sh
source "$REPO/deployment/clariden/error_trap.sh"
xwam_install_err_trap

cd "$REPO"
[[ -n "$EXPERIMENT_DIR" ]] || xwam_fail "XWAM_EVAL_EXPERIMENT_DIR must name the formal experiment directory"
[[ -n "$CHECKPOINT" ]] || xwam_fail "XWAM_EVAL_CHECKPOINT must name a complete DeepSpeed checkpoint"
[[ "$EPISODES_PER_TASK" =~ ^[1-9][0-9]*$ ]] || xwam_fail "XWAM_EVAL_EPISODES must be a positive integer"
test -s "$XWAM_EDF"
test -s "$ROBOCASA_EDF"
test -d "$ROBOCASA_ROOT/robocasa"
test -d "$ROBOSUITE_ROOT/robosuite"
test -s "$ROBOCASA_ROOT/robocasa/models/assets/fixtures/sinks/Sink025/model.xml"
test -d "$RUNTIME_OVERLAY"
test -d "$EXPERIMENT_DIR"
test -s "$EXPERIMENT_DIR/config.yaml"
if [[ "$REQUIRE_DEPTH" == true ]]; then
  grep -Eq '^[[:space:]]*use_depth:[[:space:]]*true[[:space:]]*$' "$EXPERIMENT_DIR/config.yaml" || {
    xwam_fail "RGB-D evaluation requires use_depth=true in experiment config"
  }
fi
test -d "$CHECKPOINT"
test -s "$CHECKPOINT/checkpoint/mp_rank_00_model_states.pt"
test -s "$WAN_DIR/config.json"
test -s "$MANIFEST"
test -s "$STATS"
test -s "$TOPOLOGY"
{
  printf 'branch=%s\n' "$(git branch --show-current)"
  printf 'commit=%s\n' "$(git rev-parse HEAD)"
  git status --short
} >"$LOG_ROOT/repo-state.txt"

mkdir -p "$CONTROL_ROOT" "$DEPLOY_IOPS/x-wam-eval/.locks"
exec 9>"$DEPLOY_IOPS/x-wam-eval/.locks/$EVAL_NAMESPACE-$EVAL_ID.lock"
flock -n 9 || xwam_fail "Another evaluation job is already writing $EVAL_ROOT"

# 同一 eval ID 可以续跑，但不能混入另一 checkpoint/topology/commit 的 episode。
CONTRACT="$LOG_ROOT/evaluation_contract.txt"
EXPECTED_CONTRACT="$CONTROL_ROOT/evaluation_contract.expected.txt"
{
  printf 'eval_id=%s\n' "$EVAL_ID"
  printf 'experiment_dir=%s\n' "$(realpath "$EXPERIMENT_DIR")"
  printf 'checkpoint=%s\n' "$(realpath "$CHECKPOINT")"
  printf 'episodes_per_task=%s\n' "$EPISODES_PER_TASK"
  printf 'repo_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'robocasa_commit=%s\n' "$(git -C "$ROBOCASA_ROOT" rev-parse HEAD)"
  printf 'robocasa_tracked_diff_sha256=%s\n' "$(git -C "$ROBOCASA_ROOT" diff --binary HEAD | sha256sum | awk '{print $1}')"
  printf 'robosuite_commit=%s\n' "$(git -C "$ROBOSUITE_ROOT" rev-parse HEAD)"
  printf 'robosuite_tracked_diff_sha256=%s\n' "$(git -C "$ROBOSUITE_ROOT" diff --binary HEAD | sha256sum | awk '{print $1}')"
  printf 'sink025_sha256=%s\n' "$(sha256sum "$ROBOCASA_ROOT/robocasa/models/assets/fixtures/sinks/Sink025/model.xml" | awk '{print $1}')"
  printf 'topology_sha256=%s\n' "$(sha256sum "$TOPOLOGY" | awk '{print $1}')"
  printf 'manifest_sha256=%s\n' "$(sha256sum "$MANIFEST" | awk '{print $1}')"
  printf 'stats_sha256=%s\n' "$(sha256sum "$STATS" | awk '{print $1}')"
} >"$EXPECTED_CONTRACT"
if [[ -e "$CONTRACT" ]]; then
  cmp -s "$EXPECTED_CONTRACT" "$CONTRACT" || {
    diff -u "$CONTRACT" "$EXPECTED_CONTRACT" >&2 || true
    xwam_fail "EVAL_ID already belongs to a different immutable evaluation contract"
  }
else
  cp "$EXPECTED_CONTRACT" "$CONTRACT.tmp.$SLURM_JOB_ID"
  mv "$CONTRACT.tmp.$SLURM_JOB_ID" "$CONTRACT"
fi

echo "eval_id=$EVAL_ID"
echo "experiment_dir=$EXPERIMENT_DIR"
echo "checkpoint=$CHECKPOINT"
echo "episodes_per_task=$EPISODES_PER_TASK"
echo "topology=$(basename "$TOPOLOGY")"
echo "model_seed=42 task_seed_start=42 replan=20 action_denoise_steps=10"
echo "result_root=$RESULT_ROOT"
echo "failure_report=$FAILURE_REPORT"

# FastWAM 的已验证 simulator 路径通过 PYTHONPATH 使用 Store 中带完整
# assets 的 RoboCasa/robosuite 源码，而不是 SQSH 内不完整的 /opt 副本。
SIMULATOR_PROBE="$LOG_ROOT/robocasa_eval_runtime_probe.json"
XWAM_PHASE=eval_simulator_runtime_probe
srun --nodes=1 --ntasks=1 --cpus-per-task=1 \
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
grep -q '"result": "pass"' "$SIMULATOR_PROBE"

POLICY_STEP_PID=""
cleanup_policy() {
  touch "$STOP_FILE" 2>/dev/null || true
  if [[ -n "$POLICY_STEP_PID" ]] && kill -0 "$POLICY_STEP_PID" 2>/dev/null; then
    wait "$POLICY_STEP_PID" 2>/dev/null || true
  fi
  rm -f "$STOP_FILE" "$READY_FILE" "$EXPECTED_CONTRACT"
  rmdir "$CONTROL_ROOT" "$EVAL_ROOT/.runtime" 2>/dev/null || true
}
trap cleanup_policy EXIT INT TERM

XWAM_PHASE=eval_policy_pool_srun
srun --overlap --nodes=1 --ntasks=1 --gpus-per-node=4 --cpus-per-task=16 \
  --environment="$XWAM_EDF" \
  env \
    REPO="$REPO" \
    RUNTIME_OVERLAY="$RUNTIME_OVERLAY" \
    TOPOLOGY="$TOPOLOGY" \
    EXPERIMENT_DIR="$EXPERIMENT_DIR" \
    CHECKPOINT="$CHECKPOINT" \
    WAN_DIR="$WAN_DIR" \
    MANIFEST="$MANIFEST" \
    STATS="$STATS" \
    LOG_ROOT="$LOG_ROOT" \
    READY_FILE="$READY_FILE" \
    STOP_FILE="$STOP_FILE" \
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
  --stop-file "$STOP_FILE"
' >"$LOG_ROOT/server_launcher.log" 2>&1 &
POLICY_STEP_PID=$!

XWAM_PHASE=eval_policy_pool_readiness
for _ in $(seq 1 5400); do
  if [[ -s "$READY_FILE" ]] && grep -q '"result": "ready"' "$READY_FILE"; then
    break
  fi
  if ! kill -0 "$POLICY_STEP_PID" 2>/dev/null; then
    wait "$POLICY_STEP_PID"
    xwam_fail "Policy pool exited before all configured servers became ready"
  fi
  sleep 2
done
test -s "$READY_FILE"
grep -q '"ok": true' "$READY_FILE"

XWAM_PHASE=eval_client_pool_srun
set +e
srun --overlap --nodes=1 --ntasks=1 --gpus-per-node=4 --cpus-per-task=48 \
  --environment="$ROBOCASA_EDF" \
  env \
    REPO="$REPO" \
    ROBOCASA_ROOT="$ROBOCASA_ROOT" \
    ROBOSUITE_ROOT="$ROBOSUITE_ROOT" \
    TOPOLOGY="$TOPOLOGY" \
    RESULT_ROOT="$RESULT_ROOT" \
    LOG_ROOT="$LOG_ROOT" \
    EPISODES_PER_TASK="$EPISODES_PER_TASK" \
  bash -lc '
set -Eeuo pipefail
cd "$REPO"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$ROBOCASA_ROOT:$ROBOSUITE_ROOT:$REPO${PYTHONPATH:+:$PYTHONPATH}"
python evaluation/launch_robocasa365_m6_client_pool.py \
  --topology "$TOPOLOGY" \
  --output-root "$RESULT_ROOT" \
  --log-root "$LOG_ROOT" \
  --episodes-per-task "$EPISODES_PER_TASK"
'
CLIENT_RC=$?
set -e

touch "$STOP_FILE"
XWAM_PHASE=eval_policy_pool_shutdown
set +e
wait "$POLICY_STEP_PID"
POLICY_RC=$?
set -e
POLICY_STEP_PID=""

if [[ "$CLIENT_RC" -ne 0 ]]; then
  xwam_fail "One or more simulator clients failed; rerun the same EVAL_ID to resume"
fi
if [[ "$POLICY_RC" -ne 0 ]]; then
  xwam_fail "One or more policy servers failed; inspect per-server reports"
fi

XWAM_PHASE=eval_aggregate_srun
srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=1 \
  --environment="$XWAM_EDF" \
  env REPO="$REPO" TOPOLOGY="$TOPOLOGY" RESULT_ROOT="$RESULT_ROOT" \
    EPISODES_PER_TASK="$EPISODES_PER_TASK" EVAL_ID="$EVAL_ID" \
    CHECKPOINT="$CHECKPOINT" SUMMARY_JSON="$SUMMARY_JSON" \
    SUMMARY_CSV="$SUMMARY_CSV" \
  bash -lc '
set -Eeuo pipefail
cd "$REPO"
python scripts/aggregate_robocasa365_m6_evaluation.py \
  --topology "$TOPOLOGY" \
  --output-root "$RESULT_ROOT" \
  --episodes-per-task "$EPISODES_PER_TASK" \
  --eval-id "$EVAL_ID" \
  --checkpoint "$CHECKPOINT" \
  --output "$SUMMARY_JSON" \
  --csv-output "$SUMMARY_CSV"
'

grep -q '"ok": true' "$SUMMARY_JSON"
grep -q '"result": "pass"' "$SUMMARY_JSON"
rm -f "$STOP_FILE" "$READY_FILE" "$EXPECTED_CONTRACT"
rmdir "$CONTROL_ROOT" "$EVAL_ROOT/.runtime" 2>/dev/null || true
trap - EXIT INT TERM
echo "[PASS] X-WAM M6 evaluation completed: $SUMMARY_JSON"
