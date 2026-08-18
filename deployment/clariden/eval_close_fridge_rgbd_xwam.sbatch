#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=450G
#SBATCH --time=12:00:00
#SBATCH --job-name=xwam_cf_rgbd_eval
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/close-fridge-rgbd-eval-%j.log
#SBATCH --error=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/close-fridge-rgbd-eval-%j.err
#SBATCH --export=ALL

set -Eeuo pipefail

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
DEPLOY_CAPSCR=/capstor/scratch/cscs/zjingchen/terry_nys
DEPLOY_IOPS=/iopsstor/scratch/cscs/zjingchen/terry_nys
# 评测使用独立clone，不与正在训练的主仓库共享工作区。
REPO="${XWAM_EVAL_REPO:-$DEPLOY_STORE/src/xwam-robocasa365-eval}"
ROBOCASA_ROOT="$DEPLOY_STORE/src/robocasa"
ROBOSUITE_ROOT="$DEPLOY_STORE/src/robosuite"
XWAM_EDF="$DEPLOY_STORE/containers/edf/xwam.toml"
ROBOCASA_EDF="$DEPLOY_STORE/containers/edf/robocasa365.toml"
RUNTIME_OVERLAY="$DEPLOY_IOPS/python/xwam-nvtx-0.2.15"
WAN_DIR="$DEPLOY_STORE/checkpoints/xwam/wan22_5b"
BASE_TOPOLOGY="$REPO/configs/evaluation/robocasa365_m6_atomic18_8server_16client.json"

EXPERIMENT_DIR="${XWAM_RGBD_EVAL_EXPERIMENT_DIR:-$DEPLOY_CAPSCR/experiments/xwam/close_fridge_rgbd_ratio00_seed42_8gpu}"
CHECKPOINT_ROOT="${XWAM_RGBD_EVAL_CHECKPOINT_ROOT:-$DEPLOY_IOPS/xwam_run/close_fridge_rgbd_ratio00_seed42_8gpu/checkpoints}"
CHECKPOINT_1000="${XWAM_RGBD_EVAL_CHECKPOINT_1000:-$CHECKPOINT_ROOT/epoch=5-step=1000.ckpt}"
CHECKPOINT_1500="${XWAM_RGBD_EVAL_CHECKPOINT_1500:-}"
EVAL_ID="${XWAM_RGBD_EVAL_ID:-step1000-1500_seed42-7_target_50ep}"
EPISODES="${XWAM_RGBD_EVAL_EPISODES:-50}"

if [[ -z "$EVAL_ID" || ! "$EVAL_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[FAIL] XWAM_RGBD_EVAL_ID must be nonempty and contain only A-Za-z0-9._-" >&2
  exit 2
fi
if [[ ! "$EPISODES" =~ ^[1-9][0-9]*$ ]]; then
  echo "[FAIL] XWAM_RGBD_EVAL_EPISODES must be a positive integer" >&2
  exit 2
fi

EVAL_ROOT="$DEPLOY_IOPS/x-wam-eval/close-fridge-rgbd/$EVAL_ID"
LOG_ROOT="$EVAL_ROOT/logs"
CONTROL_ROOT="$EVAL_ROOT/.runtime/$SLURM_JOB_ID"
COMPARISON_JSON="$EVAL_ROOT/comparison.json"
mkdir -p "$LOG_ROOT" "$CONTROL_ROOT"
exec > >(tee -a "$LOG_ROOT/job.log") 2> >(tee -a "$LOG_ROOT/job.err" >&2)

if [[ ! -s "$REPO/deployment/clariden/error_trap.sh" ]]; then
  echo "[FAIL] Independent evaluation repo is missing or incomplete: $REPO" >&2
  exit 2
fi
export XWAM_FAILURE_REPORT="$LOG_ROOT/failure.txt"
export XWAM_MAIN_LOG="$LOG_ROOT/job.log"
export XWAM_TRAIN_LOG=not-applicable
export XWAM_PHASE=close_fridge_rgbd_eval_preflight
source "$REPO/deployment/clariden/error_trap.sh"
xwam_install_err_trap

cd "$REPO"
test -s "$XWAM_EDF"
test -s "$ROBOCASA_EDF"
test -d "$RUNTIME_OVERLAY"
test -d "$EXPERIMENT_DIR"
test -s "$EXPERIMENT_DIR/config.yaml"
test -d "$CHECKPOINT_ROOT"
test -d "$CHECKPOINT_1000"
test -s "$CHECKPOINT_1000/checkpoint/mp_rank_00_model_states.pt"
test -s "$WAN_DIR/config.json"
test -s "$BASE_TOPOLOGY"
test -s "$ROBOCASA_ROOT/robocasa/models/assets/fixtures/sinks/Sink025/model.xml"
test -z "$(git status --porcelain)" || xwam_fail "Independent evaluation repo must be clean"

# epoch编号取决于实际clip数；只按global step匹配1500 checkpoint，不硬编epoch。
if [[ -z "$CHECKPOINT_1500" ]]; then
  shopt -s nullglob
  checkpoint_1500_matches=("$CHECKPOINT_ROOT"/epoch=*-step=1500.ckpt)
  shopt -u nullglob
  if [[ "${#checkpoint_1500_matches[@]}" -ne 1 ]]; then
    xwam_fail "Expected exactly one epoch=*-step=1500.ckpt under $CHECKPOINT_ROOT; found ${#checkpoint_1500_matches[@]}"
  fi
  CHECKPOINT_1500="${checkpoint_1500_matches[0]}"
fi
test -d "$CHECKPOINT_1500"
test -s "$CHECKPOINT_1500/checkpoint/mp_rank_00_model_states.pt"

RUN_NAMES=(step1000_seed42 step1000_seed7 step1500_seed42 step1500_seed7)
RUN_SEEDS=(42 7 42 7)
RUN_GPUS=(0 1 2 3)

mkdir -p "$DEPLOY_IOPS/x-wam-eval/.locks"
exec 9>"$DEPLOY_IOPS/x-wam-eval/.locks/close-fridge-rgbd-$EVAL_ID.lock"
flock -n 9 || xwam_fail "Another evaluation job is already writing $EVAL_ROOT"

CONTRACT="$LOG_ROOT/evaluation_contract.txt"
EXPECTED_CONTRACT="$CONTROL_ROOT/evaluation_contract.expected.txt"
{
  printf 'eval_id=%s\n' "$EVAL_ID"
  printf 'task=CloseFridge\n'
  printf 'experiment_dir=%s\n' "$(realpath "$EXPERIMENT_DIR")"
  printf 'checkpoint_root=%s\n' "$(realpath "$CHECKPOINT_ROOT")"
  printf 'checkpoint_1000=%s\n' "$(realpath "$CHECKPOINT_1000")"
  printf 'checkpoint_1500=%s\n' "$(realpath "$CHECKPOINT_1500")"
  printf 'matrix=step1000_seed42,step1000_seed7,step1500_seed42,step1500_seed7\n'
  printf 'episodes_per_run=%s\n' "$EPISODES"
  printf 'model_seed=42\nreplan_steps=20\naction_denoise_steps=10\nmax_steps=1000\n'
  printf 'repo_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'experiment_config_sha256=%s\n' "$(sha256sum "$EXPERIMENT_DIR/config.yaml" | awk '{print $1}')"
  printf 'base_topology_sha256=%s\n' "$(sha256sum "$BASE_TOPOLOGY" | awk '{print $1}')"
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
echo "checkpoint_root=$CHECKPOINT_ROOT"
echo "checkpoint_1000=$CHECKPOINT_1000"
echo "checkpoint_1500=$CHECKPOINT_1500"
echo "matrix=step1000/gpu0/seed42, step1000/gpu1/seed7, step1500/gpu2/seed42, step1500/gpu3/seed7"
echo "episodes_per_run=$EPISODES model_seed=42"
echo "result_root=$EVAL_ROOT"

SIMULATOR_PROBE="$LOG_ROOT/robocasa_eval_runtime_probe.json"
export XWAM_PHASE=close_fridge_rgbd_eval_simulator_probe
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

# 每路拷贝同一份已验证topology，仅隔离episode seed和broker端口。
# model_seed保持42，便于与既有RGB评测对齐。
PORT_BASE=$((20000 + SLURM_JOB_ID % 10000))
export XWAM_PHASE=close_fridge_rgbd_eval_topology
srun --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=1 \
  --environment="$XWAM_EDF" \
  env BASE_TOPOLOGY="$BASE_TOPOLOGY" CONTROL_ROOT="$CONTROL_ROOT" \
    PORT_BASE="$PORT_BASE" EPISODES="$EPISODES" \
  python - <<'PY'
import json
import os
from pathlib import Path

source = Path(os.environ["BASE_TOPOLOGY"])
root = Path(os.environ["CONTROL_ROOT"])
port_base = int(os.environ["PORT_BASE"])
episodes = int(os.environ["EPISODES"])
seeds = [42, 7, 42, 7]
names = ["step1000_seed42", "step1000_seed7", "step1500_seed42", "step1500_seed7"]

for run_index, (name, seed) in enumerate(zip(names, seeds, strict=True)):
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["name"] = f"close_fridge_rgbd_{name}"
    payload["model_seed"] = 42
    payload["seed_start"] = seed
    payload["episodes_per_task"] = episodes
    for server in payload["servers"]:
        server_id = int(server["server_id"])
        server["frontend_port"] = port_base + run_index * 20 + server_id
        server["backend_port"] = port_base + 1000 + run_index * 20 + server_id
    target = root / f"topology_{run_index}.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

for run_index in 0 1 2 3; do
  run_root="$EVAL_ROOT/${RUN_NAMES[$run_index]}"
  mkdir -p "$run_root/results" "$run_root/logs/clients" "$run_root/logs/servers" "$CONTROL_ROOT/run_$run_index"
done

POLICY_STEP_PID=""
cleanup_policy() {
  for run_index in 0 1 2 3; do
    touch "$CONTROL_ROOT/run_$run_index/stop-policy" 2>/dev/null || true
  done
  if [[ -n "$POLICY_STEP_PID" ]] && kill -0 "$POLICY_STEP_PID" 2>/dev/null; then
    wait "$POLICY_STEP_PID" 2>/dev/null || true
  fi
}
trap cleanup_policy EXIT INT TERM

export XWAM_PHASE=close_fridge_rgbd_eval_policy_start
srun --overlap --exact --nodes=1 --ntasks=1 --gpus-per-node=4 --cpus-per-task=32 \
  --environment="$XWAM_EDF" \
  env REPO="$REPO" RUNTIME_OVERLAY="$RUNTIME_OVERLAY" WAN_DIR="$WAN_DIR" \
    EXPERIMENT_DIR="$EXPERIMENT_DIR" \
    TOPOLOGY_0="$CONTROL_ROOT/topology_0.json" TOPOLOGY_1="$CONTROL_ROOT/topology_1.json" \
    TOPOLOGY_2="$CONTROL_ROOT/topology_2.json" TOPOLOGY_3="$CONTROL_ROOT/topology_3.json" \
    CHECKPOINT_0="$CHECKPOINT_1000" CHECKPOINT_1="$CHECKPOINT_1000" \
    CHECKPOINT_2="$CHECKPOINT_1500" CHECKPOINT_3="$CHECKPOINT_1500" \
    RUN_ROOT_0="$EVAL_ROOT/${RUN_NAMES[0]}" RUN_ROOT_1="$EVAL_ROOT/${RUN_NAMES[1]}" \
    RUN_ROOT_2="$EVAL_ROOT/${RUN_NAMES[2]}" RUN_ROOT_3="$EVAL_ROOT/${RUN_NAMES[3]}" \
    CONTROL_ROOT="$CONTROL_ROOT" \
  bash -lc '
set -Eeuo pipefail
cd "$REPO"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$RUNTIME_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"

pids=()
for run_index in 0 1 2 3; do
  topology_var="TOPOLOGY_$run_index"
  checkpoint_var="CHECKPOINT_$run_index"
  run_root_var="RUN_ROOT_$run_index"
  topology="${!topology_var}"
  checkpoint="${!checkpoint_var}"
  run_root="${!run_root_var}"
  python evaluation/launch_robocasa365_m6_policy_pool.py \
    --topology "$topology" \
    --experiment-dir "$EXPERIMENT_DIR" \
    --checkpoint "$checkpoint" \
    --wan-checkpoint-dir "$WAN_DIR" \
    --log-root "$run_root/logs" \
    --ready-file "$CONTROL_ROOT/run_$run_index/policy-ready.json" \
    --stop-file "$CONTROL_ROOT/run_$run_index/stop-policy" \
    --server-id 5 \
    --single-task-checkpoint \
    --single-task-name CloseFridge \
    --cuda-device "$run_index" \
    >"$run_root/logs/server_launcher.log" 2>&1 &
  pids+=("$!")
done

first_rc=0
finished_pid=""
while [[ -z "$finished_pid" ]]; do
  for pid in "${pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      set +e
      wait "$pid"
      first_rc=$?
      set -e
      finished_pid="$pid"
      break
    fi
  done
  [[ -n "$finished_pid" ]] || sleep 2
done
for run_index in 0 1 2 3; do
  touch "$CONTROL_ROOT/run_$run_index/stop-policy"
done
rc=$first_rc
for pid in "${pids[@]}"; do
  [[ "$pid" == "$finished_pid" ]] && continue
  wait "$pid" || rc=1
done
exit "$rc"
' &
POLICY_STEP_PID=$!

export XWAM_PHASE=close_fridge_rgbd_eval_policy_ready
for _ in $(seq 1 900); do
  ready_count=0
  for run_index in 0 1 2 3; do
    ready_file="$CONTROL_ROOT/run_$run_index/policy-ready.json"
    if [[ -s "$ready_file" ]] && grep -q '"result": "ready"' "$ready_file"; then
      ready_count=$((ready_count + 1))
    fi
  done
  if [[ "$ready_count" -eq 4 ]]; then
    break
  fi
  if ! kill -0 "$POLICY_STEP_PID" 2>/dev/null; then
    wait "$POLICY_STEP_PID" || true
    xwam_fail "One or more RGB-D policy servers exited before all four were READY"
  fi
  sleep 2
done
for run_index in 0 1 2 3; do
  ready_file="$CONTROL_ROOT/run_$run_index/policy-ready.json"
  status_file="$EVAL_ROOT/${RUN_NAMES[$run_index]}/logs/servers/server_5_status.json"
  test -s "$ready_file"
  grep -q '"ok": true' "$ready_file"
  test -s "$status_file"
  grep -q '"training_use_depth": true' "$status_file"
  grep -q '"inference_run_depth": false' "$status_file"
  grep -q '"online_depth_required": false' "$status_file"
done

export XWAM_PHASE=close_fridge_rgbd_eval_clients
if srun --overlap --exact --nodes=1 --ntasks=1 --gpus-per-node=4 --cpus-per-task=32 \
  --environment="$ROBOCASA_EDF" \
  env REPO="$REPO" ROBOCASA_ROOT="$ROBOCASA_ROOT" ROBOSUITE_ROOT="$ROBOSUITE_ROOT" \
    TOPOLOGY_0="$CONTROL_ROOT/topology_0.json" TOPOLOGY_1="$CONTROL_ROOT/topology_1.json" \
    TOPOLOGY_2="$CONTROL_ROOT/topology_2.json" TOPOLOGY_3="$CONTROL_ROOT/topology_3.json" \
    RUN_ROOT_0="$EVAL_ROOT/${RUN_NAMES[0]}" RUN_ROOT_1="$EVAL_ROOT/${RUN_NAMES[1]}" \
    RUN_ROOT_2="$EVAL_ROOT/${RUN_NAMES[2]}" RUN_ROOT_3="$EVAL_ROOT/${RUN_NAMES[3]}" \
    EPISODES="$EPISODES" \
  bash -lc '
set -Eeuo pipefail
cd "$REPO"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$ROBOCASA_ROOT:$ROBOSUITE_ROOT:$REPO${PYTHONPATH:+:$PYTHONPATH}"

pids=()
for run_index in 0 1 2 3; do
  topology_var="TOPOLOGY_$run_index"
  run_root_var="RUN_ROOT_$run_index"
  topology="${!topology_var}"
  run_root="${!run_root_var}"
  python evaluation/launch_robocasa365_m6_client_pool.py \
    --topology "$topology" \
    --output-root "$run_root/results" \
    --log-root "$run_root/logs" \
    --episodes-per-task "$EPISODES" \
    --client-id 5 \
    --cuda-visible-devices "$run_index" \
    >"$run_root/logs/client_launcher.log" 2>&1 &
  pids+=("$!")
done

rc=0
set +e
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done
exit "$rc"
'
then
  CLIENT_RC=0
else
  CLIENT_RC=$?
fi

for run_index in 0 1 2 3; do
  touch "$CONTROL_ROOT/run_$run_index/stop-policy"
done
export XWAM_PHASE=close_fridge_rgbd_eval_policy_shutdown
if wait "$POLICY_STEP_PID"; then
  POLICY_RC=0
else
  POLICY_RC=$?
fi
POLICY_STEP_PID=""

if [[ "$CLIENT_RC" -ne 0 ]]; then
  export XWAM_PHASE=close_fridge_rgbd_eval_clients
  xwam_fail "One or more CloseFridge RGB-D clients failed; inspect per-run client logs under $EVAL_ROOT"
fi
if [[ "$POLICY_RC" -ne 0 ]]; then
  export XWAM_PHASE=close_fridge_rgbd_eval_policy_shutdown
  xwam_fail "One or more CloseFridge RGB-D policy servers failed; inspect per-run server logs under $EVAL_ROOT"
fi

export XWAM_PHASE=close_fridge_rgbd_eval_summary
srun --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=1 \
  --environment="$XWAM_EDF" \
  env EVAL_ROOT="$EVAL_ROOT" COMPARISON_JSON="$COMPARISON_JSON" EPISODES="$EPISODES" \
    CHECKPOINT_1000="$CHECKPOINT_1000" CHECKPOINT_1500="$CHECKPOINT_1500" \
  python - <<'PY'
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

root = Path(os.environ["EVAL_ROOT"])
episodes_expected = int(os.environ["EPISODES"])
runs = [
    ("step1000_seed42", 1000, 42, os.environ["CHECKPOINT_1000"]),
    ("step1000_seed7", 1000, 7, os.environ["CHECKPOINT_1000"]),
    ("step1500_seed42", 1500, 42, os.environ["CHECKPOINT_1500"]),
    ("step1500_seed7", 1500, 7, os.environ["CHECKPOINT_1500"]),
]
rows = []
errors = []
for name, step, seed_start, checkpoint in runs:
    result_path = root / name / "results" / "CloseFridge" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    expected = {
        "result": "pass",
        "task": "CloseFridge",
        "episodes_expected": episodes_expected,
        "episodes_completed": episodes_expected,
        "model_seed": 42,
        "seed_start": seed_start,
        "max_steps": 1000,
        "replan_steps": 20,
        "action_denoise_steps": 10,
        "video_fps": 20,
    }
    run_errors = [
        f"{key}: expected={value!r}, actual={result.get(key)!r}"
        for key, value in expected.items()
        if result.get(key) != value
    ]
    successes = int(result.get("n_success", -1))
    success_rate = float(result.get("success_rate", -1.0))
    if not 0 <= successes <= episodes_expected:
        run_errors.append(f"n_success illegal: {successes}")
    elif not math.isclose(success_rate, successes / episodes_expected, rel_tol=0.0, abs_tol=1e-12):
        run_errors.append("success_rate does not match n_success / episodes")
    row = {
        "run": name,
        "checkpoint_step": step,
        "checkpoint": str(Path(checkpoint).resolve()),
        "seed_start": seed_start,
        "seed_end": seed_start + episodes_expected - 1,
        "episodes": episodes_expected,
        "successes": successes,
        "success_rate": success_rate,
        "success_percent": success_rate * 100.0,
        "mean_inference_time_s": result.get("mean_inference_time_s"),
        "ok": not run_errors,
        "errors": run_errors,
    }
    rows.append(row)
    errors.extend(f"{name}: {message}" for message in run_errors)
    summary_path = root / name / "summary.json"
    summary_path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

payload = {
    "schema_version": 1,
    "result": "pass" if not errors else "fail",
    "ok": not errors,
    "task": "CloseFridge",
    "training_modality": "rgbd",
    "online_inference_modalities": ["rgb", "proprio"],
    "model_seed": 42,
    "episodes_per_run": episodes_expected,
    "runs": rows,
    "errors": errors,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
}
Path(os.environ["COMPARISON_JSON"]).write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
raise SystemExit(0 if payload["ok"] else 1)
PY

grep -q '"ok": true' "$COMPARISON_JSON"
for run_index in 0 1 2 3; do
  rm -f "$CONTROL_ROOT/run_$run_index/stop-policy" "$CONTROL_ROOT/run_$run_index/policy-ready.json"
done
rm -f "$EXPECTED_CONTRACT"
trap - EXIT INT TERM
echo "[PASS] CloseFridge RGB-D 4-way checkpoint/seed evaluation completed: $COMPARISON_JSON"
