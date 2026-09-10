#!/usr/bin/env bash

# 星光单节点4卡 Atomic9 RGB-D step 12000闭环评测。
# 直接在已分配4张GPU的节点或容器内运行；不依赖Clariden EDF/Enroot。

set -Eeuo pipefail

BASE="${XWAM_STL_BASE:-/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang}"
REPO="${XWAM_STL_REPO:-$BASE/xwam-robocasa365-eval}"
POLICY_ENV="${XWAM_STL_POLICY_ENV:-xwam-robocasa365}"
SIMULATOR_ENV="${XWAM_STL_SIMULATOR_ENV:-robocasa}"
MODEL_ROOT="${XWAM_STL_MODEL_ROOT:-$BASE/models/x-wam/finetuned/robocasa365_atomic9_ratio00_rgbd_seed42}"
EXPERIMENT_DIR="${XWAM_STL_EXPERIMENT_DIR:-$MODEL_ROOT}"
CHECKPOINT="${XWAM_STL_CHECKPOINT:-$MODEL_ROOT/checkpoints/epoch=8-step=12000.ckpt}"
WAN_DIR="${XWAM_STL_WAN_DIR:-$BASE/models/Wan-AI/Wan2.2-TI2V-5B}"
MANIFEST="${XWAM_STL_MANIFEST:-$BASE/manifests/xwam/atomic9_ratio00/robocasa365_atomic9_fastwam_overlap_manifest.json}"
STATS="${XWAM_STL_STATS:-$BASE/manifests/xwam/atomic9_ratio00/robocasa365_atomic9_fastwam_overlap_global_stats.json}"
TOPOLOGY="${XWAM_STL_TOPOLOGY:-$REPO/configs/evaluation/robocasa365_atomic9_rgbd_step12000_6server_9client.json}"
EVAL_ID="${XWAM_STL_EVAL_ID:-atomic9_rgbd_step12000_seed42_target_50ep}"
EPISODES_PER_TASK="${XWAM_STL_EPISODES:-50}"
OUTPUT_BASE="${XWAM_STL_OUTPUT_BASE:-$BASE/experiments/xwam-eval/atomic9-rgbd}"
STARTUP_TIMEOUT_SECONDS="${XWAM_STL_STARTUP_TIMEOUT_SECONDS:-5400}"
EXPECTED_GPUS=4

EVAL_ROOT="$OUTPUT_BASE/$EVAL_ID"
RESULT_ROOT="$EVAL_ROOT/results"
LOG_ROOT="$EVAL_ROOT/logs"
CONTROL_ROOT="$EVAL_ROOT/.runtime/${EVAL_ID}.$$"
READY_FILE="$CONTROL_ROOT/policy-ready.json"
STOP_FILE="$CONTROL_ROOT/stop-policy"
SUMMARY_JSON="$EVAL_ROOT/aggregate.json"
SUMMARY_CSV="$EVAL_ROOT/summary_atomic9.csv"
FAILURE_REPORT="$LOG_ROOT/failure.txt"
POLICY_PROBE="$LOG_ROOT/policy_environment.json"
SIMULATOR_PROBE="$LOG_ROOT/simulator_environment.json"
EXPECTED_CONTRACT="$CONTROL_ROOT/evaluation_contract.expected.txt"
POLICY_PID=""
CLIENT_PID=""
PHASE=outer_preflight

mkdir -p "$RESULT_ROOT" "$LOG_ROOT/clients" "$LOG_ROOT/servers" "$CONTROL_ROOT"
exec > >(tee -a "$LOG_ROOT/job.log") 2> >(tee -a "$LOG_ROOT/job.err" >&2)

write_failure() {
  local exit_code="$1"
  local line="$2"
  local command="$3"
  [[ -s "$FAILURE_REPORT" ]] && return 0
  command="${command//$'\n'/ }"
  {
    printf 'timestamp=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'phase=%s\n' "$PHASE"
    printf 'exit_code=%s\n' "$exit_code"
    printf 'line=%s\n' "$line"
    printf 'command=%q\n' "${command:0:1000}"
    printf 'main_log=%s\n' "$LOG_ROOT/job.log"
  } >"$FAILURE_REPORT.tmp.$$"
  mv "$FAILURE_REPORT.tmp.$$" "$FAILURE_REPORT"
}

on_error() {
  local exit_code="$1"
  local line="$2"
  local command="$3"
  trap - ERR
  set +e
  write_failure "$exit_code" "$line" "$command"
  printf '[FAIL] 星光X-WAM评测退出：phase=%s exit_code=%s line=%s\n' \
    "$PHASE" "$exit_code" "$line" >&2
  printf '[FAIL] failure_report=%s\n' "$FAILURE_REPORT" >&2
  exit "$exit_code"
}

fail() {
  local message="$1"
  write_failure 2 "${BASH_LINENO[0]:-unknown}" "$message"
  printf '[FAIL] %s\n' "$message" >&2
  exit 2
}

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  set +e
  touch "$STOP_FILE"
  if [[ -n "$CLIENT_PID" ]] && kill -0 "$CLIENT_PID" 2>/dev/null; then
    kill -TERM "$CLIENT_PID" 2>/dev/null || true
    wait "$CLIENT_PID" 2>/dev/null || true
  fi
  if [[ -n "$POLICY_PID" ]] && kill -0 "$POLICY_PID" 2>/dev/null; then
    for _ in $(seq 1 60); do
      kill -0 "$POLICY_PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$POLICY_PID" 2>/dev/null; then
      kill -TERM "$POLICY_PID" 2>/dev/null || true
    fi
    wait "$POLICY_PID" 2>/dev/null || true
  fi
  rm -f "$READY_FILE" "$STOP_FILE" "$EXPECTED_CONTRACT"
  rmdir "$CONTROL_ROOT" 2>/dev/null || true
  exit "$exit_code"
}

trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR
trap cleanup EXIT INT TERM

[[ "$EVAL_ID" =~ ^[A-Za-z0-9._-]+$ ]] || fail "XWAM_STL_EVAL_ID只能包含A-Za-z0-9._-"
[[ "$EPISODES_PER_TASK" =~ ^[1-9][0-9]*$ ]] || fail "XWAM_STL_EPISODES必须为正整数"
[[ "$STARTUP_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail "启动超时必须为正整数秒"

exec 9>"$OUTPUT_BASE/.${EVAL_ID}.lock"
flock -n 9 || fail "已有同名评测正在写入：$EVAL_ROOT"
rm -f "$FAILURE_REPORT"

CONDA_BIN="${XWAM_STL_CONDA_BIN:-${CONDA_EXE:-}}"
if [[ -z "$CONDA_BIN" ]]; then
  CONDA_BIN="$(command -v conda || true)"
fi
[[ -n "$CONDA_BIN" && -x "$CONDA_BIN" ]] || fail "找不到conda；可设置XWAM_STL_CONDA_BIN"

test -d "$REPO" || fail "评测仓库不存在：$REPO"
test -s "$EXPERIMENT_DIR/config.yaml" || fail "缺少训练resolved config：$EXPERIMENT_DIR/config.yaml"
grep -Eq '^[[:space:]]*use_depth:[[:space:]]*true[[:space:]]*$' "$EXPERIMENT_DIR/config.yaml" \
  || fail "RGB-D checkpoint要求config.yaml中use_depth=true"
test -s "$WAN_DIR/config.json" || fail "Wan2.2目录缺少config.json：$WAN_DIR"
test -s "$MANIFEST" || fail "Atomic9 manifest不存在：$MANIFEST"
test -s "$STATS" || fail "Atomic9 global stats不存在：$STATS"
test -s "$TOPOLOGY" || fail "Atomic9 topology不存在：$TOPOLOGY"

MODEL_STATE=""
if [[ -f "$CHECKPOINT" ]]; then
  MODEL_STATE="$CHECKPOINT"
elif [[ -s "$CHECKPOINT/checkpoint/mp_rank_00_model_states.pt" ]]; then
  MODEL_STATE="$CHECKPOINT/checkpoint/mp_rank_00_model_states.pt"
elif [[ -s "$CHECKPOINT/mp_rank_00_model_states.pt" ]]; then
  MODEL_STATE="$CHECKPOINT/mp_rank_00_model_states.pt"
fi
[[ -n "$MODEL_STATE" && -s "$MODEL_STATE" ]] \
  || fail "checkpoint缺少非空mp_rank_00_model_states.pt：$CHECKPOINT"

# 推理只读取model state；不要求DeepSpeed optimizer shard或训练resume状态。
MODEL_STATE_BYTES="$(stat -c '%s' "$MODEL_STATE")"
[[ "$MODEL_STATE_BYTES" -eq 26121537691 ]] \
  || fail "model state大小不符，期望26121537691 bytes，实际$MODEL_STATE_BYTES：$MODEL_STATE"

cd "$REPO"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || fail "评测目录不是Git仓库：$REPO"
{
  printf 'branch=%s\n' "$(git branch --show-current)"
  printf 'commit=%s\n' "$(git rev-parse HEAD)"
  git status --short
} >"$LOG_ROOT/repo-state.txt"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "[WARN] 评测工作区存在改动；不会阻塞运行，状态摘要已写入repo-state.txt"
fi

PHASE=policy_environment_probe
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  "$CONDA_BIN" run --no-capture-output -n "$POLICY_ENV" \
  python "$REPO/scripts/probe_starlight_eval_runtime.py" \
    --role policy \
    --expected-gpus "$EXPECTED_GPUS" \
    --manifest "$MANIFEST" \
    --statistics-path "$STATS" \
    --output "$POLICY_PROBE"

PHASE=simulator_environment_probe
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
MUJOCO_EGL_DEVICE_ID=0 \
EGL_DEVICE_ID=0 \
ROBOSUITE_RENDER_GPU_DEVICE_ID=0 \
  "$CONDA_BIN" run --no-capture-output -n "$SIMULATOR_ENV" \
  python "$REPO/scripts/probe_starlight_eval_runtime.py" \
    --role simulator \
    --expected-gpus "$EXPECTED_GPUS" \
    --output "$SIMULATOR_PROBE"

CONTRACT="$LOG_ROOT/evaluation_contract.txt"
{
  printf 'eval_id=%s\n' "$EVAL_ID"
  printf 'experiment_dir=%s\n' "$(realpath "$EXPERIMENT_DIR")"
  printf 'checkpoint=%s\n' "$(realpath "$CHECKPOINT")"
  printf 'model_state=%s\n' "$(realpath "$MODEL_STATE")"
  printf 'model_state_bytes=%s\n' "$MODEL_STATE_BYTES"
  printf 'episodes_per_task=%s\n' "$EPISODES_PER_TASK"
  printf 'policy_env=%s\n' "$POLICY_ENV"
  printf 'simulator_env=%s\n' "$SIMULATOR_ENV"
  printf 'repo_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'repo_status_sha256=%s\n' "$(git status --porcelain | sha256sum | awk '{print $1}')"
  printf 'config_sha256=%s\n' "$(sha256sum "$EXPERIMENT_DIR/config.yaml" | awk '{print $1}')"
  printf 'topology_sha256=%s\n' "$(sha256sum "$TOPOLOGY" | awk '{print $1}')"
  printf 'manifest_sha256=%s\n' "$(sha256sum "$MANIFEST" | awk '{print $1}')"
  printf 'stats_sha256=%s\n' "$(sha256sum "$STATS" | awk '{print $1}')"
} >"$EXPECTED_CONTRACT"
if [[ -e "$CONTRACT" ]]; then
  cmp -s "$EXPECTED_CONTRACT" "$CONTRACT" || {
    diff -u "$CONTRACT" "$EXPECTED_CONTRACT" >&2 || true
    fail "同一EVAL_ID不能混入另一权重、配置、统计或代码合同"
  }
else
  cp "$EXPECTED_CONTRACT" "$CONTRACT.tmp.$$"
  mv "$CONTRACT.tmp.$$" "$CONTRACT"
fi

echo "eval_id=$EVAL_ID"
echo "checkpoint=$CHECKPOINT"
echo "model_state=$MODEL_STATE"
echo "policy_env=$POLICY_ENV"
echo "simulator_env=$SIMULATOR_ENV"
echo "topology=4 GPU / 6 policy servers / 9 simulator clients"
echo "seed=42..$((41 + EPISODES_PER_TASK))"
echo "result_root=$RESULT_ROOT"

PHASE=policy_pool_start
rm -f "$READY_FILE" "$STOP_FILE"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONUNBUFFERED=1 \
TOKENIZERS_PARALLELISM=false \
  "$CONDA_BIN" run --no-capture-output -n "$POLICY_ENV" \
  python "$REPO/evaluation/launch_robocasa365_m6_policy_pool.py" \
    --topology "$TOPOLOGY" \
    --experiment-dir "$EXPERIMENT_DIR" \
    --checkpoint "$CHECKPOINT" \
    --wan-checkpoint-dir "$WAN_DIR" \
    --multitask-manifest "$MANIFEST" \
    --statistics-path "$STATS" \
    --log-root "$LOG_ROOT" \
    --ready-file "$READY_FILE" \
    --stop-file "$STOP_FILE" \
    --startup-timeout-seconds "$STARTUP_TIMEOUT_SECONDS" \
  >"$LOG_ROOT/server_launcher.log" 2>&1 &
POLICY_PID=$!

PHASE=policy_pool_ready
ready_deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
while (( SECONDS < ready_deadline )); do
  if [[ -s "$READY_FILE" ]] && grep -q '"result": "ready"' "$READY_FILE"; then
    break
  fi
  if ! kill -0 "$POLICY_PID" 2>/dev/null; then
    wait "$POLICY_PID" || true
    fail "policy pool在6个server全部READY前退出；检查server_launcher.log"
  fi
  sleep 2
done
test -s "$READY_FILE" || fail "policy pool READY超时；检查server_launcher.log"
grep -q '"ok": true' "$READY_FILE" || fail "policy pool READY报告未通过"

PHASE=simulator_client_pool
set +e
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONUNBUFFERED=1 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
MUJOCO_EGL_DEVICE_ID=0 \
EGL_DEVICE_ID=0 \
ROBOSUITE_RENDER_GPU_DEVICE_ID=0 \
  "$CONDA_BIN" run --no-capture-output -n "$SIMULATOR_ENV" \
  python "$REPO/evaluation/launch_robocasa365_m6_client_pool.py" \
    --topology "$TOPOLOGY" \
    --output-root "$RESULT_ROOT" \
    --log-root "$LOG_ROOT" \
    --episodes-per-task "$EPISODES_PER_TASK" \
    --cuda-visible-devices 0,1,2,3 \
  >"$LOG_ROOT/client_launcher.log" 2>&1 &
CLIENT_PID=$!
wait "$CLIENT_PID"
CLIENT_RC=$?
CLIENT_PID=""
set -e

touch "$STOP_FILE"
PHASE=policy_pool_shutdown
set +e
wait "$POLICY_PID"
POLICY_RC=$?
POLICY_PID=""
set -e

[[ "$CLIENT_RC" -eq 0 ]] || fail "一个或多个simulator client失败；同一EVAL_ID可续跑已完成episode"
[[ "$POLICY_RC" -eq 0 ]] || fail "一个或多个policy server失败；检查logs/servers"

PHASE=evaluation_aggregate
"$CONDA_BIN" run --no-capture-output -n "$POLICY_ENV" \
  python "$REPO/scripts/aggregate_robocasa365_m6_evaluation.py" \
    --topology "$TOPOLOGY" \
    --output-root "$RESULT_ROOT" \
    --episodes-per-task "$EPISODES_PER_TASK" \
    --eval-id "$EVAL_ID" \
    --checkpoint "$CHECKPOINT" \
    --output "$SUMMARY_JSON" \
    --csv-output "$SUMMARY_CSV"

grep -q '"ok": true' "$SUMMARY_JSON" || fail "聚合报告未通过"
grep -q '"result": "pass"' "$SUMMARY_JSON" || fail "聚合结果不是pass"
rm -f "$FAILURE_REPORT" "$STOP_FILE" "$READY_FILE"
rmdir "$CONTROL_ROOT" 2>/dev/null || true
trap - EXIT INT TERM
echo "[PASS] 星光Atomic9 RGB-D step12000评测完成：$SUMMARY_JSON"
