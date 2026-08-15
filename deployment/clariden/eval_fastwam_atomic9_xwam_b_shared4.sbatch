#!/usr/bin/env bash
#SBATCH --job-name=fw_a9_xwam_b
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=96
#SBATCH --mem=450G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/shared-fastwam-xwam-b-%j.log
#SBATCH --error=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/shared-fastwam-xwam-b-%j.err
#SBATCH --export=ALL

set -Eeuo pipefail

# 这个外层作业复用已有FastWAM Atomic9脚本，并在同一4卡allocation中并发运行
# CloseFridge X-WAM-B（ratio00）评测：
#   GPU 0/1/2：FastWAM的6个policy server与9个simulator client
#   GPU 3：一个X-WAM-B policy server与一个CloseFridge simulator client
#
# 两个子评测拥有独立端口、日志和结果目录。任一侧失败后，外层不会主动终止
# 另一侧；只有用户取消整个Slurm作业时才同时清理两个driver。

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
DEPLOY_CAPSCR=/capstor/scratch/cscs/zjingchen/terry_nys
DEPLOY_IOPS=/iopsstor/scratch/cscs/zjingchen/terry_nys

FASTWAM_ROOT="${FASTWAM_ROOT:-$DEPLOY_STORE/src/FastWAM}"
FASTWAM_EVAL_SCRIPT="${FASTWAM_EVAL_SCRIPT:-$FASTWAM_ROOT/eval_robocasa365_atomic9_6s9c_reserve4_use3.sbatch}"

XWAM_B_REPO="${XWAM_B_EVAL_REPO:-$DEPLOY_STORE/src/xwam-robocasa365-eval}"
XWAM_B_EVAL_SCRIPT="$XWAM_B_REPO/deployment/clariden/eval_close_fridge_ab_xwam.sbatch"
XWAM_B_EVAL_ID="${XWAM_B_EVAL_ID:-}"
XWAM_B_EXPERIMENT_DIR="${XWAM_B_EXPERIMENT_DIR:-$DEPLOY_CAPSCR/experiments/xwam/close_fridge_rgb_ratio00_seed42_4gpu}"
XWAM_B_CHECKPOINT="${XWAM_B_CHECKPOINT:-$DEPLOY_STORE/checkpoints/xwam/close_fridge_rgb_ratio00_seed42_4gpu/checkpoints/final-step=1000.ckpt}"
XWAM_B_EPISODES="${XWAM_B_EPISODES:-50}"

if [[ ! -s "$FASTWAM_EVAL_SCRIPT" ]]; then
  echo "[FAIL] FastWAM evaluation script not found: $FASTWAM_EVAL_SCRIPT" >&2
  echo "[HINT] Submit with FASTWAM_EVAL_SCRIPT=/absolute/path/to/eval_robocasa365_atomic9_6s9c_reserve4_use3.sbatch" >&2
  exit 2
fi
for expected in "export NUM_GPUS=3" "export NUM_SERVERS=6" "export NUM_CLIENTS=9"; do
  if ! grep -Fq "$expected" "$FASTWAM_EVAL_SCRIPT"; then
    echo "[FAIL] FastWAM script no longer matches the verified 3-GPU topology: $expected" >&2
    exit 2
  fi
done
if grep -Eq '^[[:space:]]*(export[[:space:]]+)?CUDA_VISIBLE_DEVICES=("3"|3)[[:space:]]*$' "$FASTWAM_EVAL_SCRIPT"; then
  echo "[FAIL] FastWAM script explicitly binds a process to GPU 3" >&2
  exit 2
fi
if [[ ! -s "$XWAM_B_EVAL_SCRIPT" ]]; then
  echo "[FAIL] X-WAM-B evaluation script not found: $XWAM_B_EVAL_SCRIPT" >&2
  exit 2
fi
if [[ -z "$XWAM_B_EVAL_ID" || ! "$XWAM_B_EVAL_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[FAIL] XWAM_B_EVAL_ID must be nonempty and contain only A-Za-z0-9._-" >&2
  exit 2
fi
if [[ ! "$XWAM_B_EPISODES" =~ ^[1-9][0-9]*$ ]]; then
  echo "[FAIL] XWAM_B_EPISODES must be a positive integer" >&2
  exit 2
fi

SHARED_LOG_ROOT="$DEPLOY_IOPS/x-wam-eval/shared-fastwam-xwam-b/$SLURM_JOB_ID"
mkdir -p "$SHARED_LOG_ROOT"
FASTWAM_DRIVER_LOG="$SHARED_LOG_ROOT/fastwam_driver.log"
XWAM_B_DRIVER_LOG="$SHARED_LOG_ROOT/xwam_b_driver.log"

# FastWAM脚本使用26000以上的动态端口；X-WAM单任务topology固定使用
# server 5的12005/13005，二者不重叠。启动前先拒绝已被其他进程占用的端口。
for port in 12005 13005; do
  if timeout 1 bash -c "</dev/tcp/127.0.0.1/${port}" 2>/dev/null; then
    echo "[FAIL] X-WAM-B port is already in use: $port" >&2
    exit 2
  fi
done

# 固定FastWAM端口，使外层可以等待它的6个server完成加载后再启动X-WAM-B。
export PORT_BASE="${PORT_BASE:-$((26000 + SLURM_JOB_ID % 3000))}"

cat <<EOF_INFO
================================================================
Shared FastWAM Atomic9 + X-WAM-B CloseFridge evaluation
job                 : $SLURM_JOB_ID
node                : $SLURM_JOB_NODELIST
FastWAM GPUs        : 0,1,2
FastWAM ports       : $PORT_BASE..$((PORT_BASE + 5))
FastWAM script      : $FASTWAM_EVAL_SCRIPT
FastWAM script sha  : $(sha256sum "$FASTWAM_EVAL_SCRIPT" | awk '{print $1}')
X-WAM-B GPU         : 3
X-WAM-B ports       : 12005 / 13005
X-WAM-B eval id     : $XWAM_B_EVAL_ID
X-WAM-B experiment  : $XWAM_B_EXPERIMENT_DIR
X-WAM-B checkpoint  : $XWAM_B_CHECKPOINT
X-WAM-B episodes    : $XWAM_B_EPISODES
shared logs         : $SHARED_LOG_ROOT
================================================================
EOF_INFO

FASTWAM_PID=""
XWAM_B_PID=""
cleanup() {
  local pid
  for pid in "${XWAM_B_PID:-}" "${FASTWAM_PID:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${XWAM_B_PID:-}" "${FASTWAM_PID:-}"; do
    if [[ -n "$pid" ]]; then
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

# 子脚本中的#SBATCH行在这里作为普通注释；所有srun都属于当前4卡allocation。
# 原FastWAM脚本已经逐进程设置CUDA_VISIBLE_DEVICES=0/1/2，永不使用3。
bash "$FASTWAM_EVAL_SCRIPT" >"$FASTWAM_DRIVER_LOG" 2>&1 &
FASTWAM_PID=$!

echo "[wait] waiting for FastWAM policy servers before starting X-WAM-B..."
FASTWAM_READY=0
ready_deadline=$((SECONDS + 1800))
while [[ "$SECONDS" -lt "$ready_deadline" ]]; do
  ready_count=0
  for offset in 0 1 2 3 4 5; do
    port=$((PORT_BASE + offset))
    if timeout 1 bash -c "</dev/tcp/127.0.0.1/${port}" 2>/dev/null; then
      ready_count=$((ready_count + 1))
    fi
  done
  if [[ "$ready_count" -eq 6 ]]; then
    FASTWAM_READY=1
    echo "[PASS] FastWAM servers ready; X-WAM-B may now load on GPU 3"
    break
  fi
  if ! kill -0 "$FASTWAM_PID" 2>/dev/null; then
    echo "[WARN] FastWAM driver exited before readiness; X-WAM-B will still run" >&2
    break
  fi
  sleep 2
done
if [[ "$FASTWAM_READY" -ne 1 ]] && kill -0 "$FASTWAM_PID" 2>/dev/null; then
  echo "[WARN] FastWAM readiness timed out; X-WAM-B will start without stopping FastWAM" >&2
fi

# 共享模式让X-WAM的srun step看到allocation的全部4卡，但policy和simulator
# 都只暴露逻辑GPU 3。默认checkpoint对应A/B中的ratio00（B组）。
env \
  XWAM_EVAL_REPO="$XWAM_B_REPO" \
  XWAM_EVAL_ID="$XWAM_B_EVAL_ID" \
  XWAM_EVAL_EXPERIMENT_DIR="$XWAM_B_EXPERIMENT_DIR" \
  XWAM_EVAL_CHECKPOINT="$XWAM_B_CHECKPOINT" \
  XWAM_EVAL_EPISODES="$XWAM_B_EPISODES" \
  XWAM_EVAL_STEP_GPUS=4 \
  XWAM_EVAL_CUDA_DEVICE=3 \
  bash "$XWAM_B_EVAL_SCRIPT" >"$XWAM_B_DRIVER_LOG" 2>&1 &
XWAM_B_PID=$!

# 故障域保持隔离：先结束的一侧失败不会触发另一侧cleanup。
set +e
wait "$FASTWAM_PID"
FASTWAM_RC=$?
FASTWAM_PID=""
wait "$XWAM_B_PID"
XWAM_B_RC=$?
XWAM_B_PID=""
set -e

trap - EXIT INT TERM

echo "FastWAM return code : $FASTWAM_RC"
echo "X-WAM-B return code : $XWAM_B_RC"
echo "FastWAM driver log  : $FASTWAM_DRIVER_LOG"
echo "X-WAM-B driver log  : $XWAM_B_DRIVER_LOG"

if [[ "$FASTWAM_RC" -ne 0 || "$XWAM_B_RC" -ne 0 ]]; then
  echo "[FAIL] shared evaluation finished with at least one failed side" >&2
  exit 1
fi

echo "[PASS] FastWAM Atomic9 and X-WAM-B CloseFridge evaluations completed"
