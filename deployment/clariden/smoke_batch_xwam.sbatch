#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --job-name=xwam_batch
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/batch-%j.log

set -euo pipefail

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
EDF="$DEPLOY_STORE/containers/edf/xwam.toml"
DATASET="$DEPLOY_STORE/datasets/robocasa/v1.0/pretrain/atomic/CloseFridge/20250819/lerobot"
REPORT="$DEPLOY_STORE/logs/xwam/clariden_batch-${SLURM_JOB_ID}.json"

test -s "$EDF"
test -s "$DATASET/meta/info.json"

srun --environment="$EDF" bash -lc "
set -euo pipefail
git rev-parse HEAD
git status --short
python scripts/audit_robocasa365_batch.py \\
  --dataset '$DATASET' \\
  --task-name CloseFridge \\
  --index 0 \\
  --num-workers 0 \\
  --output '$REPORT'
"

test -s "$REPORT"
echo "report=$REPORT"
echo "[PASS] X-WAM Clariden real RoboCasa365 batch"
