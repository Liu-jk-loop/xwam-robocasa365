#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=450G
#SBATCH --time=12:00:00
#SBATCH --job-name=xwam_a9_d13k
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/atomic9-rgbd-13k-eval-%j.log
#SBATCH --error=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/atomic9-rgbd-13k-eval-%j.err
#SBATCH --export=ALL

set -Eeuo pipefail

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
DEPLOY_IOPS=/iopsstor/scratch/cscs/zjingchen/terry_nys
EXP_NAME=robocasa365_atomic9_fastwam_overlap_ratio00_rgbd_seed42_8gpu
CHECKPOINT="${XWAM_RGBD_STEP13000_CHECKPOINT:-$DEPLOY_IOPS/xwam_run/$EXP_NAME/checkpoints/epoch=9-step=13000.ckpt}"

# 严格绑定用户指定的epoch=9-step=13000.ckpt，不搜索最近step或final checkpoint。
if [[ "$(basename "$CHECKPOINT")" != "epoch=9-step=13000.ckpt" ]]; then
  echo "[FAIL] RGB-D step13000 evaluation requires epoch=9-step=13000.ckpt" >&2
  exit 2
fi
export XWAM_RGBD_EVAL_STEP=13000
export XWAM_RGBD_EVAL_CHECKPOINT="$CHECKPOINT"
export XWAM_RGBD_EVAL_ID="${XWAM_RGBD_STEP13000_EVAL_ID:-atomic9_rgbd_step13000_seed42_target_50ep}"

# 复用已验证的4 GPU / 6 server / 9 client、seed42～91和50 episodes/task合同。
exec bash "$DEPLOY_STORE/src/xwam-robocasa365-eval/deployment/clariden/eval_atomic9_rgbd_step12000_xwam.sbatch"
