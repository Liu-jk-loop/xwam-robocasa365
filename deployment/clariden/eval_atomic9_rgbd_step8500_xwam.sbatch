#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=450G
#SBATCH --time=12:00:00
#SBATCH --job-name=xwam_a9_d85k
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/atomic9-rgbd-85k-eval-%j.log
#SBATCH --error=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/atomic9-rgbd-85k-eval-%j.err
#SBATCH --export=ALL

set -Eeuo pipefail

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
DEPLOY_IOPS=/iopsstor/scratch/cscs/zjingchen/terry_nys
EXP_NAME=robocasa365_atomic9_fastwam_overlap_ratio00_rgbd_seed42_8gpu

export XWAM_RGBD_EVAL_STEP=8500
# step 8500不是3000-step永久保存点，默认只能从训练IOPS滚动目录查找。
# 如果此前已把8.5k复制到其他目录，提交前覆盖XWAM_RGBD_STEP8500_CHECKPOINT_ROOT。
export XWAM_RGBD_EVAL_CHECKPOINT_ROOT="${XWAM_RGBD_STEP8500_CHECKPOINT_ROOT:-$DEPLOY_IOPS/xwam_run/$EXP_NAME/checkpoints}"
export XWAM_RGBD_EVAL_ID="${XWAM_RGBD_STEP8500_EVAL_ID:-atomic9_rgbd_step8500_seed42_target_50ep}"

exec bash "$DEPLOY_STORE/src/xwam-robocasa365-eval/deployment/clariden/eval_atomic9_rgbd_step12000_xwam.sbatch"
