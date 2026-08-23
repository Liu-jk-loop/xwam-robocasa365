#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=450G
#SBATCH --time=12:00:00
#SBATCH --job-name=xwam_a9_d12k
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/atomic9-rgbd-12k-eval-%j.log
#SBATCH --error=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/atomic9-rgbd-12k-eval-%j.err
#SBATCH --export=ALL

set -Eeuo pipefail

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
EXP_NAME=robocasa365_atomic9_fastwam_overlap_ratio00_rgbd_seed42_8gpu

export XWAM_ATOMIC9_EXP_NAME="$EXP_NAME"
# 12000是3000-step durable周期的固定点；评测优先读取Store，避免与继续训练时
# IOPS rolling checkpoint的淘汰发生竞争。
export XWAM_ATOMIC9_CHECKPOINT_ROOT="$DEPLOY_STORE/checkpoints/xwam/$EXP_NAME/checkpoints"
export XWAM_ATOMIC9_TOPOLOGY="$DEPLOY_STORE/src/xwam-robocasa365-eval/configs/evaluation/robocasa365_atomic9_rgbd_step12000_seed42_92_8server_16client.json"
export XWAM_ATOMIC9_GROUP_A=step12000_seed42
export XWAM_ATOMIC9_GROUP_B=step12000_seed92
export XWAM_ATOMIC9_STEP_A=12000
export XWAM_ATOMIC9_STEP_B=12000
export XWAM_ATOMIC9_REQUIRE_DEPTH=true
export XWAM_EVAL_ID="${XWAM_RGBD_EVAL_ID:-atomic9_rgbd_step12000_seed42-141_target_100ep}"
export XWAM_EVAL_EPISODES="${XWAM_RGBD_EVAL_EPISODES:-50}"

exec bash "$DEPLOY_STORE/src/xwam-robocasa365-eval/deployment/clariden/eval_atomic9_step6500_vs_step7500_xwam.sbatch"
