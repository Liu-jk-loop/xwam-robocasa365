#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=450G
#SBATCH --time=12:00:00
#SBATCH --job-name=xwam_r05_55v70
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/atomic9-ratio05-55v70-eval-%j.log
#SBATCH --error=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/atomic9-ratio05-55v70-eval-%j.err
#SBATCH --export=ALL

set -Eeuo pipefail

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
DEPLOY_IOPS=/iopsstor/scratch/cscs/zjingchen/terry_nys
export XWAM_ATOMIC9_EXP_NAME=robocasa365_atomic9_fastwam_overlap_ratio05_rgb_seed42_8gpu
export XWAM_ATOMIC9_CHECKPOINT_ROOT="${XWAM_RATIO05_CHECKPOINT_ROOT:-$DEPLOY_IOPS/xwam_run/$XWAM_ATOMIC9_EXP_NAME/checkpoints}"
export XWAM_EVAL_ID="${XWAM_EVAL_ID:-atomic9_ratio05_step5500_vs_step7000_seed42_50ep}"

exec bash "$DEPLOY_STORE/src/xwam-robocasa365-eval/deployment/clariden/eval_atomic9_step5500_vs_step7000_xwam.sbatch"
