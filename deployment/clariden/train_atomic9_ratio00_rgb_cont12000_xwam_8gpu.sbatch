#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --job-name=xwam_a9_rgb_cont
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/atomic9-rgb-cont12000-8gpu-%j.log
#SBATCH --export=ALL

set -Eeuo pipefail

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
DEPLOY_IOPS=/iopsstor/scratch/cscs/zjingchen/terry_nys
RATIO00_EVIDENCE="$DEPLOY_STORE/manifests/xwam/atomic9_ratio00"
CONT_EVIDENCE="$DEPLOY_STORE/manifests/xwam/atomic9_rgb_cont7500_12000"
SOURCE_EXP=robocasa365_atomic9_fastwam_overlap_ratio00_rgb_seed42_8gpu

export XWAM_M6_NUM_NODES=2
export XWAM_M6_DEVICES_PER_NODE=4
export XWAM_M6_HARDWARE_CONFIG=configs/hardware/gh200x8_96gb_gbs128.yaml
export XWAM_M6_DATA_CONFIG=configs/data/robocasa365_atomic9_fastwam_overlap.yaml
export XWAM_M6_EXPERIMENT_CONFIG=configs/experiment/robocasa365_atomic9_ratio00_gh200_rgb_cont7500_12000.yaml
export XWAM_M6_MANIFEST="$RATIO00_EVIDENCE/robocasa365_atomic9_fastwam_overlap_manifest.json"
export XWAM_M6_STATS="$RATIO00_EVIDENCE/robocasa365_atomic9_fastwam_overlap_global_stats.json"
export XWAM_M6_PREFLIGHT="$CONT_EVIDENCE/robocasa365_atomic9_ratio00_rgb_cont7500_12000_preflight.json"
export XWAM_M6_TOTAL_STEPS=12000
export XWAM_M6_BOOTSTRAP_STEP=7500
export XWAM_M6_BOOTSTRAP_HOT_CHECKPOINT_ROOT="$DEPLOY_IOPS/xwam_run/$SOURCE_EXP/checkpoints"
export XWAM_M6_BOOTSTRAP_DURABLE_CHECKPOINT_ROOT="$DEPLOY_STORE/checkpoints/xwam/$SOURCE_EXP/checkpoints"
export XWAM_M6_FORMAL_NAME=robocasa365_atomic9_ratio00_rgb_cont7500_12000_seed42_8gpu
export XWAM_M6_LOG_STEM=atomic9-ratio00-rgb-cont7500-12000-formal-8gpu

exec bash "$DEPLOY_STORE/src/xwam-robocasa365/deployment/clariden/train_m6_formal_xwam.sbatch"
