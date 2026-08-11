#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --job-name=xwam_m6_8gpu
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/m6-formal-8gpu-%j.log
#SBATCH --export=ALL

set -Eeuo pipefail

export XWAM_M6_NUM_NODES=2
export XWAM_M6_DEVICES_PER_NODE=4
export XWAM_M6_HARDWARE_CONFIG=configs/hardware/gh200x8_96gb_gbs128.yaml
export XWAM_M6_FORMAL_NAME=robocasa365_m6_atomic_seen18_rgb_seed42_8gpu
export XWAM_M6_LOG_STEM=m6-formal-8gpu

exec bash /capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/src/xwam-robocasa365/deployment/clariden/train_m6_formal_xwam.sbatch
