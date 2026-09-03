#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --job-name=xwam_a9_r00_pm
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/atomic9-ratio00-pointmap-8gpu-%j.log
#SBATCH --export=ALL

set -Eeuo pipefail

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
DEPLOY_IOPS=/iopsstor/scratch/cscs/zjingchen/terry_nys
RATIO00_EVIDENCE="$DEPLOY_STORE/manifests/xwam/atomic9_ratio00"
POINTMAP_EVIDENCE="$DEPLOY_STORE/manifests/xwam/pointmap/atomic9"
SCHEDULE_ENV="$POINTMAP_EVIDENCE/robocasa365_atomic9_ratio00_pointmap_14000step_schedule.env"

test -s "$SCHEDULE_ENV"
# shellcheck disable=SC1090
source "$SCHEDULE_ENV"
[[ "$XWAM_SCHEDULE_TOTAL_STEPS" == 14000 ]]
[[ "$XWAM_SCHEDULE_MILESTONE_EPOCH" == 5 ]]

export XWAM_M6_NUM_NODES=2
export XWAM_M6_DEVICES_PER_NODE=4
export XWAM_M6_MODEL_CONFIG=configs/model/wan22_5b_robocasa365_atomic_pointmap.yaml
export XWAM_M6_HARDWARE_CONFIG=configs/hardware/gh200x8_96gb_gbs128_pointmap.yaml
export XWAM_M6_DATA_CONFIG=configs/data/robocasa365_atomic9_fastwam_overlap_pointmap.yaml
export XWAM_M6_EXPERIMENT_CONFIG=configs/experiment/robocasa365_atomic9_ratio00_gh200_pointmap.yaml
export XWAM_M6_MANIFEST="$RATIO00_EVIDENCE/robocasa365_atomic9_fastwam_overlap_manifest.json"
export XWAM_M6_STATS="$RATIO00_EVIDENCE/robocasa365_atomic9_fastwam_overlap_global_stats.json"
export XWAM_M6_PREFLIGHT="$POINTMAP_EVIDENCE/robocasa365_atomic9_ratio00_pointmap_14000step_preflight.json"
export XWAM_M6_GATE_AUDIT="$DEPLOY_STORE/manifests/xwam/pointmap/close_fridge_training/close_fridge_pointmap_resume_gate.json"
export XWAM_M6_GATE_REQUIRE_ZERO=false
export XWAM_M6_EXPECT_DEPTH=false
export XWAM_M6_EXPECT_POINTMAP=true
export XWAM_M6_POINTMAP_CACHE_ROOT="$DEPLOY_IOPS/x-wam-pointmap-cache/robocasa365_camera_xyz_flexpi_v1/atomic"
export XWAM_M6_POINTMAP_MANIFEST="$POINTMAP_EVIDENCE/robocasa365_atomic9_pointmap_cache_manifest.json"
export XWAM_M6_POINTMAP_AUDIT="$POINTMAP_EVIDENCE/robocasa365_atomic9_pointmap_cache_audit.json"
export XWAM_M6_TOTAL_STEPS="$XWAM_SCHEDULE_TOTAL_STEPS"
export XWAM_M6_MILESTONE_STEP="$XWAM_SCHEDULE_MILESTONE_STEP"
export XWAM_M6_FORMAL_NAME=robocasa365_atomic9_fastwam_overlap_ratio00_pointmap_seed42_8gpu
export XWAM_M6_LOG_STEM=atomic9-ratio00-pointmap-formal-8gpu

exec bash "$DEPLOY_STORE/src/xwam-robocasa365/deployment/clariden/train_m6_formal_xwam.sbatch"
