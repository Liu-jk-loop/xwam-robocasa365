#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --job-name=xwam_a9_r00_rgbd
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/atomic9-ratio00-rgbd-8gpu-%j.log
#SBATCH --export=ALL

set -Eeuo pipefail

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
DEPLOY_IOPS=/iopsstor/scratch/cscs/zjingchen/terry_nys
RATIO00_EVIDENCE="$DEPLOY_STORE/manifests/xwam/atomic9_ratio00"
RGBD_EVIDENCE="$DEPLOY_STORE/manifests/xwam/atomic9_rgbd"
SCHEDULE_ENV="$RGBD_EVIDENCE/robocasa365_atomic9_ratio00_rgbd_14000step_schedule.env"

test -s "$SCHEDULE_ENV"
# shellcheck disable=SC1090
source "$SCHEDULE_ENV"
[[ "$XWAM_SCHEDULE_TOTAL_STEPS" == 14000 ]]
[[ "$XWAM_SCHEDULE_MILESTONE_EPOCH" == 5 ]]

export XWAM_M6_NUM_NODES=2
export XWAM_M6_DEVICES_PER_NODE=4
export XWAM_M6_MODEL_CONFIG=configs/model/wan22_5b_robocasa365_atomic_rgbd.yaml
export XWAM_M6_HARDWARE_CONFIG=configs/hardware/gh200x8_96gb_gbs128_rgbd.yaml
export XWAM_M6_DATA_CONFIG=configs/data/robocasa365_atomic9_fastwam_overlap_rgbd.yaml
export XWAM_M6_EXPERIMENT_CONFIG=configs/experiment/robocasa365_atomic9_ratio00_gh200_rgbd.yaml
export XWAM_M6_MANIFEST="$RATIO00_EVIDENCE/robocasa365_atomic9_fastwam_overlap_manifest.json"
export XWAM_M6_STATS="$RATIO00_EVIDENCE/robocasa365_atomic9_fastwam_overlap_global_stats.json"
export XWAM_M6_PREFLIGHT="$RGBD_EVIDENCE/robocasa365_atomic9_ratio00_rgbd_14000step_preflight.json"
export XWAM_M6_GATE_AUDIT="$RGBD_EVIDENCE/close_fridge_rgbd_resume_3108551.json"
export XWAM_M6_GATE_REQUIRE_ZERO=false
export XWAM_M6_EXPECT_DEPTH=true
export XWAM_M6_DEPTH_CACHE_ROOT="$DEPLOY_IOPS/x-wam-depth-cache/robocasa365_inverse_metric_global_q_v1/atomic"
export XWAM_M6_DEPTH_ENCODING="$RGBD_EVIDENCE/robocasa365_inverse_metric_global_q_v1.json"
export XWAM_M6_DEPTH_MANIFEST="$RGBD_EVIDENCE/robocasa365_atomic9_rgbd_cache_manifest.json"
export XWAM_M6_DEPTH_AUDIT="$RGBD_EVIDENCE/robocasa365_atomic9_rgbd_cache_audit.json"
export XWAM_M6_TOTAL_STEPS="$XWAM_SCHEDULE_TOTAL_STEPS"
export XWAM_M6_MILESTONE_STEP="$XWAM_SCHEDULE_MILESTONE_STEP"
export XWAM_M6_FORMAL_NAME=robocasa365_atomic9_fastwam_overlap_ratio00_rgbd_seed42_8gpu
export XWAM_M6_LOG_STEM=atomic9-ratio00-rgbd-formal-8gpu

exec bash "$DEPLOY_STORE/src/xwam-robocasa365/deployment/clariden/train_m6_formal_xwam.sbatch"
