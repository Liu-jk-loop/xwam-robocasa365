#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=450G
#SBATCH --time=06:00:00
#SBATCH --job-name=xwam_a9_pm85k
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/atomic9-pointmap-85k-eval-%j.log
#SBATCH --error=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/atomic9-pointmap-85k-eval-%j.err
#SBATCH --export=ALL

set -Eeuo pipefail

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
DEPLOY_CAPSCR=/capstor/scratch/cscs/zjingchen/terry_nys
DEPLOY_IOPS=/iopsstor/scratch/cscs/zjingchen/terry_nys
REPO="${XWAM_EVAL_REPO:-$DEPLOY_STORE/src/xwam-robocasa365-eval}"
EXP_NAME=robocasa365_atomic9_fastwam_overlap_ratio00_pointmap_seed42_8gpu
CHECKPOINT_STEP=8500
CHECKPOINT_ROOT="${XWAM_POINTMAP_STEP8500_CHECKPOINT_ROOT:-$DEPLOY_IOPS/xwam_run/$EXP_NAME/checkpoints}"
EXACT_CHECKPOINT="${XWAM_POINTMAP_STEP8500_CHECKPOINT:-}"

# 8.5k是滚动保存点而非Store的3000-step永久点。默认从IOPS训练目录精确查找，
# 如已手工复制到其他目录，可覆盖CHECKPOINT_ROOT或直接传入EXACT_CHECKPOINT。
checkpoint_complete() {
  local candidate="$1"
  local rank
  local rank_matches=()
  local optimizer_shards=()
  [[ -d "$candidate" ]] || return 1
  [[ -s "$candidate/checkpoint/mp_rank_00_model_states.pt" ]] || return 1
  shopt -s nullglob
  optimizer_shards=("$candidate"/checkpoint/*zero_pp_rank_*_mp_rank_00_optim_states.pt)
  shopt -u nullglob
  [[ "${#optimizer_shards[@]}" -eq 8 ]] || return 1
  for rank in $(seq 0 7); do
    shopt -s nullglob
    rank_matches=("$candidate"/checkpoint/*zero_pp_rank_"$rank"_mp_rank_00_optim_states.pt)
    shopt -u nullglob
    [[ "${#rank_matches[@]}" -eq 1 ]] || return 1
    [[ -s "${rank_matches[0]}" ]] || return 1
  done
}

CHECKPOINT=""
if [[ -n "$EXACT_CHECKPOINT" ]]; then
  checkpoint_name="$(basename "$EXACT_CHECKPOINT")"
  if [[ ! "$checkpoint_name" =~ ^(epoch=[0-9]+-step=|final-step=)${CHECKPOINT_STEP}\.ckpt$ ]]; then
    echo "[FAIL] exact checkpoint basename does not match step $CHECKPOINT_STEP: $checkpoint_name" >&2
    exit 2
  fi
  if ! checkpoint_complete "$EXACT_CHECKPOINT"; then
    echo "[FAIL] exact checkpoint is incomplete: $EXACT_CHECKPOINT" >&2
    exit 2
  fi
  CHECKPOINT="$EXACT_CHECKPOINT"
else
  FINAL_CANDIDATE="$CHECKPOINT_ROOT/final-step=$CHECKPOINT_STEP.ckpt"
  if checkpoint_complete "$FINAL_CANDIDATE"; then
    CHECKPOINT="$FINAL_CANDIDATE"
  else
    shopt -s nullglob
    epoch_candidates=("$CHECKPOINT_ROOT"/epoch=*-step="$CHECKPOINT_STEP".ckpt)
    shopt -u nullglob
    complete_candidates=()
    for candidate in "${epoch_candidates[@]}"; do
      if checkpoint_complete "$candidate"; then
        complete_candidates+=("$candidate")
      fi
    done
    if [[ "${#complete_candidates[@]}" -ne 1 ]]; then
      echo "[FAIL] expected exactly one complete epoch=*-step=$CHECKPOINT_STEP.ckpt under $CHECKPOINT_ROOT; found ${#complete_candidates[@]}" >&2
      echo "[HINT] step8500 is rolling; submit before later saves evict it or set XWAM_POINTMAP_STEP8500_CHECKPOINT" >&2
      exit 2
    fi
    CHECKPOINT="${complete_candidates[0]}"
  fi
fi

export XWAM_EVAL_REPO="$REPO"
export XWAM_EVAL_MANIFEST="$DEPLOY_STORE/manifests/xwam/atomic9_ratio00/robocasa365_atomic9_fastwam_overlap_manifest.json"
export XWAM_EVAL_STATS="$DEPLOY_STORE/manifests/xwam/atomic9_ratio00/robocasa365_atomic9_fastwam_overlap_global_stats.json"
export XWAM_EVAL_TOPOLOGY="$REPO/configs/evaluation/robocasa365_atomic9_pointmap_step8500_6server_9client.json"
export XWAM_EVAL_EXPERIMENT_DIR="$DEPLOY_CAPSCR/experiments/xwam/$EXP_NAME"
export XWAM_EVAL_CHECKPOINT="$CHECKPOINT"
export XWAM_EVAL_ID="${XWAM_POINTMAP_STEP8500_EVAL_ID:-atomic9_pointmap_step8500_seed42_target_50ep}"
export XWAM_EVAL_EPISODES="${XWAM_POINTMAP_EVAL_EPISODES:-50}"
export XWAM_EVAL_NAMESPACE=atomic9-pointmap
export XWAM_EVAL_SUMMARY_STEM=atomic9
export XWAM_EVAL_REQUIRE_DEPTH=false
export XWAM_EVAL_REQUIRE_POINTMAP=true

# GPU0/1各承载两个server，GPU2/3各承载一个server；9个client各负责一个任务。
# 全部任务仅评测seed 42～91。PointMap是训练期辅助监督，在线闭环仍只输入RGB和proprio。
exec bash "$REPO/deployment/clariden/eval_m6_atomic18_xwam.sbatch"
