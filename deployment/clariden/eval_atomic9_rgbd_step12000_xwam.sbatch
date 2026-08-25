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
DEPLOY_CAPSCR=/capstor/scratch/cscs/zjingchen/terry_nys
REPO="$DEPLOY_STORE/src/xwam-robocasa365-eval"
EXP_NAME=robocasa365_atomic9_fastwam_overlap_ratio00_rgbd_seed42_8gpu
CHECKPOINT_ROOT="${XWAM_RGBD_EVAL_CHECKPOINT_ROOT:-$DEPLOY_STORE/checkpoints/xwam/$EXP_NAME/checkpoints}"
CHECKPOINT_STEP="${XWAM_RGBD_EVAL_STEP:-12000}"
EXACT_CHECKPOINT="${XWAM_RGBD_EVAL_CHECKPOINT:-}"

if [[ ! "$CHECKPOINT_STEP" =~ ^[1-9][0-9]*$ ]]; then
  echo "[FAIL] XWAM_RGBD_EVAL_STEP must be a positive integer" >&2
  exit 2
fi

# 只接受含model state与8份ZeRO optimizer shard的完整目标step。优先final，
# 否则要求epoch命名的候选唯一，避免把“最近checkpoint”误当成目标权重。
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
      exit 2
    fi
    CHECKPOINT="${complete_candidates[0]}"
  fi
fi

export XWAM_EVAL_REPO="$REPO"
export XWAM_EVAL_MANIFEST="$DEPLOY_STORE/manifests/xwam/atomic9_ratio00/robocasa365_atomic9_fastwam_overlap_manifest.json"
export XWAM_EVAL_STATS="$DEPLOY_STORE/manifests/xwam/atomic9_ratio00/robocasa365_atomic9_fastwam_overlap_global_stats.json"
export XWAM_EVAL_TOPOLOGY="$REPO/configs/evaluation/robocasa365_atomic9_rgbd_step12000_6server_9client.json"
export XWAM_EVAL_EXPERIMENT_DIR="$DEPLOY_CAPSCR/experiments/xwam/$EXP_NAME"
export XWAM_EVAL_CHECKPOINT="$CHECKPOINT"
export XWAM_EVAL_ID="${XWAM_RGBD_EVAL_ID:-atomic9_rgbd_step${CHECKPOINT_STEP}_seed42_target_50ep}"
export XWAM_EVAL_EPISODES="${XWAM_RGBD_EVAL_EPISODES:-50}"
export XWAM_EVAL_NAMESPACE=atomic9-rgbd
export XWAM_EVAL_SUMMARY_STEM=atomic9
export XWAM_EVAL_REQUIRE_DEPTH=true

# GPU0: server0/1；GPU1: server2/3；GPU2: server4；GPU3: server5。
# 9个client各执行一个Atomic9任务，全部只跑seed 42～91。
exec bash "$REPO/deployment/clariden/eval_m6_atomic18_xwam.sbatch"
