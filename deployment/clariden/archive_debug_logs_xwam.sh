#!/usr/bin/env bash

set -Eeuo pipefail

DEPLOY_STORE="${XWAM_DEPLOY_STORE:-/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys}"
LOG_ROOT="$DEPLOY_STORE/logs/xwam"
DEBUG_LOG_ROOT="$LOG_ROOT/debug"

test -d "$LOG_ROOT"
mkdir -p "$DEBUG_LOG_ROOT"

debug_patterns=(
  "build-*"
  "validate-*"
  "batch-*"
  "clariden_batch-*"
  "overlay-*"
  "wandb-overlay-*"
  "train1-*"
  "train4-resume-*"
  "m6-data-*"
  "m6-gate-*"
)

shopt -s nullglob
moved=0
for pattern in "${debug_patterns[@]}"; do
  for source in "$LOG_ROOT"/$pattern; do
    target="$DEBUG_LOG_ROOT/$(basename "$source")"
    if [[ -e "$target" ]]; then
      printf '[FAIL] Debug archive target already exists: %s\n' "$target" >&2
      exit 1
    fi
    mv -- "$source" "$target"
    printf '[MOVE] %s -> %s\n' "$source" "$target"
    moved=$((moved + 1))
  done
done
shopt -u nullglob

printf '[INFO] Files remaining at the formal log root:\n'
find "$LOG_ROOT" -maxdepth 1 -type f -print | sort
printf '[PASS] Archived %d X-WAM debug log files under %s\n' \
  "$moved" "$DEBUG_LOG_ROOT"
