#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --job-name=xwam_a9_rgbd_p1
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/atomic9-rgbd-p1-%j.log

set -Eeuo pipefail

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
REPO="$DEPLOY_STORE/src/xwam-robocasa365"
EDF="$DEPLOY_STORE/containers/edf/xwam.toml"
DATA_ROOT="$DEPLOY_STORE/datasets/robocasa/v1.0/pretrain/atomic"
EVIDENCE_ROOT="$DEPLOY_STORE/manifests/xwam/atomic9_rgbd"
REPORT="$EVIDENCE_ROOT/robocasa365_atomic9_rgbd_replay_inputs.json"
TASK_MANIFEST=configs/tasks/robocasa365_atomic9_fastwam_overlap.json

cd "$REPO"
test -s "$EDF"
test -d "$DATA_ROOT"
test -z "$(git status --porcelain)" || {
  echo "[FAIL] Git worktree must be clean before RGB-D provenance audit" >&2
  git status --short >&2
  exit 2
}
mkdir -p "$EVIDENCE_ROOT"

srun --kill-on-bad-exit=1 --environment="$EDF" \
  env REPO="$REPO" DATA_ROOT="$DATA_ROOT" REPORT="$REPORT" TASK_MANIFEST="$TASK_MANIFEST" \
  bash -lc '
set -Eeuo pipefail
cd "$REPO"
git rev-parse HEAD
test -z "$(git status --porcelain)"

python scripts/audit_robocasa365_depth_replay_inputs.py \
  --dataset-root "$DATA_ROOT" \
  --task-manifest "$TASK_MANIFEST" \
  --episodes-to-decode 3 \
  --output "$REPORT"
'

test -s "$REPORT"
grep -q '"ok": true' "$REPORT"
grep -q '"result": "pass"' "$REPORT"
grep -q '"passed_tasks": 9' "$REPORT"

echo "rgbd_replay_inputs=$REPORT"
echo "[PASS] Atomic9 RGB-D replay inputs structurally ready"
