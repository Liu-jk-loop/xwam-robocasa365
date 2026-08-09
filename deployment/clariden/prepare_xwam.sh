#!/usr/bin/env bash
set -euo pipefail

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
DEPLOY_CAPSCR=/capstor/scratch/cscs/zjingchen/terry_nys
DEPLOY_IOPS=/iopsstor/scratch/cscs/zjingchen/terry_nys
XWAM_REPO="$DEPLOY_STORE/src/xwam-robocasa365"
XWAM_CHECKPOINT_ROOT="$DEPLOY_STORE/checkpoints/xwam"
WAN_DIR="$XWAM_CHECKPOINT_ROOT/wan22_5b"
WAN_SOURCE="$DEPLOY_STORE/checkpoints/fastwam/models/Wan-AI/Wan2.2-TI2V-5B"
TOKENIZER_SOURCE="$DEPLOY_STORE/checkpoints/fastwam/models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"
WAN_REVISION=921dbaf3f1674a56f47e83fb80a34bac8a8f203e
XWAM_CHECKPOINT_REVISION=bb6fd1643cfa8bdc751612a7ace0bd8062a8917a

mkdir -p \
  "$DEPLOY_STORE/build/xwam" \
  "$DEPLOY_STORE/checkpoints/xwam" \
  "$DEPLOY_STORE/containers/edf" \
  "$DEPLOY_STORE/logs/xwam" \
  "$DEPLOY_STORE/manifests" \
  "$DEPLOY_CAPSCR/containers" \
  "$DEPLOY_IOPS/cache/huggingface" \
  "$DEPLOY_IOPS/cache/torch_extensions"

if [[ ! -d "$XWAM_REPO/.git" ]]; then
  git clone \
    --branch dev/atomic-robocasa365 \
    https://github.com/Liu-jk-loop/xwam-robocasa365.git \
    "$XWAM_REPO"
fi

git -C "$XWAM_REPO" fetch origin dev/atomic-robocasa365
git -C "$XWAM_REPO" checkout dev/atomic-robocasa365
git -C "$XWAM_REPO" pull --ff-only origin dev/atomic-robocasa365
git -C "$XWAM_REPO" status --short --branch
git -C "$XWAM_REPO" branch --show-current
git -C "$XWAM_REPO" rev-parse HEAD | tee "$DEPLOY_STORE/manifests/xwam_commit.txt"
git -C "$XWAM_REPO" remote -v

mkdir -p "$WAN_DIR/google"
for DEPLOY_FILE in \
  Wan2.2_VAE.pth \
  diffusion_pytorch_model-00001-of-00003.safetensors \
  diffusion_pytorch_model-00002-of-00003.safetensors \
  diffusion_pytorch_model-00003-of-00003.safetensors \
  models_t5_umt5-xxl-enc-bf16.pth; do
  test -s "$WAN_SOURCE/$DEPLOY_FILE"
  if [[ ! -e "$WAN_DIR/$DEPLOY_FILE" ]]; then
    ln -s "$WAN_SOURCE/$DEPLOY_FILE" "$WAN_DIR/$DEPLOY_FILE"
  fi
done

test -s "$TOKENIZER_SOURCE/spiece.model"
if [[ ! -e "$WAN_DIR/google/umt5-xxl" ]]; then
  ln -s "$TOKENIZER_SOURCE" "$WAN_DIR/google/umt5-xxl"
fi

for DEPLOY_FILE in \
  config.json \
  configuration.json \
  diffusion_pytorch_model.safetensors.index.json; do
  if [[ ! -s "$WAN_DIR/$DEPLOY_FILE" ]]; then
    curl \
      --fail \
      --location \
      --retry 5 \
      --output "$WAN_DIR/$DEPLOY_FILE.partial" \
      "https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/resolve/$WAN_REVISION/$DEPLOY_FILE?download=true"
    mv "$WAN_DIR/$DEPLOY_FILE.partial" "$WAN_DIR/$DEPLOY_FILE"
  fi
done

PRETRAINED_ROOT="$XWAM_CHECKPOINT_ROOT/pretrained"
PRETRAINED_MODEL="$PRETRAINED_ROOT/checkpoints/last.ckpt/checkpoint/mp_rank_00_model_states.pt"
mkdir -p "$(dirname "$PRETRAINED_MODEL")"

if [[ ! -s "$PRETRAINED_ROOT/config.yaml" ]]; then
  curl \
    --fail \
    --location \
    --retry 5 \
    --output "$PRETRAINED_ROOT/config.yaml.partial" \
    "https://huggingface.co/sharinka0715/X-WAM-checkpoints/resolve/$XWAM_CHECKPOINT_REVISION/pretrained/config.yaml?download=true"
  mv "$PRETRAINED_ROOT/config.yaml.partial" "$PRETRAINED_ROOT/config.yaml"
fi

if [[ ! -s "$PRETRAINED_MODEL" ]]; then
  curl \
    --continue-at - \
    --fail \
    --location \
    --retry 10 \
    --retry-delay 10 \
    --output "$PRETRAINED_MODEL.partial" \
    "https://huggingface.co/sharinka0715/X-WAM-checkpoints/resolve/$XWAM_CHECKPOINT_REVISION/pretrained/checkpoints/last.ckpt/checkpoint/mp_rank_00_model_states.pt?download=true"
  mv "$PRETRAINED_MODEL.partial" "$PRETRAINED_MODEL"
fi

test -s "$WAN_DIR/config.json"
test -s "$WAN_DIR/diffusion_pytorch_model.safetensors.index.json"
test -s "$PRETRAINED_MODEL"

printf '%s\n' \
  "wan_revision=$WAN_REVISION" \
  "wan_dir=$WAN_DIR" \
  "xwam_checkpoint_revision=$XWAM_CHECKPOINT_REVISION" \
  "xwam_pretrained=$PRETRAINED_ROOT/checkpoints/last.ckpt" \
  > "$DEPLOY_STORE/manifests/xwam_models.txt"

echo "[PASS] X-WAM source and checkpoint layout prepared"
