#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --job-name=xwam_build
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/build-%j.log

set -euo pipefail

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
DEPLOY_CAPSCR=/capstor/scratch/cscs/zjingchen/terry_nys
BUILD_DIR="$DEPLOY_STORE/build/xwam"
MANIFEST_DIR="$DEPLOY_STORE/manifests"
XWAM_SHA="$(<"$MANIFEST_DIR/xwam_commit.txt")"
IMAGE_TAG="xwam-ngc2410-cu126-${XWAM_SHA:0:8}"
SQSH="$DEPLOY_CAPSCR/containers/${IMAGE_TAG}.sqsh"
EDF="$DEPLOY_STORE/containers/edf/xwam.toml"

echo "node=$(hostname)"
echo "arch=$(uname -m)"
echo "xwam_sha=$XWAM_SHA"
echo "image_tag=$IMAGE_TAG"
echo "sqsh=$SQSH"
podman info
podman images
df -h /dev/shm "$DEPLOY_CAPSCR"

podman build \
  --pull=always \
  --build-arg "XWAM_SHA=$XWAM_SHA" \
  --tag "localhost/$IMAGE_TAG:latest" \
  --file "$BUILD_DIR/Containerfile" \
  "$BUILD_DIR"

podman run \
  --rm \
  --env NVIDIA_VISIBLE_DEVICES=void \
  "localhost/$IMAGE_TAG:latest" \
  python -c 'import sys, torch, numpy, flash_attn, decord, diffusers, transformers, deepspeed, huggingface_hub, safetensors; print("python:", sys.version); print("torch:", torch.__version__); print("torch CUDA build:", torch.version.cuda); print("numpy:", numpy.__version__); print("flash-attn:", flash_attn.__version__); print("decord:", decord.__version__); print("diffusers:", diffusers.__version__); print("transformers:", transformers.__version__); print("huggingface-hub:", huggingface_hub.__version__); print("safetensors:", safetensors.__version__); print("deepspeed:", deepspeed.__version__)'

if [[ -e "$SQSH" ]]; then
  echo "Refusing to overwrite existing SQSH: $SQSH" >&2
  exit 2
fi

ENROOT_STATUS=0
enroot import \
  -x mount \
  -o "$SQSH" \
  "podman://localhost/$IMAGE_TAG:latest" || ENROOT_STATUS=$?

if (( ENROOT_STATUS != 0 )); then
  if [[ -s "$SQSH" ]] && unsquashfs -s "$SQSH" >/dev/null; then
    echo "[WARN] enroot import exited $ENROOT_STATUS after producing a valid SQSH; continuing finalization" >&2
  else
    echo "enroot import failed with status $ENROOT_STATUS and produced no valid SQSH" >&2
    exit "$ENROOT_STATUS"
  fi
fi

test -s "$SQSH"
sed "s|@SQSH@|$SQSH|g" "$BUILD_DIR/xwam.toml.template" > "$EDF"

{
  printf 'xwam_sha=%s\n' "$XWAM_SHA"
  printf 'base_image=%s\n' 'nvcr.io/nvidia/pytorch:24.10-py3'
  printf 'image_tag=%s\n' "$IMAGE_TAG"
  printf 'sqsh=%s\n' "$SQSH"
  printf 'edf=%s\n' "$EDF"
  printf 'build_job=%s\n' "${SLURM_JOB_ID:-unknown}"
  printf 'build_node=%s\n' "$(hostname)"
  printf 'build_date=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$MANIFEST_DIR/xwam_container.txt"

podman run \
  --rm \
  --env NVIDIA_VISIBLE_DEVICES=void \
  "localhost/$IMAGE_TAG:latest" \
  python -m pip freeze > "$MANIFEST_DIR/xwam_environment.txt"

ls -lh "$SQSH" "$EDF" "$MANIFEST_DIR/xwam_container.txt" "$MANIFEST_DIR/xwam_environment.txt"
echo "[PASS] X-WAM image built and exported before allocation exit"
