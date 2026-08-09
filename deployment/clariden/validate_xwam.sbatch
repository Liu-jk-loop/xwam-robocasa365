#!/usr/bin/env bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --job-name=xwam_validate
#SBATCH --output=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/logs/xwam/validate-%j.log

set -euo pipefail

DEPLOY_STORE=/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys
EDF="$DEPLOY_STORE/containers/edf/xwam.toml"

srun --environment="$EDF" bash -lc '
set -euo pipefail
nvidia-smi
python - <<"PY"
import json
import platform
import sys
from pathlib import Path

import torch
import numpy
import flash_attn
from flash_attn import flash_attn_func

from modules.attention import HAS_FLASH_ATTN

store = Path("/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys")
wan = store / "checkpoints/xwam/wan22_5b"
pretrained = store / "checkpoints/xwam/pretrained/checkpoints/last.ckpt/checkpoint/mp_rank_00_model_states.pt"
dataset = store / "datasets/robocasa/v1.0/pretrain/atomic/CloseFridge/20250819/lerobot"

print("node_arch:", platform.machine())
print("python:", sys.version)
print("torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("NumPy:", numpy.__version__)
print("FlashAttention:", flash_attn.__version__)
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))

assert platform.machine() == "aarch64"
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 4
assert torch.__version__.startswith("2.9.0")
assert torch.version.cuda == "12.6"
assert numpy.__version__ == "1.23.5"
assert flash_attn.__version__ == "2.8.3"
assert HAS_FLASH_ATTN

q = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
out = flash_attn_func(q, q, q, causal=False)
out.float().square().mean().backward()
torch.cuda.synchronize()
assert out.shape == q.shape
assert torch.isfinite(out).all()
assert q.grad is not None and torch.isfinite(q.grad).all()

required = [
    wan / "config.json",
    wan / "configuration.json",
    wan / "diffusion_pytorch_model.safetensors.index.json",
    wan / "diffusion_pytorch_model-00001-of-00003.safetensors",
    wan / "diffusion_pytorch_model-00002-of-00003.safetensors",
    wan / "diffusion_pytorch_model-00003-of-00003.safetensors",
    wan / "models_t5_umt5-xxl-enc-bf16.pth",
    wan / "google/umt5-xxl/spiece.model",
    wan / "Wan2.2_VAE.pth",
    pretrained,
    dataset / "meta/info.json",
]
missing = [str(path) for path in required if not path.is_file()]
assert not missing, json.dumps(missing, indent=2)

print("[PASS] X-WAM GH200 CUDA, FlashAttention, model and dataset discovery")
PY
'
