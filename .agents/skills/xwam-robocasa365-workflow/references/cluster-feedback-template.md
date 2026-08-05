# Cluster feedback template

## Identity

- Date/time and timezone:
- Git commit SHA:
- Branch:
- Slurm/Kubernetes job or pod ID:
- Operator:

## Objective

- Milestone/task:
- Expected acceptance evidence:
- Result: `pass`, `fail`, or `blocked`

## Command and configuration

- Working directory:
- Exact command:
- Resolved configuration path or attached configuration:
- Dataset root and manifest/task subset:
- Wan2.2 path:
- X-WAM checkpoint path and initialization mode:
- Output directory:

## Environment

- Host/container image:
- Python:
- PyTorch:
- CUDA runtime:
- NVIDIA driver:
- GPU model, count, and memory:
- DeepSpeed:
- FlashAttention:
- RoboCasa/robosuite/MuJoCo revisions for simulator work:

## Observed behavior

- Stage reached:
- Tensor shapes relevant to the issue:
- Peak allocated/reserved GPU memory:
- Throughput or step latency:
- Loss/metric sample:
- Last successful operation:
- First failing operation:

## Logs

Attach the full scheduler log when practical. Otherwise include the complete traceback and at least 100 lines before the first exception. Do not include secrets or access tokens.

## Artifacts

- Saved checkpoint/result/video paths:
- Small diagnostic files attached:
- Can the failure be reproduced with the same command and commit?
- Any manual changes made on the cluster checkout:
