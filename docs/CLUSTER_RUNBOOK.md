# Starlight cluster runbook

## Repository checkout

After the development branch is published:

```bash
git clone --branch dev/atomic-robocasa365 --single-branch \
  https://github.com/Liu-jk-loop/xwam-robocasa365.git

cd xwam-robocasa365
git submodule update --init --recursive
git rev-parse HEAD
git status --short --branch
```

Always include the `git rev-parse HEAD` value in feedback.

`git submodule update --init --recursive` downloads the exact third-party revisions referenced by the main repository. In this project they are RoboCasa, robosuite, and RoboTwin. It is normally required once after the first clone; later runs only need it when the referenced submodule commits change.

## M1 metadata audit

After the M1 batch-1 commit is published, replace the placeholders and run:

```bash
python scripts/audit_robocasa365_dataset.py \
  --dataset /ABSOLUTE/PATH/TO/ONE/ROBOCASA365/TASK \
  --task-name CloseFridge \
  --require-data \
  --require-videos \
  --output /tmp/robocasa365_close_fridge_audit.json
```

This command does not import Torch or decode video. It verifies official metadata, the 16D state, 12D action, three RGB camera keys, atomic-only task membership, and the presence of Parquet/MP4 files.

## External model paths

Use the existing complete Wan2.2 model:

```text
/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/Wan-AI/Wan2.2-TI2V-5B
```

Use the X-WAM pretrained checkpoint:

```text
/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/x-wam/xwam_checkpoints/pretrained/checkpoints/last.ckpt
```

Verify without loading Torch:

```bash
WAN_DIR=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/Wan-AI/Wan2.2-TI2V-5B
XWAM_CKPT=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/x-wam/xwam_checkpoints/pretrained/checkpoints/last.ckpt

test -s "$WAN_DIR/config.json"
test -s "$WAN_DIR/Wan2.2_VAE.pth"
test -s "$WAN_DIR/models_t5_umt5-xxl-enc-bf16.pth"
test -s "$XWAM_CKPT/checkpoint/mp_rank_00_model_states.pt"
```

Do not copy Wan2.2 into the repository. Runtime configuration will point to the existing directory.

## Validation levels

### Local-static

- Python syntax compilation without imports.
- Skill validation.
- Documentation/change-record check.
- Dependency-free unit tests and schema fixtures.

### Cluster-smoke

- Import Torch/CUDA/FlashAttention/DeepSpeed.
- Load model components and checkpoint.
- Load one real dataset batch.
- Run one forward/backward step.
- Create one simulator environment and complete one rollout.

### Cluster-train

- Multi-GPU initialization.
- Measured memory and throughput.
- Save and resume checkpoint.
- H100 training and scheduled atomic evaluation.

Commands for runtime smoke tests will be added with the implementation that they validate; this prevents stale placeholder commands from being treated as supported.

## Hardware roles

- A100/A800: environment bring-up, memory probing, one-batch training, short overfit, inference, and rollout debugging.
- H100: training after the A100/A800 smoke gate passes.
- Record exact GPU memory capacity; A100 40GB and 80GB require different profiles.

## Feedback

Copy `.agents/skills/xwam-robocasa365-workflow/references/cluster-feedback-template.md`, fill every applicable field, and attach the relevant log tail. Do not report only the final exception line.
