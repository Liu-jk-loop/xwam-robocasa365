# 星光超算操作手册

## 拉取仓库

开发分支发布后执行：

```bash
git clone --branch dev/atomic-robocasa365 --single-branch \
  https://github.com/Liu-jk-loop/xwam-robocasa365.git

cd xwam-robocasa365
git submodule update --init --recursive
git rev-parse HEAD
git status --short --branch
```

每次反馈都必须包含 `git rev-parse HEAD` 的输出。

`git submodule update --init --recursive` 会下载主仓库锁定版本的第三方代码。本项目包含 RoboCasa、robosuite 和 RoboTwin。通常首次 clone 后执行一次；以后只有主仓库引用的子模块 commit 发生变化时才需要再次执行。M1 metadata audit 不依赖这些子模块，可以暂缓到模拟器或评测环境配置阶段。

## M1 元数据审计

拉取 M1 第一批修改后，将占位路径替换为真实目录并执行：

```bash
python scripts/audit_robocasa365_dataset.py \
  --dataset /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/robocasa/robocasa/datasets/v1.0/pretrain/atomic/CloseFridge/20250819 \
  --task-name CloseFridge \
  --require-data \
  --require-videos \
  --output /tmp/robocasa365_close_fridge_audit.json
```

该命令不会导入 Torch，也不会解码视频。它负责检查官方元数据、16 维 state、12 维 action、三路 RGB 相机键、atomic-only 任务归属，以及 Parquet/MP4 文件是否存在。

执行完成后反馈以下内容：

1. `git rev-parse HEAD` 输出。
2. 完整 audit 命令和返回码。
3. `/tmp/robocasa365_close_fridge_audit.json` 内容。
4. 实际数据路径及 `meta/`、`data/`、`videos/` 的两级目录结构。

任务目录通常包含日期层，例如 `CloseFridge/20250819`。audit 的 `--dataset` 必须指向该日期目录或其中的 `lerobot/`，不能只指向 `pretrain/atomic/` 总目录。

## M1 原生 RGB-only batch 验收

本批新增 PyArrow 16.1.0，用于直接读取官方 LeRobot Parquet。已有独立环境不需要重建；拉取代码后在该环境内重新执行受约束安装器：

```bash
conda activate xwam-robocasa365
export TORCH_EXTENSIONS_DIR=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/.cache/torch_extensions/xwam-robocasa365

cd /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/xwam-robocasa365
git pull --ff-only origin dev/atomic-robocasa365
git rev-parse HEAD

bash scripts/install_starlight_dependencies.sh dry-run
bash scripts/install_starlight_dependencies.sh apply
```

该操作沿用现有约束，不替换 Torch 2.9.0、CUDA、FlashAttention 或 DeepSpeed。随后读取一个真实 `CloseFridge` clip 和 DataLoader batch：

```bash
python scripts/audit_robocasa365_batch.py \
  --dataset /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/robocasa/robocasa/datasets/v1.0/pretrain/atomic/CloseFridge/20250819 \
  --task-name CloseFridge \
  --index 0 \
  --num-workers 0 \
  --log-file logs/cluster/robocasa365_close_fridge_batch.json
```

预期核心证据：

- `video`: `[3, 9, 3, 256, 320]`、`float32`、范围在 `[-1,1]`。
- `proprios`: `[9,16]`；`actions`: `[32,12]`。
- `frame_ids`: `[0,4,8,12,16,20,24,28,32]`；`action_ids`: `[0..31]`。
- `camera_type_mask`: `[0,0,1]`；batch 不含 `depths`。
- 关闭 augmentation 后，同一索引重复读取完全一致。
- 最终 `result=pass`、`ok=true`，完整报告持久化在项目 `logs/cluster/`。

本轮不要运行 `scripts/train_sft.py`。M1 只验证原始 16D state 和 12D action 的读取、时序与维度；`configs/data/robocasa365.yaml` 以 `training_ready: false` 明确阻止在 M2 normalization/action schema 完成前训练。

反馈以下内容：commit SHA、完整命令和返回码、`logs/cluster/robocasa365_close_fridge_batch.json`。如果失败，保留完整 `error` 和 `traceback`，不要只截取最后一行。

## M2 PandaOmron schema 与 checkpoint 契约审计

M1 已在 commit `e4249b9` 通过。拉取 M2 第一批后，在 policy 环境执行：

```bash
conda activate xwam-robocasa365

cd /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/xwam-robocasa365
git pull --ff-only origin dev/atomic-robocasa365
git rev-parse HEAD

python scripts/audit_robocasa365_m2_contract.py \
  --dataset /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/robocasa/robocasa/datasets/v1.0/pretrain/atomic/CloseFridge/20250819 \
  --task-name CloseFridge \
  --checkpoint /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/x-wam/xwam_checkpoints \
  --episode-index 0 \
  --log-file logs/cluster/robocasa365_close_fridge_m2_contract.json
```

该命令完成四项只读检查：

1. 将真实 `meta/modality.json` 的每个名称、切片和 original key 与版本化 schema 对比。
2. 读取 `meta/stats.json` 的 state/action q01、q99、min、max，并报告退化 quantile 维度。
3. 对真实 episode 0 执行 state/action 未裁剪 round-trip，报告 component range、control-mode 值和训练 clip 比例。
4. 使用 FakeTensorMode 和 mmap 只读取公开 DeepSpeed checkpoint 元数据，报告 action/proprio encoder/decoder 边界 shape，不构造 5B 模型、不占用 GPU 权重显存。

预期所有 `checks` 为 `true`，尤其是：

- state/action 分别为 16D/12D。
- `control_mode` 只包含 `-1/+1`。
- unclipped round-trip 最大误差小于 `1e-5`。
- checkpoint action 边界为 legacy 14D，proprio 边界为 16D。
- 最终 `result=pass`、`ok=true`。

如果 `weights_only` 或 `mmap` 读取 checkpoint 失败，不要改为普通全量 `torch.load`；把完整报告反馈回来，避免不必要的 CPU 内存峰值。M2 contract 通过前仍不要启动训练。

## M2 12D checkpoint 加载与单步反传

M2 contract 已在 commit `95808cd` 通过。拉取 M2 第二批 commit 后，先只构造模型并验证公开 X-WAM 权重适配，不要直接启动训练：

```bash
conda activate xwam-robocasa365

cd /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/xwam-robocasa365
git pull --ff-only origin dev/atomic-robocasa365
git rev-parse HEAD

python scripts/audit_xwam_checkpoint_loading.py \
  --mode xwam_pretrained \
  --wan-checkpoint-dir /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/Wan-AI/Wan2.2-TI2V-5B \
  --xwam-checkpoint /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/x-wam/xwam_checkpoints \
  --output logs/cluster/xwam_pretrained_panda_omron_loading.json
```

预期报告应为 `result=pass`、`ok=true`，并满足：

- `action_remap` 恰好包含 action encoder 输入 weight、decoder 输出 weight/bias 三项。
- `reinitialize` 恰好包含 proprio encoder 输入和 decoder 输出的 weight/bias 四项。
- `missing_source`、`unexpected_source`、`shape_errors` 和 `errors` 均为空。
- RGB-only 目标不存在的 depth `extra_blocks/extra_heads` 只能进入 `discard_source`，不能被当作普通 missing 静默跳过。

然后验证不使用公开 X-WAM cross-embodiment 权重的 `wan_base` 消融初始化：

```bash
python scripts/audit_xwam_checkpoint_loading.py \
  --mode wan_base \
  --wan-checkpoint-dir /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/Wan-AI/Wan2.2-TI2V-5B \
  --output logs/cluster/wan_base_panda_omron_loading.json
```

两份加载报告都通过后，再执行一次真实 `CloseFridge` batch 的完整训练 step。该 M2 配置只运行一步、batch size 1、0 worker、RGB-only、gradient checkpointing，并关闭大 checkpoint 保存。

M2 smoke 还显式设置 `enable_tensorboard=false`：只保留控制台/`tee` 日志，不要求环境安装可选的 `tensorboard` 或 `tensorboardX`。正式训练配置仍默认启用 TensorBoard。

第一次 A800 运行已经完成 forward/backward，但 AdamW 在首次创建约 37.5 GiB 两组 FP32 moment state 时 OOM。新版 smoke 因此使用 ZeRO-2 CPU optimizer offload、1e8 communication bucket 并关闭 overlap；只改变优化器/通信内存位置，不改变数据、12D action、16D proprio、模型输入或 loss。运行前确认主机可用内存，建议 `available` 至少 64 GiB；不足时不要启动：

```bash
free -h

mkdir -p logs/cluster
set -o pipefail

python scripts/train_sft.py \
  model_config=configs/model/wan22_5b_robocasa365_atomic_m2.yaml \
  wan_checkpoint_dir=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/Wan-AI/Wan2.2-TI2V-5B \
  pretrained_checkpoint=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/x-wam/xwam_checkpoints \
  dataset.dataset_path=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/robocasa/robocasa/datasets/v1.0/pretrain/atomic/CloseFridge/20250819 \
  exp_root=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/experiments/xwam-robocasa365 \
  exp_name=close_fridge_m2_single_step \
  2>&1 | tee logs/cluster/close_fridge_m2_single_step.log

echo "train_exit_code=${PIPESTATUS[0]}"
```

日志中必须出现 `offload_optimizer: True`、`offload_optimizer_device: cpu` 和 `allgather_bucket_size: 100000000`。反馈三份 JSON/配置、完整终端日志、返回码、`free -h`、GPU 峰值和首个 loss。若第一条加载 audit 失败，不要继续训练，也不要改为 `strict=False` 或手动删除报错参数。

## X-WAM Conda 环境

先按 `docs/ENVIRONMENT_PLAN.md` 的 E0 步骤审计当前 `abot_m05`，日志写入 `logs/cluster/`。`ok=false` 只表示不能直接运行；当 `clone_base_ok=true` 且 `reuse_recommendation=clone_then_patch` 时，可以 clone 为独立环境后补依赖。不要在 `abot_m05` 中直接运行全量依赖安装。

当前 A800 日志已经满足 clone 条件。执行以下完整命令；它不会取消现有代理：

```bash
conda env list
conda create -n xwam-robocasa365 --clone abot_m05
conda activate xwam-robocasa365

export TORCH_EXTENSIONS_DIR=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/.cache/torch_extensions/xwam-robocasa365

cd /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/xwam-robocasa365
python -m pip uninstall -y wam
bash scripts/install_starlight_dependencies.sh dry-run
bash scripts/install_starlight_dependencies.sh apply

python scripts/audit_starlight_environment.py \
  --require-gpu \
  --include-deepspeed-report \
  --log-file logs/cluster/xwam_robocasa365_environment.json
```

如果同名环境已经存在，跳过 `conda create`，直接激活并从 `dry-run` 开始。不要为了重试自行删除环境。反馈以下文件：

1. `logs/cluster/xwam_dependency_dry-run_*.log`。
2. `logs/cluster/xwam_dependency_apply_*.log`。
3. `logs/cluster/xwam_dependency_apply_*_after.txt`。
4. `logs/cluster/xwam_robocasa365_environment.json`。

安装脚本会拒绝在 `abot_m05` 中执行，也会拒绝保留从 ABot 继承且依赖冲突的 editable `wam`；核心约束文件会阻止 pip 替换已验证的 Torch、torchvision、torchaudio 和 FlashAttention。卸载 clone 中的 `wam` 不会删除 ABot 源码或修改母环境。

当前容器中 `pip check` 会把旧 Decord 0.6.0 wheel tag 报为平台不支持。只有当该提示是唯一输出且 Decord runtime import 成功时，安装器和环境 audit 才将其记录为 warning；其他冲突仍返回失败。真实视频解码验证使用 `CloseFridge` episode 0 左相机 MP4。

## 外部模型路径

复用已有完整 Wan2.2 模型：

```text
/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/Wan-AI/Wan2.2-TI2V-5B
```

使用 X-WAM pretrained checkpoint：

```text
/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/x-wam/xwam_checkpoints/pretrained/checkpoints/last.ckpt
```

在不加载 Torch 的情况下检查文件：

```bash
WAN_DIR=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/Wan-AI/Wan2.2-TI2V-5B
XWAM_CKPT=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/x-wam/xwam_checkpoints/pretrained/checkpoints/last.ckpt

test -s "$WAN_DIR/config.json"
test -s "$WAN_DIR/Wan2.2_VAE.pth"
test -s "$WAN_DIR/models_t5_umt5-xxl-enc-bf16.pth"
test -s "$XWAM_CKPT/checkpoint/mp_rank_00_model_states.pt"
```

不要把 Wan2.2 复制进 Git 仓库；运行配置会直接指向现有目录。

## 验证级别

### `local-static`：本地静态验证

- 不执行 import 的 Python 语法编译。
- Skill 结构验证。
- 文档与变更记录检查。
- 不依赖 Torch 的单元测试和 schema fixture。

### `cluster-smoke`：超算烟测

- 导入 Torch/CUDA/FlashAttention/DeepSpeed。
- 加载模型组件和 checkpoint。
- 加载一个真实数据 batch。
- 完成一步 forward/backward。
- 创建一个模拟器环境并完成一次 rollout。

### `cluster-train`：超算正式训练

- 多 GPU 初始化。
- 测量显存和吞吐。
- 保存并恢复 checkpoint。
- H100 训练和计划内 atomic 评测。

运行时烟测命令会与对应实现一起加入仓库，避免尚未实现的占位命令被误认为可用。

## 硬件分工

- A100/A800：配置环境、测量显存、单 batch 训练、短程过拟合、推理和 rollout 调试。
- H100：只有 A100/A800 烟测通过后才用于正式训练。
- 必须记录准确显存容量；A100 40GB 和 80GB 使用不同配置。

## 反馈要求

复制 `.agents/skills/xwam-robocasa365-workflow/references/cluster-feedback-template.md`，填写所有适用字段并附上相关日志。不能只反馈最后一行异常。
