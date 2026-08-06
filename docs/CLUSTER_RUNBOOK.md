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
