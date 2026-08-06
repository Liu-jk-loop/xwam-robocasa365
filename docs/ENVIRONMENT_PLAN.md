# 星光 X-WAM 环境复用方案

## 一、结论

`abot_m05` 已完成首轮审计，结论为“可以作为 clone 底座，但不能原样运行 X-WAM”。不要直接在原环境中安装或升级包；使用 Conda clone 创建独立的 `xwam-robocasa365` policy 环境后，再按报告处理不兼容项。

已确认可复用的核心是 Python 3.10.20、Torch 2.9.0+cu128、A800 CUDA runtime 和 FlashAttention 2.8.3。原环境没有安装 DeepSpeed，因此不存在可继承的 DeepSpeed 编译产物；不要为了补包修改 `abot_m05`。

官方依据：

- [Conda 官方文档](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html#cloning-an-environment)支持使用 `conda create --name NEW --clone OLD` 精确复制环境。
- [DeepSpeed 官方安装说明](https://www.deepspeed.ai/tutorials/advanced-install/)指出默认按需 JIT 编译实际使用的 op，并推荐先运行 `ds_report`；不需要预编译全部 op。
- [PyTorch 官方历史版本页](https://pytorch.org/get-started/previous-versions/)提供 Torch 2.8.0 的 CUDA 12.8 wheel。
- [FlashAttention 官方仓库](https://github.com/Dao-AILab/flash-attention#installation-and-features)说明 FlashAttention 2 支持 Ampere/A100/A800 与 Hopper/H100，CUDA 要求为 12.0 及以上。

## 二、环境边界

### X-WAM policy 环境

负责：

- Torch、CUDA、X-WAM、Wan2.2、FlashAttention、Lightning 和 DeepSpeed。
- 数据 batch、训练、checkpoint 加载和 policy inference。

环境名：`xwam-robocasa365`。

### RoboCasa simulator 环境

负责：

- RoboCasa、robosuite、MuJoCo、环境创建、状态恢复和渲染。
- 闭环评测时通过 broker 与 policy 环境通信。

仓库锁定版本：

- RoboCasa `0.2.0`，[commit `756598a`](https://github.com/robocasa/robocasa/commit/756598a5be52e052339bb2d957426e39015c2afb)。
- robosuite `1.5.2`，[commit `232ce7d`](https://github.com/ARISE-Initiative/robosuite/commit/232ce7d4a6ed89c949a9aba024a05c8c32fdd08b)。
- MuJoCo `3.2.6`。
- RoboTwin [commit `c3ddfa8`](https://github.com/RoboTwin-Platform/RoboTwin/commit/c3ddfa8b97d5519efa828b075999bd0006778e5e)；当前 atomic-only RoboCasa 工作不使用它。

不要为了训练把 simulator 依赖强行安装进 policy 环境。两个环境将在 M4 通过 ZeroMQ broker 对接。

## 三、E0：审计 `abot_m05`

在 A800 节点中执行：

```bash
cd /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/xwam-robocasa365
git rev-parse HEAD
git status --short --branch

conda activate abot_m05
python scripts/audit_starlight_environment.py \
  --require-gpu \
  --include-deepspeed-report \
  --log-file logs/cluster/abot_m05_xwam_environment.json
```

该命令只读取环境信息、执行 `pip check`，并在隔离子进程中测试关键 import；它不安装包，也不会主动编译 DeepSpeed op。脚本会先原子写入项目下的持久化日志，再向终端打印 JSON。`logs/cluster/` 已加入 `.gitignore`，不会把集群日志提交到 Git。

返回非零表示环境尚不能直接运行 X-WAM，不等于不能作为 clone 底座。查看以下字段：

- `ok`：当前环境是否已经满足全部 X-WAM 依赖。
- `clone_base_ok`：Python/Torch/CUDA 核心是否足以作为 clone 底座。
- `reuse_recommendation`：`clone_ready`、`clone_then_patch` 或需要重建。

需要反馈：

1. `git rev-parse HEAD`。
2. `logs/cluster/abot_m05_xwam_environment.json` 完整内容。
3. 若命令异常退出，附完整 traceback。

## 四、E1：复用判定

首轮报告（commit `16ff913`）已经确认：

- `abot_m05` 使用 Python 3.10.20。
- Torch 2.9.0+cu128、torchvision 0.24.0、torchaudio 2.9.0 均可 import。
- Torch 正确识别 A800 80GB，compute capability 为 8.0。
- 系统 nvcc 与 Torch CUDA 均为 12.8。
- FlashAttention 2.8.3 可 import。
- `pip check` 通过。
- 需要在 clone 中处理 NumPy 1.26.4、Transformers 4.55.2，并补 Lightning、DeepSpeed、SciPy、Decord、OmegaConf 等包。

因此采用 `clone_then_patch`。完成补包后，目标环境需要满足：

- Python 为 3.10。
- Torch 至少为 2.4，Torch runtime 能识别 A800。
- NumPy 位于 `[1.23.5, 1.26.0)`。
- Transformers 位于 `[4.49.0, 4.51.3]`。
- DeepSpeed 至少为 0.16，`deepspeed` import 和 `ds_report` 成功。
- FlashAttention、Lightning、Diffusers、Decord 等关键 import 成功。
- `pip check` 不存在依赖冲突。

源环境的新版持久化报告确认 `clone_base_ok=true` 后执行：

```bash
conda create -n xwam-robocasa365 --clone abot_m05
conda activate xwam-robocasa365

export TORCH_EXTENSIONS_DIR=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/.cache/torch_extensions/xwam-robocasa365

cd /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/xwam-robocasa365
python scripts/audit_starlight_environment.py \
  --require-gpu \
  --include-deepspeed-report \
  --log-file logs/cluster/xwam_robocasa365_environment.json
python -m pip check
```

clone 会保留环境中已安装的 Python 包和二进制扩展。`TORCH_EXTENSIONS_DIR` 使用独立目录，避免其他环境留下的 Torch/CUDA JIT 缓存造成 ABI 冲突。

## 五、DeepSpeed 处理原则

- 不设置 `DS_BUILD_OPS=1`，不预编译全部 op。
- `abot_m05` 当前没有安装 DeepSpeed；在 clone 中安装 Python 包时默认不会预编译全部 CUDA op。
- 当前 X-WAM 使用 `torch.optim.AdamW`，且 `DeepSpeedStrategy` 尚未配置 CPU/NVMe offload；因此 CPUAdam、AIO 等未安装不构成当前烟测失败。
- 先用 `ds_report` 查看 installed/compatible 状态，再由真实一步训练确定实际需要的 op。
- 若某个必需 op 首次 JIT 过慢，只编译该 op，或为相同 Python/Torch/CUDA/GPU 架构构建一次 wheel 后复用。
- 禁止使用 `DS_SKIP_CUDA_CHECK=1` 掩盖 CUDA 不匹配。
- A800 为 Ampere（SM80），H100 为 Hopper（SM90）；若后续必须自行编译 CUDA 扩展，构建产物必须覆盖实际训练卡架构。

## 六、全新环境后备方案

仅当 `abot_m05` 的 Python/Torch ABI 或核心依赖无法兼容时启用：

- Python 3.10。
- Torch 2.8.0、torchvision 0.23.0、torchaudio 2.8.0。
- PyTorch CUDA 12.8 wheel 源：`https://download.pytorch.org/whl/cu128`。
- FlashAttention 2.8.3。
- 其余依赖遵守 `configs/environment/xwam_starlight.json`。

具体安装命令要等 E0 报告后生成，避免无依据地重装 DeepSpeed/FlashAttention。
