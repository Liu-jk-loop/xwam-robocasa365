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

- RoboCasa `1.0.1`；RoboCasa365 的 65 atomic / 300 composite 和 Gym 注册接口从 `1.0` 系列引入，`0.2.x` 只能视为旧版环境。版本化任务 horizon 至少包含官方 `29f7ce8` 更新。
- robosuite `>=1.5.2`。
- MuJoCo `3.3.1`、NumPy `2.2.5`；RoboCasa `1.0.1` import 会检查这两个精确版本。
- Gymnasium、ImageIO、ImageIO-FFmpeg 和 PyZMQ，用于 Gym wrapper、视频和 simulator/policy broker。
- RoboTwin [commit `c3ddfa8`](https://github.com/RoboTwin-Platform/RoboTwin/commit/c3ddfa8b97d5519efa828b075999bd0006778e5e)；当前 atomic-only RoboCasa 工作不使用它。

不要为了训练把 simulator 依赖强行安装进 policy 环境。两个环境将在 M4 通过 ZeroMQ broker 对接。

### M4.0：复用已有 RoboCasa 环境

先执行 `conda env list`，然后逐个激活已有 RoboCasa 候选环境。每个环境先运行不创建 MuJoCo 的注册审计，再运行 runtime smoke；两个命令都只读环境，不会安装、升级或修改包：

```bash
conda env list
conda activate <候选环境名>

cd /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/xwam-robocasa365

python scripts/audit_robocasa365_simulator_environment.py \
  --output logs/cluster/robocasa365_simulator_<候选环境名>_registry.json

python scripts/audit_robocasa365_simulator_environment.py \
  --runtime-smoke \
  --output logs/cluster/robocasa365_simulator_<候选环境名>_runtime.json
```

判定规则：

- `reuse_ready`：Atomic-Seen 18 全部注册，`CloseFridge(target)` 可 reset/step，state 合计 16D、动作合计 12D，三路 `256x256 uint8` RGB 和 EGL render 均通过；可作为 M4 simulator 环境。
- `runtime_smoke_required`：版本和任务注册满足，但还没有创建真实 simulator；必须继续跑第二条命令。
- `legacy_robocasa_only`：检测到 RoboCasa `0.x`，可保留给旧项目，但不能运行 RoboCasa365 benchmark。
- `registry_blocked` / `runtime_blocked` / `dependency_blocked`：把完整 JSON 返回给 Codex，在确认根因前不修改该环境。

报告会记录当前 Conda 名称、Python executable、包版本、editable module 路径、源码 Git commit/status 和完整子进程输出。第一次筛选不运行 X-WAM、不需要 Torch/GPU，也不修改代理。

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

因此采用 `clone_then_patch`。环境日志对应 commit `d5cef4d`，没有 hard blocker；DeepSpeed 报错仅由包尚未安装引起。完成补包后，目标环境需要满足：

- Python 为 3.10。
- Torch 至少为 2.4，Torch runtime 能识别 A800。
- NumPy 位于 `[1.23.5, 1.26.0)`。
- Transformers 位于 `[4.49.0, 4.51.3]`。
- Lightning 位于 `[2.5, 2.7)`，DeepSpeed 位于 `[0.18, 0.20)`，`deepspeed` import 和 `ds_report` 成功。
- FlashAttention、Lightning、Diffusers、Decord 等关键 import 成功。
- `pip check` 不存在依赖冲突。

## 五、E2：clone 与增量安装

环境创建、依赖解析和安装均在登录到 A800 任务节点后执行。现有代理保持不变，不执行 `unset http_proxy`、`unset https_proxy` 或类似命令。

先创建独立环境；如果 `conda env list` 已存在同名环境，不要重复 clone，也不要删除，先激活后运行 audit 判断其状态：

```bash
conda env list
conda create -n xwam-robocasa365 --clone abot_m05
conda activate xwam-robocasa365

export TORCH_EXTENSIONS_DIR=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/.cache/torch_extensions/xwam-robocasa365

cd /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/xwam-robocasa365
python -m pip uninstall -y wam
bash scripts/install_starlight_dependencies.sh dry-run
bash scripts/install_starlight_dependencies.sh apply
```

clone 会保留环境中已安装的 Python 包和二进制扩展，也会继承 ABot 的 editable `wam`。该包固定要求 NumPy 1.26.4 和 Transformers 4.55.2，与 X-WAM upstream 约束冲突，因此只在 clone 环境中卸载；不会删除 ABot 源码或修改 `abot_m05`。安装脚本会在检测到 `wam` 时拒绝继续。

安装脚本只允许在 `xwam-robocasa365` 中运行，会拒绝修改 `abot_m05`；它使用 `configs/environment/xwam_starlight_constraints.txt` 保护已经验证的 Torch 2.9.0、torchvision 0.24.0、torchaudio 2.9.0、FlashAttention 2.8.3 以及 CUDA 12.8 组合。pip 子进程显式使用 `DS_BUILD_OPS=0`，避免安装阶段预编译 DeepSpeed CUDA op；脚本不会修改代理变量。

`dry-run` 只进行 pip 依赖解析，结果写入 `logs/cluster/xwam_dependency_dry-run_<UTC时间>.log`。`apply` 执行相同解析并安装，随后运行 `pip check`，同时写入安装前后 `pip freeze`。首次集群烟测通过后，以 `*_after.txt` 为依据生成完整锁文件；在此之前，Lightning/DeepSpeed 使用经过约束的稳定版本区间，而不是假装已有集群验证的精确版本。

安装完成后执行验收：

```bash
python scripts/audit_starlight_environment.py \
  --require-gpu \
  --include-deepspeed-report \
  --log-file logs/cluster/xwam_robocasa365_environment.json

python -m pip check
```

预期 `ok=true`、`reuse_recommendation=clone_ready`。PyPI 的 Decord 0.6.0 wheel 在当前 x86_64/glibc 2.39 容器中可 import，并已成功解码真实 `CloseFridge` MP4；新版 pip 仍会因旧 manylinux tag 输出 `is not supported on this platform`。环境 manifest 只允许将这一条精确提示降级为 warning，且必须先通过 Decord runtime import；任何其他 `pip check` 输出仍为错误。

`ds_report` 中当前未使用的 CPUAdam、AIO、GDS、FP quantizer 和 sparse attention 等 op 显示未安装或不兼容不构成当前烟测失败；当前训练使用 `torch.optim.AdamW` 且没有 CPU/NVMe offload。以 DeepSpeed report 返回码和后续真实一步训练为准。`TORCH_EXTENSIONS_DIR` 使用独立目录，避免其他环境留下的 Torch/CUDA JIT 缓存造成 ABI 冲突。

## 六、DeepSpeed 处理原则

- 不设置 `DS_BUILD_OPS=1`，不预编译全部 op。
- `abot_m05` 当前没有安装 DeepSpeed；在 clone 中安装 Python 包时默认不会预编译全部 CUDA op。
- 当前 X-WAM 使用 `torch.optim.AdamW`，且 `DeepSpeedStrategy` 尚未配置 CPU/NVMe offload；因此 CPUAdam、AIO 等未安装不构成当前烟测失败。
- 先用 `ds_report` 查看 installed/compatible 状态，再由真实一步训练确定实际需要的 op。
- 若某个必需 op 首次 JIT 过慢，只编译该 op，或为相同 Python/Torch/CUDA/GPU 架构构建一次 wheel 后复用。
- 禁止使用 `DS_SKIP_CUDA_CHECK=1` 掩盖 CUDA 不匹配。
- A800 为 Ampere（SM80），H100 为 Hopper（SM90）；若后续必须自行编译 CUDA 扩展，构建产物必须覆盖实际训练卡架构。

## 七、全新环境后备方案

仅当 `abot_m05` 的 Python/Torch ABI 或核心依赖无法兼容时启用：

- Python 3.10。
- Torch 2.8.0、torchvision 0.23.0、torchaudio 2.8.0。
- PyTorch CUDA 12.8 wheel 源：`https://download.pytorch.org/whl/cu128`。
- FlashAttention 2.8.3。
- 其余依赖遵守 `configs/environment/xwam_starlight.json`。

该方案当前不启用；现有报告已证明 clone 底座可用。
