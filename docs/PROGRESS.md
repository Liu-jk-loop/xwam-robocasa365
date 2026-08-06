# 项目进度

更新时间：2026-08-06

## 当前状态

- 当前分支：`dev/atomic-robocasa365`
- 当前阶段：M1——RoboCasa365 数据契约与原生 loader
- 本地运行能力：没有可用 Torch，只执行静态验证
- 超算运行状态：M1 metadata audit 已通过；`abot_m05` clone 底座已确认
- 任务范围：只包含 atomic，排除 composite

## 阶段状态

| 阶段 | 当前状态 | 超算状态 | 下一门禁 |
| --- | --- | --- | --- |
| M0 工程与协作基线 | 已完成 | 主仓库已 clone | 模拟器阶段开始时确认第三方子模块 |
| M1 原生 RoboCasa365 loader | metadata 门禁通过，原生 loader 待实现 | 数据 audit 通过；环境需 clone 后补包 | 持久化环境日志并创建独立 policy 环境 |
| M2 动作与 checkpoint 适配 | 未开始 | 待验证 | 冻结官方 PandaOmron schema |
| M3 RGB-only 训练烟测 | 未开始 | 待验证 | A100/A800 单 batch forward/backward |
| M4 闭环评测器 | 未开始 | 待验证 | 完成一个 atomic 闭环 rollout |
| M5 离线深度试点 | 未开始 | 待验证 | 完成 1～3 个任务的对齐缓存 |
| M6 Atomic 正式训练与评测 | 未开始 | 待验证 | 通过 H100 正式训练门禁 |
| M7 复现与维护 | 未开始 | 待验证 | clean clone 完整复现 |

## 已确认资源

- 已有 Wan2.2-TI2V-5B 路径可以复用，不再重复下载。
- X-WAM 公开 cross-embodiment pretrained checkpoint 已下载或正在约定的 X-WAM 模型目录下完成下载。
- A100/A800 用于调试，H100 用于正式训练。

## 已记录的集群证据

- 日期：2026-08-06。
- 执行环境：`abot_m05`，A800，系统 CUDA 12.8。
- 数据：`CloseFridge/20250819`。
- 结果：106 episodes、26888 frames、16D state、12D action、三路相机各 106 个 MP4、106 个 Parquet；warnings/errors 均为空，`ok=true`。
- 数据解析路径：`/XYAIFS00/HDD_POOL/sysu_xdliang/sysu_xdliang_5/nieyunshuang/robocasa/robocasa/datasets/v1.0/pretrain/atomic/CloseFridge/20250819/lerobot`。
- 证据状态：数据契约门禁通过；因本次反馈缺少 `git rev-parse HEAD`，commit 归属待补。

环境审计证据：

- 测试 commit：`16ff9132ac5265d7f2448b4628db8720a47d7c10`。
- 原环境：`abot_m05`，Python 3.10.20。
- GPU：NVIDIA A800 80GB PCIe，compute capability 8.0，driver 550.54.15。
- CUDA：Torch 2.9.0+cu128，nvcc 12.8；Torch CUDA probe 通过。
- 可复用：torchvision 0.24.0、torchaudio 2.9.0、FlashAttention 2.8.3、Diffusers 0.38.0；关键 import 通过。
- 需修补：NumPy 1.26.4、Transformers 4.55.2 超出 X-WAM 契约；缺少 Lightning、DeepSpeed、SciPy、Decord、OmegaConf 等依赖。
- 结论：`abot_m05` 适合作为 clone 底座，不可直接用于 X-WAM；禁止修改原环境。

## 待提供输入

- 早先数据 audit 所在的 `git rev-parse HEAD`（如仍可确认）。
- 使用新版脚本生成 `logs/cluster/abot_m05_xwam_environment.json` 持久化报告。
- 按 `docs/CLUSTER_RUNBOOK.md` 中的路径检查确认 X-WAM pretrained 文件。

## 当前执行过程

1. Codex 发布持久化日志修复与 `clone_then_patch` 判定逻辑。
2. 用户拉取最新开发分支，在 `abot_m05` 中生成 `logs/cluster/` 环境报告。
3. 用户 clone `abot_m05` 为独立 `xwam-robocasa365` Conda 环境，不修改原环境。
4. Codex 提供 clone 环境的精确增量依赖安装与验证命令。
5. Codex 基于已验证环境和真实数据 schema 实现原生 Parquet tensor adapter。
