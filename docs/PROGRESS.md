# 项目进度

更新时间：2026-08-06

## 当前状态

- 当前分支：`dev/atomic-robocasa365`
- 当前阶段：M1——RoboCasa365 数据契约与原生 loader
- 本地运行能力：没有可用 Torch，只执行静态验证
- 超算运行状态：M1 metadata 与 policy 环境核心门禁已通过；原生 loader 已实现，真实 batch 待验收
- 任务范围：只包含 atomic，排除 composite

## 阶段状态

| 阶段 | 当前状态 | 超算状态 | 下一门禁 |
| --- | --- | --- | --- |
| M0 工程与协作基线 | 已完成 | 主仓库已 clone | 模拟器阶段开始时确认第三方子模块 |
| M1 原生 RoboCasa365 loader | 原生 v2.1 Parquet/MP4 adapter 与 batch audit 已实现 | 环境 kernel/解码门禁通过；真实 batch 为 `cluster-pending` | 读取一个真实 RGB-only batch |
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

- 测试 commit：`d5cef4df498faf6a4df0a7c99c2335aa1d6c73c1`。
- 原环境：`abot_m05`，Python 3.10.20。
- GPU：NVIDIA A800 80GB PCIe，compute capability 8.0，driver 550.54.15。
- CUDA：Torch 2.9.0+cu128，nvcc 12.8；Torch CUDA probe 通过。
- 可复用：torchvision 0.24.0、torchaudio 2.9.0、FlashAttention 2.8.3、Diffusers 0.38.0；关键 import 通过。
- 需修补：NumPy 1.26.4、Transformers 4.55.2 超出 X-WAM 契约；缺少 Lightning、DeepSpeed、SciPy、Decord、OmegaConf 等依赖。
- 结论：`abot_m05` 适合作为 clone 底座，不可直接用于 X-WAM；禁止修改原环境。
- 安装方案：clone 为 `xwam-robocasa365`，先 dry-run、再 apply；约束文件保护现有 Torch/CUDA/FlashAttention，完整日志与安装前后 freeze 均写入 `logs/cluster/`。

独立 policy 环境验证证据：

- 测试 commit：`2da8e020952086de8a0805a8c16ef2a4d783d228`；环境为 `xwam-robocasa365`，A800 80GB。
- 版本：Torch 2.9.0+cu128、NumPy 1.23.5、Transformers 4.51.3、FlashAttention 2.8.3、Lightning 2.6.5、DeepSpeed 0.19.4、Decord 0.6.0。
- Torch BF16 矩阵乘 CUDA kernel 通过；FlashAttention BF16 CUDA kernel 通过。
- Decord 成功解码真实 `CloseFridge` episode 0 左相机视频：294 帧、`256x256x3`、`uint8`、范围 `[0,255]`。
- DeepSpeed import 与 `ds_report` 返回码通过；当前不用的 async I/O、GDS、FP quantizer、sparse attention 不作为阻塞。
- 已从 clone 环境移除 ABot editable `wam`；原 `abot_m05` 未修改。
- 唯一剩余原始 `pip check` 输出是 Decord 0.6.0 旧 wheel tag；已有 runtime import 和真实解码证据，按精确白名单降级为 warning。
- 结论：policy 环境核心门禁通过；需要拉取新版 audit 并复跑，生成机器可读的最终 `ok=true` 证据。

M1 原生 loader 本地证据：

- 新增 LeRobot v2.1 episode 索引、路径模板、语言任务解析和 clip 时间窗逻辑。
- 新增 PyArrow Parquet 读取与三路 Decord MP4 同步解码，输出 X-WAM 既有 batch key。
- 默认窗口为 9 个观测帧、`frame_skip=4`、32 个 12D action；state 为 9 个 16D 向量。
- 相机顺序固定为左 agentview、右 agentview、eye-in-hand；类型 mask 为 `[0,0,1]`。
- `use_depth=false`，batch 中不产生 `depths`；augmentation 关闭时重复读取必须逐项完全一致。
- M1 state/action 尚未归一化，原生配置以 `training_ready=false` 阻止误启动训练；M2 冻结 schema 后解除。
- 本地 22 项无 Torch 测试和 Python 语法编译通过；真实 Parquet/MP4 batch 为 `cluster-pending`。

## 待提供输入

- 早先数据 audit 所在的 `git rev-parse HEAD`（如仍可确认）。
- 按 `docs/CLUSTER_RUNBOOK.md` 中的路径检查确认 X-WAM pretrained 文件。
- 拉取新版 audit 后复跑，反馈最终 `ok=true` 报告。
- 安装新增的 PyArrow 16.1.0，并运行 M1 真实 RGB-only batch audit。

## 当前执行过程

1. 用户已完成独立环境增量安装，并移除 clone 中冲突的 ABot `wam`。
2. 用户已通过 Torch/FlashAttention CUDA kernel 和真实 Decord MP4 解码。
3. Codex 发布只允许 Decord 精确旧 wheel tag warning 的 audit，并修复安装后 freeze 留存。
4. Codex 已实现原生 Parquet/MP4 tensor adapter、固定时间窗和 batch audit，并保留训练门禁。
5. 用户拉取指定 commit，在独立 policy 环境补装受约束的 PyArrow，读取真实 `CloseFridge` batch。
6. Codex 根据报告核对 shape、dtype、帧序、动作窗口和确定性，通过后关闭 M1 并进入 M2。
