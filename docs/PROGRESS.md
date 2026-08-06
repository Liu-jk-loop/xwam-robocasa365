# 项目进度

更新时间：2026-08-06

## 当前状态

- 当前分支：`dev/atomic-robocasa365`
- 当前阶段：M3.1——单任务极小样本、checkpoint 与 resume 门禁
- 本地运行能力：没有可用 Torch，只执行静态验证
- 超算运行状态：M1、M2 全部门禁通过；M3.1 FP32 CPUAdam 已完成 8 个训练 step，但在 step 8 checkpoint/退出阶段失去 Pod，120 GiB 低内存修复待验证
- 任务范围：只包含 atomic，排除 composite

## 阶段状态

| 阶段 | 当前状态 | 超算状态 | 下一门禁 |
| --- | --- | --- | --- |
| M0 工程与协作基线 | 已完成 | 主仓库已 clone | 模拟器阶段开始时确认第三方子模块 |
| M1 原生 RoboCasa365 loader | 已完成 | commit `e4249b9` 真实 batch `ok=true` | 已关闭 |
| M2 动作与 checkpoint 适配 | 已完成 | 两种初始化、完整动作契约及 DeepSpeedCPUAdam 单 batch 参数更新均通过 | 已关闭 |
| M3 RGB-only 训练烟测 | M3.1 低内存修复中 | FP32 CPUAdam 完成 step 0～7；checkpoint/result 未通过 | 120 GiB profile 保存 step 8，再恢复到 step 10 |
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
- 本地 22 项无 Torch 测试和 Python 语法编译通过。

M1 原生 loader 星光验收证据：

- 测试 commit：`e4249b9a763f5f47c7b9b5de0c4ffdefdc27f9e6`；结果 `pass`、`ok=true`。
- 环境：Python 3.10.20、Torch 2.9.0+cu128、NumPy 1.23.5、PyArrow 16.1.0、Decord 0.6.0。
- 数据：`CloseFridge/20250819`，106 episodes、23496 valid clips。
- sample：RGB `[3,9,3,256,320]`、state `[9,16]`、action `[32,12]`；DataLoader batch 维度正确。
- 帧序 `[0,4,...,32]`、动作序列 `[0..31]`、camera mask `[0,0,1]`，无 depth。
- 所有 tensor 有限；无 augmentation 时重复读取完全一致。
- 结论：M1 退出条件全部满足，阶段关闭。

M2 第一批本地证据：

- 增加版本化 PandaOmron schema，固定 16D state 和 12D action 的命名、切片、表示与 normalization policy。
- 增加真实 `modality.json` 严格匹配和 `stats.json` q01/q99/min/max 维度审计。
- 增加 NumPy state/action codec，保留完整 12D action，支持 unclipped round-trip、训练区间 clip 和 control-mode 环境离散化。
- 增加 checkpoint 低内存 shape inventory，目标是确认公开 checkpoint 为 legacy action 14D/proprio 16D，并输出后续迁移计划。
- 本地 26 项测试、Python compile 和 diff 检查通过。

M2 contract 星光验收证据：

- 测试 commit：`95808cd8daa49b87907cb4f8c4eafc1c0862cae3`；结果 `pass`、`ok=true`。
- 环境：Python 3.10.20、Torch 2.9.0+cu128、NumPy 1.23.5、PyArrow 16.1.0。
- 真实 `modality.json` 与版本化 schema 完全一致；schema SHA 为 `e95f2b71...f08b4`，modality 文件 SHA 为 `59589093...73fb`。
- episode 0 的 state/action 无裁剪往返最大误差均为 `1.1920928955078125e-07`；训练裁剪比例分别约为 `0.276%` 和 `1.559%`。
- control mode 只出现 `-1`，294 帧全部通过符号域检查；完整 12D environment action 门禁通过。
- 公开 checkpoint 文件约 38.9 GB、1555 个 tensor；实际隐藏维度为 3072，action 边界为 14D，proprio 边界为 16D，所有检查为 true。
- 结论：真实数据语义、统计量和 checkpoint 迁移前提均已确认，M2 第二批可以按冻结合同实现。

M2 第二批本地证据：

- 原生 Dataset 已接入 `panda_omron_v1` codec；训练 sample 在转为 Torch tensor 前按 named component 归一化并裁剪，仍保留 16D state 和完整 12D action。
- `robocasa365.yaml` 已解除 M1 audit-only 门禁，但只适用于单任务 task-local stats；正式多任务统计在 M3/M6 另行冻结。
- checkpoint adapter 严格分类 exact load、action remap、proprio reinitialize、RGB-only discard、missing、unexpected 和 shape error。
- action encoder/decoder 只把 legacy arm `[0:7]` 复制到 PandaOmron `[5:12]`；base/control `[0:5]` 保留新初始化；proprio 输入/输出边界因语义变化全部重初始化。
- 支持 `xwam_pretrained` 和 `wan_base`；后者明确禁止传入 X-WAM checkpoint，避免实验来源混淆。
- RGB-only evaluator 已删除只执行前 7 维的旧逻辑，环境动作维度不等于 policy 输出时立即失败；在线 observation 提取仍属于 M4，当前不能据此启动闭环评测。
- 增加 M2 单步 A800 smoke 配置和 checkpoint load audit；本地 30 项测试、Python compile、CLI help 和 diff 检查通过。
- 本地没有 Torch，真实 5B 模型构造、38.9 GB checkpoint 加载和 forward/backward 均为 `cluster-pending`。

M2 `xwam_pretrained` 星光真实加载证据：

- 测试 commit：`4951844a1085c6929426d9ae5561e7586550729e`；模式 `xwam_pretrained`，结果 `pass`、`ok=true`。
- 环境：Python 3.10.20、Torch 2.9.0+cu128、NVIDIA A800 80GB PCIe；CUDA 可用。
- 真实构造 RGB-only PandaOmron 12D/16D runner，并全量读取公开 X-WAM checkpoint：source 1555 tensors，target 1282 tensors。
- target 1282 tensors 全部可解释：1275 项严格同名同 shape 加载、3 项 action 边界部分映射、4 项 proprio 语义边界重初始化；不存在 target missing。
- source 1555 tensors 全部可解释：1275 项加载、3 项映射、4 项 proprio 边界不采用、273 项 depth `extra_blocks/extra_heads` 因 RGB-only 目标而明确丢弃；不存在 unexpected 或 shape error。
- action remap 精确包含 encoder 输入 weight、decoder 输出 weight/bias；初始化报告同时记录 base/control `[0:5]` 的三个部分切片与四个 proprio 边界参数。
- `missing_source`、`unexpected_source`、`shape_errors`、`errors` 均为空，说明 checkpoint 适配没有静默漏载。
- 本门禁只验证模型构造和参数装载，不读取真实 batch、不执行 CUDA forward/backward、不产生 loss，也不证明左臂 warm-start 策略优于其他初始化方案。

M2 `wan_base` 星光真实加载证据：

- 测试 commit：`4951844a1085c6929426d9ae5561e7586550729e`；模式 `wan_base`，结果 `pass`、`ok=true`。
- 环境：Python 3.10.20、Torch 2.9.0+cu128、NVIDIA A800 80GB PCIe；CUDA 可用。
- `checkpoint=null`，确认没有误加载公开 X-WAM cross-embodiment checkpoint。
- 目标共 1282 tensors：Wan2.2 backbone 由 `from_pretrained` 加载；view embedding、action encoder/decoder、proprio encoder/decoder 由代码初始化。
- `remapped`、`missing`、`unexpected` 均为空，消融初始化来源清晰；该门禁同样不执行 forward/backward。
- M2 smoke 配置进一步固定 `devices=1`，避免容器暴露多卡时单 batch 调试意外占用全部 GPU。

M2 单步训练第一次启动反馈：

- 测试 commit：`54920d8404ed83d04d7b1abe0d80c43047c05ec1`；解析后的配置正确指向 12D/16D、RGB-only、单 GPU、一步训练和真实 `CloseFridge/20250819`。
- 运行在创建 `TensorBoardLogger` 时因环境没有可选包 `tensorboard/tensorboardX` 退出，尚未构造 Dataset、模型或 optimizer，也未执行 forward/backward。
- 已有 Torch/CUDA/DeepSpeed/FlashAttention 和两种模型初始化证据仍然有效；该错误不表示核心训练环境损坏。
- M2 smoke 显式设置 `enable_tensorboard=false`，保留 `ConsoleLogger` 与 `tee` 日志；正式 upstream 配置显式保持 `enable_tensorboard=true`。

M2 单步训练第二次启动反馈：

- 真实 `CloseFridge` batch、X-WAM checkpoint 适配、GPU forward 和 backward 均已完成；数据、RGB-only、12D action 和 16D proprio 链路没有发现新错误。
- OOM 精确发生在首次 `optimizer.step()`：Torch AdamW 为 5B trainable parameters 初始化 `exp_avg_sq` 时尝试再申请 18.77 GiB；当时 allocated 77.645 GiB、reserved 77.748 GiB、空闲约 239 MiB。
- reserved-but-unallocated 仅约 103 MiB，因此不是 CUDA allocator 碎片问题；设置 `expandable_segments` 不能解决约 18.77 GiB 的真实容量缺口。
- 两组 FP32 Adam moment 合计约 37.5 GiB。M2 单卡配置改用 ZeRO-2 CPU optimizer offload，同时把 allgather/reduce bucket 从 5e8 降为 1e8 并关闭通信 overlap，为参数更新留出显存余量。
- upstream legacy 配置仍保持 ZeRO-2、GPU optimizer、5e8 bucket 和 overlap；本次只改变 atomic M2 单卡 smoke。
- 新配置需要主机提供足够内存，运行前以 `free -h` 确认可用内存，建议至少 64 GiB；真实参数更新仍为 `cluster-pending`。

M2 单步训练第三次启动反馈：

- 解析配置正确包含 ZeRO-2、CPU optimizer offload、1e8 bucket 和关闭 overlap，说明 commit `4bdb272` 的显存策略 wiring 已生效；反馈没有附 `git rev-parse HEAD`，正式 SHA 仍待补。
- DeepSpeed 在初始化阶段发现 runner 仍提供 `torch.optim.AdamW`，按默认保护抛出 `ZeRORuntimeException`，要求使用 `DeepSpeedCPUAdam` 或显式绕过保护。
- 本轮失败发生在 `strategy.setup/deepspeed.initialize`，未读取真实 batch、未执行 forward/backward 或 optimizer step；21.294 GiB CUDA peak 不能代表完整 offload 训练峰值。
- runner 改为根据 `deepspeed_offload_optimizer` 选择 backend：M2 smoke 使用 `DeepSpeedCPUAdam`，正式默认配置 offload false 时仍使用 `torch.optim.AdamW`。
- 不采用 `zero_force_ds_cpu_optimizer=false` 绕过方案；CPUAdam 扩展加载/编译和一次完整参数更新仍为 `cluster-pending`。

M2 单步训练第四次启动与关闭证据：

- 运行逻辑包含 commit `2c107b2` 独有的 optimizer backend 日志；反馈未单独附 `git rev-parse HEAD`，该复现字段在 M3 必须补齐。
- checkpoint initialization 为 `xwam_pretrained/pass`；DeepSpeed 配置为 ZeRO-2、CPU optimizer offload、1e8 bucket、关闭 overlap，backend 为 `deepspeed_cpu_adam`。
- 真实 `CloseFridge` batch 产生 video/action/proprio loss `0.191945/1.254097/1.533597`，depth loss 为 0，总 loss `2.979639` 与分项求和一致。
- Trainer 正常报告 `max_steps=1 reached`，无 traceback；因此 forward、backward 和第一次 CPUAdam 参数更新全部通过。
- CUDA peak allocated/reserved 为 `30.677/40.076 GiB`，证明 M2 单卡 offload profile 可在 A800 80GB 上完成训练 step。
- M2 的 schema round-trip、checkpoint 可解释加载、完整 12D 动作、RGB-only 数据与单 batch 训练全部满足退出条件，阶段正式关闭。

M3.1 首次 8-step/checkpoint 反馈：

- 固定 `CloseFridge` clip 0、RGB-only、A800 单卡和 ZeRO-2 FP32 CPUAdam 成功执行 step 0～7；所有 loss 有限，depth loss 始终为 0，Lightning 报告 `max_steps=8 reached`。
- step 7 日志间隔由约 30 秒增至约 180 秒，恰好对应 `save_interval=8`；随后缺少 `CUDA peak memory`、run result JSON 路径和 Python traceback，tmux/Pod 同时失效。
- 结论为训练计算已完成、checkpoint/teardown 门禁失败；结合 5B trainable、6.4B frozen 和 120 GiB 主机内存，最高概率是 checkpoint 期间 cgroup OOM，但在取得 `memory.events`/Pod exit 137 前不标记为已确认 OOM。
- 新增独立 `a800_80gb_120g_debug` profile：只在 M3 工程门禁使用 BF16 CPUAdam state，并排除冻结 T5/VAE checkpoint；原 A800、M2 和 upstream profile 保持 FP32。
- checkpoint 事件将 fsync 写入 JSONL；真实保存和 resume 仍为 `cluster-pending`。正式 H100 必须恢复 FP32 optimizer state 并重新通过保存/恢复门禁。

## 待提供输入

- 拉取 M3.1 120 GiB 修复 commit 后，使用新的实验名和低内存 hardware profile 对固定 `CloseFridge` clip 训练到 step 8；只有 checkpoint、`checkpoint_save_complete` 和 result JSON 均通过才恢复到 step 10。
- 反馈 `git rev-parse HEAD`、两次返回码、`free -h`/`df -h`、两份完整日志、全部 run config/metadata/result JSON、连续 loss、checkpoint 目录大小和最终 CUDA 峰值。
- 不在本轮运行闭环 simulator；新版在线 16D observation 提取将在 M4 实现和验收。

## 当前执行过程

1. 用户已完成独立环境增量安装，并移除 clone 中冲突的 ABot `wam`。
2. 用户已通过 Torch/FlashAttention CUDA kernel 和真实 Decord MP4 解码。
3. Codex 发布只允许 Decord 精确旧 wheel tag warning 的 audit，并修复安装后 freeze 留存。
4. 用户已在 commit `e4249b9` 读取真实 `CloseFridge` batch，M1 全部门禁通过。
5. Codex 已发布 M2 第一批：版本化 schema、normalization codec 与 checkpoint shape audit。
6. 用户已在 commit `95808cd` 执行 M2 contract audit，全部检查为 true。
7. Codex 已根据真实 3072/14D/16D shape 实现 Dataset normalization、shape-aware loader、双初始化和完整 12D 动作执行。
8. 用户已在 A800 上完成 `xwam_pretrained` 真实加载；1275 exact + 3 remap + 4 reinitialize 完整覆盖 1282 个目标 tensor，门禁通过。
9. 用户已完成 `wan_base` 初始化 audit，确认只加载 Wan2.2 base 且机器人模块由代码初始化，门禁通过。
10. Codex 将 M2 smoke 固定为单 GPU；用户执行 `xwam_pretrained` 单 batch forward/backward，通过后关闭 M2 并进入 M3。
11. 第一次启动在可选 TensorBoard logger 构造阶段退出；Codex 改为 smoke 显式关闭 TensorBoard，等待同命令重跑。
12. 第二次启动已完成 forward/backward，在首次 AdamW optimizer state 初始化时因 80GB 显存容量不足退出；Codex 改为 ZeRO-2 CPU optimizer offload，等待单 batch 参数更新复测。
13. 第三次启动在 DeepSpeed 初始化阶段因 offload 仍收到 Torch AdamW 而退出；Codex 将 M2 offload 分支切换为 DeepSpeedCPUAdam，正式配置保持 Torch AdamW，等待复测。
14. 第四次启动已用 DeepSpeedCPUAdam 完成真实 batch 的 forward、backward 和 optimizer update；CUDA allocated peak 30.677 GiB、首个 loss 2.979639，M2 关闭并进入 M3 准备。
15. Codex 已实现 M3.1 分层配置、固定单 clip、8/10 step 调度边界、DeepSpeed resume、自定义 RNG state 恢复和运行 provenance；等待 A800 两段式验收。
16. 第一次 M3.1 运行完成 8 个训练 step，但 Pod 在 checkpoint/退出阶段失效；Codex 已实现 120 GiB 独立低内存 profile、冻结参数排除和 checkpoint cgroup/RSS 诊断，等待 fresh experiment 复测。
