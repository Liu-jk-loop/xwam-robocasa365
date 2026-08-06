# 项目进度

更新时间：2026-08-06

## 当前状态

- 当前分支：`dev/atomic-robocasa365`
- 当前阶段：M2——PandaOmron schema、normalization 与 checkpoint 适配
- 本地运行能力：没有可用 Torch，只执行静态验证
- 超算运行状态：M1 全部门禁通过；M2 contract 与 `xwam_pretrained` 12D 真实加载已通过，`wan_base` 与单步反传待执行
- 任务范围：只包含 atomic，排除 composite

## 阶段状态

| 阶段 | 当前状态 | 超算状态 | 下一门禁 |
| --- | --- | --- | --- |
| M0 工程与协作基线 | 已完成 | 主仓库已 clone | 模拟器阶段开始时确认第三方子模块 |
| M1 原生 RoboCasa365 loader | 已完成 | commit `e4249b9` 真实 batch `ok=true` | 已关闭 |
| M2 动作与 checkpoint 适配 | contract 已通过；训练归一化、14D→12D loader、双初始化和完整 12D 执行已实现 | `xwam_pretrained` commit `4951844` 真实加载 `ok=true`；`wan_base`/反传待验证 | `wan_base` 报告 + 单 batch forward/backward |
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

## 待提供输入

- 在 A800 上生成 `wan_base` 初始化报告；该模式不得提供 X-WAM checkpoint。
- `wan_base` 通过后运行真实单 batch forward/backward，反馈 CPU 内存、GPU 峰值、首个 loss、完整日志和初始化 JSON。
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
9. 用户继续执行 `wan_base` 初始化 audit 和单 batch forward/backward；通过后关闭 M2 并进入 M3。
