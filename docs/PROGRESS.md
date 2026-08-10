# 项目进度

更新时间：2026-08-10

## 当前状态

- 当前分支：`dev/atomic-robocasa365`
- 当前阶段：Clariden 迁移——独立 X-WAM aarch64/GH200 policy 容器部署
- 本地运行能力：没有可用 Torch，只执行静态验证
- 超算运行状态：M1～M4.3门禁均已在原 A800 环境通过；Clariden 镜像/SQSH/EDF、4×GH200 CUDA、FlashAttention BF16 kernel、checkpoint 发现和真实 CloseFridge batch 已通过。首次 1×GH200 单步训练 Job `3044897` 在 DeepSpeed NVTX wrapper 中失败；精确 NVTX runtime overlay 修复待计算节点验证
- 任务范围：只包含 atomic，排除 composite

## 阶段状态

| 阶段 | 当前状态 | 超算状态 | 下一门禁 |
| --- | --- | --- | --- |
| M0 工程与协作基线 | 已完成 | 主仓库已 clone | 模拟器阶段开始时确认第三方子模块 |
| M1 原生 RoboCasa365 loader | 已完成 | commit `e4249b9` 真实 batch `ok=true` | 已关闭 |
| M2 动作与 checkpoint 适配 | 已完成 | 两种初始化、完整动作契约及 DeepSpeedCPUAdam 单 batch 参数更新均通过 | 已关闭 |
| M3 RGB-only 训练烟测 | 已完成 | commit `5420c89` 训练、commit `50b11a4` audit：16 项全真，result `pass/global_step=12` | 已关闭 |
| M4 闭环评测器 | 已完成 | commit `f9e1b6b`：故意中断/确定性恢复后完成 CloseFridge 900步，机器审计 `pass` | 已关闭 |
| M5 离线深度试点 | 未开始 | 待验证 | 完成 1～3 个任务的对齐缓存 |
| M6 Atomic 正式训练与评测 | 18任务RGB训练实现完成 | cluster-pending | 生成全局统计，完成4×H100 step 2→4保存恢复门禁 |
| M7 复现与维护 | 未开始 | 待验证 | clean clone 完整复现 |

## Clariden 部署状态

- 登录节点确认为 `clariden-ln002`，架构为 `aarch64`；Store、IOPS scratch 和 Capstor scratch 可用。
- 复用既有 RoboCasa365 SQSH/EDF，不修改 simulator：`robocasa365-ngc2410-b4684e6e.sqsh` 已存在。
- 复用既有 65 个 atomic human pretrain 任务；训练入口继续只指向 `pretrain/atomic`，不读取同级 composite。
- 复用 FastWAM 已下载的 Wan2.2 三个 safetensors 分片、T5 和 VAE；tokenizer 复用其 Wan2.1 UMT5 目录。
- FastWAM 的 Wan2.2 目录缺少 X-WAM Diffusers loader 必需的三个小文件，部署脚本从官方固定 revision `921dbaf` 补 `config.json`、`configuration.json` 和权重 index。
- X-WAM 公开 cross-embodiment checkpoint 在 Clariden Store 尚不存在；部署脚本只下载 `pretrained` 初始化权重，不下载 RoboCasa/Robotwin SFT 权重。
- 容器基线冻结为 NGC 24.10（Ubuntu 22.04、Python 3.10、CUDA 12.6.2）+ 官方 aarch64 Torch 2.9.0 cu126；FlashAttention 2.8.3 和 Decord 0.6.0 使用官方固定 commit 源码构建。
- `deployment/clariden/` 已提供 Containerfile、约束、Store/bootstrap、allocation 内 build→validate→enroot import、EDF 模板和 4×GH200 kernel/discovery 门禁。
- Decord/FlashAttention 的固定 commit 与递归子模块更新顺序已完成静态门禁；build allocation 时限为8小时，实际编译耗时仍待 Clariden 证据。
- 首次 Clariden build 的 Torch 2.9.0/cu126 安装已通过；requirements 因 Diffusers 0.38.0 要求 safetensors >=0.8.0-rc.0，与旧 pin 0.5.3 冲突而退出。现已改为 safetensors 0.8.0、huggingface-hub 0.36.0，并移除与 Torch 2.9 冲突的未使用 NGC Transformer Engine/Torch-TensorRT；等待重试。
- 第二次 build 证明上述依赖修复有效，Decord 和 FlashAttention 2.8.3 aarch64 wheel 均成功构建。当前失败仅在 STEP 17 无 GPU image validation：T5 在 import 时调用 CUDA device。已改为构造时延迟选择 device，等待同节点缓存复用重试。
- commit `6594d89` 的 image validation 修复后成功生成有效 16.98 GiB SQSH；EDF/manifest 因 enroot 收尾非零手工补齐。后续 4×GH200 validation 全部通过，容器、CUDA、FlashAttention kernel 和模型发现门禁关闭；validation Job ID 未提供。
- Clariden 真实 batch smoke 已由用户回报通过，但未提供 Job ID；三路 RGB、16D state、12D action 与确定性解码门禁关闭。
- Clariden 首次单步训练 Job `3044897` 已完成 model/data/DeepSpeed 初始化并进入 `training_step`，随后在 DeepSpeed 0.19.4 的 NVTX domain wrapper 调用 `nvtx.get_domain` 时失败。峰值显存 allocated/reserved 为 30.677/40.076 GiB，因此该次失败不是 OOM，也尚未进入真实 X-WAM forward/backward。
- 当前 SQSH 中实际 import 的 `nvtx` 不提供 domain API。新增哈希锁定的 `nvtx==0.2.12` IOPS runtime overlay，训练作业会在加载模型前验证版本、模块来源和 `get_domain`；同时把同一版本加入下一次镜像构建的固定依赖。
- 本地静态状态：`local-static`；runtime overlay 安装与 1×GH200 单步训练复跑均为 `cluster-pending`。下一门禁先运行 `prepare_runtime_overlay_xwam.sbatch`，再运行 `smoke_train_xwam.sbatch`。

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

M4.0 simulator 环境验证证据：

- 测试实现 commit：`8c42521e8fc38570bc0e4d8fa500494c73e7558f`；环境为 `robocasa-abot`、Python 3.11.15。
- 版本为 RoboCasa 1.0.1、robosuite 1.5.2、MuJoCo 3.3.1、NumPy 2.2.5、Gymnasium 0.29.1、ImageIO 2.37.4、ImageIO-FFmpeg 0.6.0 和 PyZMQ 27.1.0；所有依赖约束通过。
- Atomic-Seen 18 全部注册；`CloseFridge(target, seed=0)` 在 `MUJOCO_GL=egl` 下成功 reset、执行一条动作并关闭，runtime 返回码为 0。
- 在线 observation 为五个具名分量、合计 16D；Gym 动作为五个具名分量、合计 12D；三路 RGB 与 render 均为 `[256,256,3] uint8`，runtime `errors=[]`。
- 最终报告为 `ok=true`、`reuse_recommendation=reuse_ready`、`blockers=[]`、`warnings=[]`；M4.0 运行能力门禁关闭。
- `robosuite_models`、GR1 mink 和 Gym observation-space warning 不阻塞 PandaOmron runtime；正式 benchmark 前仍需解释 robosuite editable 工作区的大量 tracked 修改，当前不执行 reset 或覆盖。

M4.1 随机闭环集群证据：

- run id `20260807T152317Z`，测试 commit `a9165477f0407a1fb1f064ac597a043dce79e5b9`；分支正确、服务器工作区干净，配置/runtime horizon 都是 900。
- 固定 `CloseFridge(target, seed=0, layout=1, style=1)` 完成 20/20 step；episode expected/observed/completed 为 `1/1/1`，failed/missing 为 `0/0`，summary `result=pass`。
- 在线 state/action dimension 为 16/12；随机动作在20步中底盘均非零，control mode 与 gripper 都覆盖 `-1/+1`。随机任务没有成功，符合工程门禁预期，不作为策略指标。
- 三相机视频为有效 H.264、`768x256`、5 FPS、6帧、102479 bytes；抽帧确认左/右 agentview 与 eye-in-hand 顺序、内容和运动连续性正常。
- RoboCasa 1.0.1、robosuite 1.5.2、MuJoCo 3.3.1、Gymnasium 0.29.1、PyZMQ 27.1.0 和 EGL provenance 完整；M4.1 正式关闭。

M4.2 X-WAM 单请求闭环集群证据：

- run id `20260808T015103Z`，测试 commit `b5b6f53da075919cfb7cab4588623dd7d1e5ba76`；server/client 均记录正确分支、相同 commit 和干净工作区。
- A800 policy server 从 M3 global step 10 checkpoint 定向严格恢复；missing frozen 为438、missing non-frozen 为0、unexpected为0，使用真实 CloseFridge stats 和版本化 schema。
- Broker 完成一次请求转发；server 返回 `[32,12]`，inference/round-trip 为30.566/31.593秒，processed/failed requests为1/0，CUDA峰值 allocated/reserved 为31.985/32.070 GiB。
- Client在固定 `CloseFridge(target, seed=0, layout=1, style=1)` 中执行前4步，episode expected/observed/completed为`1/1/1`，failed/missing为`0/0`，四步底盘均非零，结果为`pass`。
- 视频为H.264、`768x256`、5 FPS、5帧、66961 bytes；三相机顺序与运动连续性正常。4步内`success=false`不影响工程门禁，也不代表正式策略指标。
- Policy与simulator分别运行于`xwam-robocasa365`和`robocasa-abot`；M4.2正式关闭，下一步为900-step长horizon运行可靠性和证据设计。

M4.3 900-step 可恢复闭环集群证据：

- run id `20260808T023440Z`，测试 commit `f9e1b6bebde87eeb99cff2151a0eef3bc2501a31`；分支为 `dev/atomic-robocasa365`，工作区干净。
- 使用 M3 `epoch=9-step=10.ckpt` 的 DeepSpeed model state；固定 `CloseFridge(target, seed=0, layout=1, style=1)` 完成官方 900/900 step，共 225 次 policy request。
- 首轮故意中断后使用同一 run 恢复；机器审计的 `intentional_resume_verified=true`，progress、request 数、server journal、单一 checkpoint、请求 shape、16D state/12D action 和动作边界检查均通过。
- RGB-only 视频证据为 46 帧；帧名称、持久化帧完整性和非空视频检查均通过。审计 19 项 checks 全真、`errors=[]`、`ok=true/result=pass`，M4.3 工程门禁关闭。
- episode 在 900 step 时 `success=false`。这是仅训练 10 step 的工程 checkpoint，结果证明长时推理、断点恢复、完整动作和落盘链路可靠，不代表 RoboCasa365 benchmark 策略性能达标。
- 本机只归档了 M4.3 `audit.log`；同级 `summary.json`、`metadata.json`、`episode.json` 仍是 M4.2 run `20260808T015103Z` 的旧副本。完整 M4.3 原始证据继续保存在超算忽略目录，不加入 Git。

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

M3.1 120 GiB 保存与首次 resume 证据：

- 基于 commit `54d1f01` 的低内存 profile 已在固定 clip 上正常达到 `global_step=8`，`checkpoint_save_start`、`checkpoint_save_complete` 和 run result `pass` 均存在；保存时进程 RSS 约 97.4 GB，CUDA peak allocated/reserved 为 `30.677/40.076 GiB`。
- step-8 checkpoint 路径可被 DeepSpeed 解析，首次 resume 进入模型状态恢复后失败；没有开始 step 8/9，也没有进入 optimizer/scheduler state 恢复，不是 CUDA/主机 OOM。
- 缺失 key 全部属于已冻结的 `text_encoder.model.*` 和 `vae.model.*`，与保存侧 `exclude_frozen_parameters=true` 一致；日志没有显示 `model.*` 可训练参数缺失。
- 当前补丁仅在“resume + exclude frozen”组合下按当前 `requires_grad` 集合放行冻结参数缺失，并继续拒绝可训练参数、buffer、unexpected key 和 shape mismatch；同时修正 generator state 在 DeepSpeed 实际 hook 顺序下的应用。
- 原 step-8 checkpoint 保留并复用；补丁后的 resume-to-10 为 `cluster-pending`。

M3.1 resume 修复验收证据：

- commit `fabaaba` 从原 `epoch=7-step=8.ckpt` 成功恢复；日志显示 `missing_frozen=438`、`missing_non_frozen=0`、`unexpected=0`，随后明确恢复 generator state 和全部训练状态。
- step 8 的 video/action/proprio/total loss 为 `0.233982/0.801885/0.897742/1.933608`；step 9 命中 clean-action 分支，因此 action/proprio loss 按设计为 0，不是数据或恢复缺失。
- Trainer 正常达到 `max_steps=10`，step-10 checkpoint 的 start/complete 事件齐全，result 为 `pass`；CUDA peak allocated/reserved 仍为 `30.677/40.076 GiB`，保存期 RSS 约 97.3 GB。
- M3.1 checkpoint/resume 工程门禁关闭。BF16 CPUAdam state 结论仍只适用于 120 GiB 调试，不进入数值或正式训练结论。

M3.2 本地准备：

- 根据用户确认，M3.1 已有 10 个 step 中 5 个 action/proprio 监督 step，加上 M2 单步共 6 个有效监督更新，足以作为短程 loss/参数更新 smoke；取消尚未运行的 50-step、`clean_action_ratio=0` 人工过拟合诊断，不宣称已完成严格过拟合。
- 新增三个 Atomic-Seen 任务 manifest：默认选择 `PickPlaceCounterToCabinet`、`OpenCabinet`、`TurnOnMicrowave`，分别覆盖取放、关节物体和电器操作；自动解析唯一日期目录，多日期并存时强制显式选择。
- 新增平衡索引 adapter，虚拟索引按 `0→1→2` 布局，使完整 Dataset 的三个任务样本总量相等。三个子 Dataset 分别执行既有 PandaOmron schema 与 task-local normalization，外层不修改 12D action、16D proprio 或 RGB tensor 合同；Lightning/DeepSpeed sampler 可在训练时随机重排索引。
- 新增 12-step FP32/no-checkpoint 配置，保持原 X-WAM `clean_action_ratio=0.5`；训练日志增加监督比例、task index 和 manifest provenance。
- 无 Torch 机器审计要求 step 0～11 完整、task index 合法、三个任务均覆盖且计数最大差不超过 2；同时检查两类监督分支、有限 loss、depth loss 恒为 0 和 Trainer/result 正常结束，不要求固定取样顺序。

M3.2 三任务数据审计证据：

- 测试 commit：`3d49c970a00317b3adb466ae8c139d5912946e96`；manifest schema 1、`scope=atomic_only`、`sampling=balanced_round_robin`、`ok=true`、`result=pass`，errors 为空。
- `PickPlaceCounterToCabinet/20250819`：108 episodes、24,225 frames；`OpenCabinet/20250819`：107 episodes、37,492 frames；`TurnOnMicrowave/20250819`：107 episodes、14,010 frames。
- 合计 322 episodes、75,727 frames；三个任务均为 16D state、12D action，且相机顺序一致为 left agentview、right agentview、eye-in-hand。
- 三个任务均属于版本化 Atomic-Seen 清单；真实数据/视频门禁通过，可以进入 12-step RGB-only 训练。该结论不代表训练运行已经通过。

M3.2 12-step 训练反馈与 audit 修正：

- 训练日志完整包含 step 0～11，Trainer 达到 `max_steps=12`，run result 为 `pass`；CUDA peak allocated/reserved 为 `30.677/40.076 GiB`，没有 traceback、NaN、OOM 或 depth 读取。
- `clean_action_ratio=0.5` 保持不变；监督 step 为 0、1、5、7、8，其他 step 的 action/proprio loss 按设计为 0，监督比例和 loss 对应关系通过。
- 实际 task index 为 `0,2,1,1,1,1,1,0,2,2,0,2`，三个任务计数为 `3/5/4`。旧 audit 错把 Dataset 的平衡索引布局当成 Trainer 的固定读取顺序，因此只有 `balanced_round_robin` 一项误报失败。
- 修正后保留真实 Lightning/DeepSpeed shuffle，只检查合法索引、三个任务覆盖和计数最大差不超过 2；同一原始日志在本地 dependency-free 重审为 `ok=true/result=pass`。集群需拉取新 commit 后只重跑 audit，无需重新训练。
- 正式证据已补齐：run id `20260807T105307Z`，训练 commit `5420c8986836f1ca26fbccc67f5c93db3c263119`，分支 `dev/atomic-robocasa365`，工作区干净；修正版 audit commit 为 `50b11a4`。
- metadata 确认单卡 A800 80GB、Python 3.10.20、Torch 2.9.0+cu128、CUDA 12.8、Lightning 2.6.5、DeepSpeed 0.19.4；使用 ZeRO-2、FP32 DeepSpeedCPUAdam state、CPU offload、`xwam_pretrained`、RGB-only 和完整三任务 manifest。
- result 确认 `pass/global_step=12/error=null`，耗时 518.53 秒，CUDA peak allocated/reserved 为 `30.677/40.076 GiB`，且按配置没有生成 checkpoint。
- 进程 VmHWM 约 108.74 GiB；结束时 cgroup current/max 为 `119.9987/120 GiB`，`memory.events.max=4`、`oom=0`、`oom_kill=0`。本次 smoke 通过，但该 FP32 CPU-offload profile 几乎没有主机内存余量，禁止直接扩展为长训练或正式 H100 profile。
- M3 的真实 batch、参数更新、短程 loss、低内存 checkpoint/resume、三任务 RGB-only 训练和机器审计均已通过，阶段关闭。

## 待提供输入

- M3 不再需要补充输入或重跑。
- M4.0 不再需要重跑；`robocasa-abot` 已选为当前 simulator smoke 环境。
- M4.1 不再需要重跑；本地 `log/` 证据只读保留并由根级 ignore 排除，机器日志、视频、模型和评测产物继续位于 Git 之外。
- M4.2 不再需要补充输入或重跑；原始日志与视频保持在 Git ignore 目录，不上传仓库。
- M4.3 不再需要重跑；900-step、225 次请求、故意中断恢复和视频证据已通过机器审计。
- 下一步由用户在新commit上生成M6 18任务manifest/global stats和preflight JSON，再运行4×H100 step 2→4门禁；depth试点暂不混入首轮H100 RGB基线。

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
17. 120 GiB profile 的 fresh experiment 已完成 step 8、checkpoint 保存和正常退出，证明低内存保存门禁通过。
18. 首次 resume 因 DeepSpeed 严格要求 checkpoint 包含已主动排除的冻结 T5/VAE 参数而失败；Codex 已实现定向严格恢复和 generator hook 时序修复，等待从原 step-8 checkpoint 复测到 step 10。
19. commit `fabaaba` 已成功恢复 step 8～9，并在 global step 10 完成 checkpoint 保存和正常退出；M3.1 关闭。
20. 用户决定保留原 `clean_action_ratio=0.5`，并以已有 6 个有效监督更新结束单任务趋势 smoke；未运行的 50-step/clean-ratio-zero 诊断被取消。
21. Codex 已准备 M3.2 三个 atomic 任务的 manifest、平衡轮询 adapter、12-step FP32/no-checkpoint 配置与机器审计，等待 A800 运行。
22. 用户已在 commit `3d49c97` 完成三任务真实数据审计；322 episodes、75,727 frames、16D/12D/三相机合同全部通过，下一步为 12-step 训练。
23. 三任务 12-step 训练正常完成；旧 audit 因 Lightning/DeepSpeed 随机 sampler 打乱固定顺序而误报。Codex 已改为覆盖与短前缀计数容差审计，同一日志本地重审通过，等待服务器生成正式 audit JSON 并补 metadata。
24. 用户补齐修正版 audit、metadata 和 result：训练 commit `5420c89` 工作区干净，16 项 audit 全真，result `pass/global_step=12`；M3 正式关闭并进入 M4 准备。
25. Codex 审计官方 RoboCasa365 `1.0.1` Gym 接口，发现原环境文档仍锁定旧 `0.2.0`；已更正 simulator 契约，并实现 M4.0 候选环境只读审计器。真实 Conda 环境、assets 和 EGL smoke 为 `cluster-pending`。
26. 用户在 `robocasa-abot` 补齐 PyZMQ 27.1.0 后重跑 runtime：Atomic-Seen 18、`CloseFridge(target)`、16D state、12D action、三路 RGB、单步和 EGL 全部通过，报告为 `reuse_ready`；M4.0 关闭，进入 M4.1。
27. Codex 已实现 M4.1 dependency-light 随机闭环门禁：冻结 Atomic-Seen 18 官方 horizon，固定 `CloseFridge` scene/seed，按名称完成在线 16D observation 与完整 12D Gym action 映射，保存逐 episode JSON、三相机视频和可重算 summary。本地不具备 RoboCasa runtime，真实 20-step rollout 标记为 `cluster-pending`。
28. 用户在 commit `a916547` 完成 M4.1：20步、完整12D动作、三相机视频和可重算 summary 全部通过；M4.1 关闭。
29. Codex 已实现 M4.2 版本化 NPZ/JSON policy 协议、透明 broker、严格 M3 checkpoint policy server 和 X-WAM simulator client；补充单任务 checkpoint/request 一致性、至少一次真实请求和 CUDA 峰值证据门禁。本地协议/路由/静态测试通过，A800真实模型加载与一次4-action闭环为 `cluster-pending`。
30. 用户在 commit `b5b6f53` 完成 M4.2：M3 step-10 checkpoint 严格恢复、一次32x12推理、broker往返、四步完整12D环境动作和三相机视频全部通过；M4.2关闭，进入M4.3长horizon准备。
31. Codex 已实现 M4.3 900-step配置、逐请求/逐动作原子progress、相同seed动作回放与16D state漂移阻塞、可恢复PNG帧缓存、server fsync请求JSONL和长运行机器审计；本地静态验证完成，星光故意中断/恢复及完整horizon为`cluster-pending`。
32. 用户在 commit `f9e1b6b` 完成 M4.3：故意中断后确定性恢复，最终完成 CloseFridge 900步和225次模型请求；机器审计19项全真、`ok=true/result=pass`。M4工程链路全部关闭，下一步转入H100 RGB-only正式训练门禁规划。
33. Codex 已实现M6 18任务自然采样manifest、跨任务统计、按GBS 128对齐的5-epoch调度、4×H100首选/回退profile、FP32 optimizer实态审计、多卡保存恢复和正式final checkpoint；本地静态门禁通过，真实集群验证为`cluster-pending`。
