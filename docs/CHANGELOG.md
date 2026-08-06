# 变更记录

## 2026-08-06 — M2 smoke 解除可选 TensorBoard 依赖

- 分支：`dev/atomic-robocasa365`
- 失败 commit：`54920d8404ed83d04d7b1abe0d80c43047c05ec1`
- 运行状态：配置解析通过；在 logger 构造阶段失败，真实 Dataset/model/forward/backward 均未开始

### 问题与诊断

- `TensorBoardLogger` 构造时发现环境没有 `tensorboard` 或 `tensorboardX`，抛出 `ModuleNotFoundError`。
- 该包只用于可视化日志，不属于 Torch、CUDA、DeepSpeed、FlashAttention、Dataset 或 checkpoint 核心链路；此前环境和模型加载门禁不受影响。
- 单步 smoke 已使用 `tee` 持久化控制台日志，没有必要为了这一门禁修改已验证环境或新增 TensorBoard 依赖。

### 新增和修改逻辑

- 训练入口新增 `enable_tensorboard` 配置开关；只有显式启用时才构造 `TensorBoardLogger`，`ConsoleLogger` 始终保留。
- upstream legacy 配置显式保持 `enable_tensorboard=true`，维持原训练行为。
- M2 atomic 单步配置设为 `enable_tensorboard=false`，原训练命令和输出路径不变。
- 增加静态配置测试，防止 M2 smoke 再次意外依赖可选 TensorBoard 包。

### 验证、风险和回滚

- 本地无 Torch；35 项无 Torch 测试、Python compile、文档门禁和 diff 检查通过后发布，真实 Lightning logger 分支为 `cluster-pending`。
- 关闭 TensorBoard 只减少事件文件，不影响控制台 loss、初始化 JSON、训练计算、梯度或 checkpoint 策略。
- 回退本次 commit 或在运行时覆盖 `enable_tensorboard=true` 可恢复 TensorBoard；环境安装 `tensorboard` 不是本轮必要操作。

## 2026-08-06 — M2 `wan_base` 证据与单 GPU smoke 约束

- 分支：`dev/atomic-robocasa365`
- 测试 commit：`4951844a1085c6929426d9ae5561e7586550729e`
- 运行状态：两种初始化均已通过；`xwam_pretrained` 单 batch forward/backward 为 `cluster-pending`

### 星光证据

- `wan_base` 报告为 `pass`、`ok=true`；环境为 Python 3.10.20、Torch 2.9.0+cu128、NVIDIA A800 80GB PCIe。
- 报告明确 `checkpoint=null`：只加载 Wan2.2 backbone，view/action/proprio 模块由代码初始化；目标共 1282 tensors。
- `remapped`、`missing`、`unexpected` 均为空，说明消融路径没有误加载 X-WAM checkpoint 或遗留参数差异。

### 新增和修改逻辑

- M2 单步 smoke 配置固定 `devices=1`，不再依赖 Lightning 在当前容器中自动选择 GPU 数量。
- 训练入口在构造 Dataset/5B 模型前校验可见 GPU、请求设备数和 `WORLD_SIZE` 的整除关系；配置无效时提前失败。
- Trainer 显式接收解析后的设备数，日志同时打印 trainer device、visible device、world size 和 node 数。
- 集群命令使用 `/usr/bin/time -v` 与 `tee` 持久化终端输出和 CPU 最大常驻内存；训练入口在成功或异常退出时打印 CUDA allocated/reserved 峰值。

### 验证、风险和回滚

- 本地无 Torch，拓扑解析的真实 Lightning/DeepSpeed 行为为 `cluster-pending`；Python compile、34 项无 Torch 测试、文档门禁和 diff 检查必须通过后发布。
- 本轮 smoke 只允许单 GPU、batch size 1、一步训练并关闭 checkpoint 保存，不代表 H100 正式训练配置。
- 回退本次 commit 即恢复自动设备选择；不会修改外部数据、权重或 Conda 环境。

## 2026-08-06 — M2 `xwam_pretrained` 真实 checkpoint 加载证据

- 分支：`dev/atomic-robocasa365`
- 测试 commit：`4951844a1085c6929426d9ae5561e7586550729e`
- 运行状态：`xwam_pretrained` 模型构造与参数装载通过；`wan_base` 和单 batch forward/backward 为 `cluster-pending`

### 星光证据与结论

- 环境为 Python 3.10.20、Torch 2.9.0+cu128、NVIDIA A800 80GB PCIe，CUDA 可用；报告 `result=pass`、`ok=true`。
- 公开 checkpoint source 共 1555 tensors，RGB-only PandaOmron target 共 1282 tensors。
- 1275 项参数严格同名同 shape 加载；3 项 action boundary 按 legacy `[0:7]` → PandaOmron `[5:12]` 部分映射；4 项 proprio 语义边界重新初始化。
- 273 项 source-only 参数全部属于 depth `extra_blocks/extra_heads`，在 RGB-only 目标中明确记录为 `discard_source`。
- target 侧 `1275 + 3 + 4 = 1282`，source 侧 `1275 + 3 + 4 + 273 = 1555`；missing、unexpected、shape error 和 errors 全为空，没有静默漏载。

### 边界与下一门禁

- 本轮没有修改源代码、配置或外部权重，只把真实集群证据写入项目记录。
- 此 audit 不读取 RoboCasa365 batch、不把模型移到 GPU 执行 forward/backward、不计算 loss，也不评估左臂 warm-start 的训练优劣。
- 下一步先验证 `wan_base` 初始化来源可解释，再运行 `CloseFridge` 单 batch forward/backward 并记录 CPU/GPU 峰值和首个 loss。

## 2026-08-06 — M2 第二批：训练归一化、14D→12D checkpoint loader 与双初始化

- 分支：`dev/atomic-robocasa365`
- 基线 commit：`95808cd`
- 运行状态：真实 M2 contract 已通过；5B 模型加载和单 batch forward/backward 为 `cluster-pending`

### 星光输入证据

- commit `95808cd8daa49b87907cb4f8c4eafc1c0862cae3` 的 M2 contract 报告为 `pass`、`ok=true`，所有 14 项检查均为 true。
- 真实数据为 `CloseFridge/20250819`；state 16D、action 12D、三路 RGB，真实 modality 与版本化 schema 完全一致。
- state/action 无裁剪 round-trip 最大误差均为 `1.1920928955078125e-07`；控制模式符号域和完整 12D 环境动作检查通过。
- 公开 checkpoint 约 38.9 GB、1555 个 tensor；实际 hidden dim 为 3072，action 输入/输出为 14D，proprio 输入/输出为 16D。

### 新增和修改逻辑

- 将版本化 `PandaOmronTensorCodec` 接入原生 Dataset：按真实 `stats.json` 对 named component 归一化和裁剪，再输出 16D state、12D action 及同 shape mask。
- 将 RoboCasa365 数据配置从 M1 audit-only 切换到 M2 单任务可训练状态，并强制指定版本化 schema；`normalization=none` 仍只保留给审计和诊断。
- 增加无 Torch checkpoint adaptation contract，逐项分类 exact load、action remap、proprio reinitialize、RGB-only discard、missing、unexpected 和 shape error。
- action boundary 只复制 legacy arm `[0:7]` 到 PandaOmron `[5:12]`；新 base/control `[0:5]` 保留 runner 初始化。任何其他 shape mismatch、missing 或 unexpected 参数都会阻塞加载。
- proprio 虽然同为 16D，但语义不同，因此 encoder 输入边界和 decoder 输出边界的 weight/bias 全部重新初始化；中间层仍严格同名同 shape 加载。
- 增加 `xwam_pretrained` 与 `wan_base` 两种显式初始化模式和 JSON 报告；`wan_base` 不允许同时提供 X-WAM checkpoint。
- 训练入口支持显式 `model_config`/`data_config`、0 worker DataLoader，并始终生成初始化报告；原 upstream 配置保留 `legacy_strict` 行为。
- 增加 atomic-only M2 单步 smoke 配置和星光 checkpoint loading audit。
- RGB-only runner 默认关闭 depth 分支；旧 evaluator 删除 `pad_action[:7]`，改为严格检查并执行完整环境动作向量。

### 涉及文件与验证

- 数据：`data/robocasa365_dataset.py`、`configs/data/robocasa365.yaml`。
- checkpoint：`project_tools/xwam_checkpoint_contract.py`、`utils/xwam_checkpoint_loader.py`。
- 训练：`scripts/train_sft.py`、`runners/xwam_runner.py`、`configs/model/wan22_5b_robocasa365_atomic_m2.yaml`。
- 验收与测试：`scripts/audit_xwam_checkpoint_loading.py`、`tests/test_xwam_checkpoint_contract.py`。
- 动作执行：`evaluation/robocasa_client.py`。
- 本地 30 项测试、Python compile、audit CLI help、文档同步门禁和 Git diff check 通过。

### 兼容性、风险和回滚

- 原 `wan22_5b_sft.yaml` 默认仍为 legacy dataset、14D action、16D proprio 和 `legacy_strict`，不改变 upstream 训练语义。
- 本轮 normalization 使用当前单任务目录的 task-local `stats.json`；不能直接视为多任务正式训练统计。
- 公开 checkpoint 全量 runtime load 约 38.9 GB，星光首次测试需要记录 CPU 峰值；本地无法验证 Torch 2.9 运行时。
- evaluator 目前只完成“不得丢弃动作维度”的边界修复；在线 16D observation 构造和 simulator 闭环仍属于 M4，当前不可运行正式评测。
- 回退本次 commit 即恢复 M1 audit-only Dataset 和 upstream strict loader；不会改动外部数据、Wan/X-WAM 权重或 Conda 环境。

## 2026-08-06 — M2 第一批：PandaOmron schema、normalization codec 与 checkpoint inventory

- 分支：`dev/atomic-robocasa365`
- 基线 commit：`e4249b9`
- 运行状态：M1 真实 batch 已通过；M2 本地静态验证完成，真实 contract 为 `cluster-pending`

### M1 集群证据

- commit `e4249b9a763f5f47c7b9b5de0c4ffdefdc27f9e6`，真实 `CloseFridge/20250819` batch 报告为 `pass`、`ok=true`。
- 106 episodes、23496 clips；RGB `[3,9,3,256,320]`、state `[9,16]`、action `[32,12]`。
- 帧序、动作窗口、三相机 mask、无 depth、有限数值和无 augmentation 确定性全部通过。
- M1 阶段正式关闭。

### 问题

M1 只输出未归一化 16D/12D tensor。训练和闭环运行还需要确认每一维语义、统计量和环境动作打包。公开 X-WAM checkpoint 的 action 为 legacy 双臂 14D；PandaOmron 虽然 proprio 同为 16D，但语义完全不同，不能因 shape 相同而静默加载边界层。

### 新增和修改逻辑

- 增加 atomic-only PandaOmron v1 schema，固定 state/action component 名称、连续切片、原始 key、表示和 normalization policy。
- state：base position、base quaternion、相对 EEF position/quaternion、双 gripper qpos；action：4D base motion、control mode、EEF delta position/axis-angle、gripper close。
- 增加真实 `modality.json` 严格匹配；字段名称、顺序、切片或 original key 变化都会阻塞。
- 增加 `stats.json` q01/q99/min/max 检查和 named-component NumPy codec。
- quantile 组件按 q01/q99 归一化；quaternion、controller-range 和离散控制字段保留原生 `[-1,1]`。
- 支持无裁剪可逆审计、训练输入裁剪率统计，以及输出到环境前将 control mode 离散为 `-1/+1`；完整 12D action 始终保留。
- 增加低内存 checkpoint inventory：使用 FakeTensorMode+mmap 读取参数 shape，禁止为了审计普通全量加载 5B checkpoint。
- 固定下一批迁移策略：legacy 左臂 7D 边界权重复制到目标 action `[5:12]`；新 base/control `[0:5]` 初始化；proprio 边界因语义变化重新初始化；其他参数严格按名称和 shape 加载。

### 涉及文件与验证

- schema：`configs/schemas/robocasa365_panda_omron_v1.json`。
- schema/codec：`data/robocasa365_schema.py`。
- 星光审计：`scripts/audit_robocasa365_m2_contract.py`。
- 测试：`tests/test_robocasa365_schema.py`，覆盖真实切片契约、12D round-trip、control-mode 离散、clip 和错误切片拒绝。
- 本地 26 项测试、Python compile 和 diff 检查通过；真实 modality/stats/checkpoint 为 `cluster-pending`。

### 风险与回滚

- 版本化 schema 依据公开 RoboCasa365 数据契约建立，但星光真实文件仍是最终门禁；不一致时不会自动修正。
- 当前 codec 尚未接入训练 Dataset，`training_ready=false` 继续生效。
- checkpoint 本批只做 shape inventory，尚未加载或改写任何参数。
- 回退本次 commit 即移除 M2 schema/audit，不影响已通过的 M1 loader、外部数据、权重或环境。

## 2026-08-06 — M1 原生 LeRobot v2.1 Parquet/MP4 batch adapter

- 分支：`dev/atomic-robocasa365`
- 基线 commit：`f856401`
- 运行状态：本地静态验证完成；真实 RoboCasa365 batch 为 `cluster-pending`

### 问题

此前只有不读取张量的 metadata audit，以及旧 JSON+video loader 的 RGB-only 修复。官方 RoboCasa365 数据是 LeRobot v2.1 的 episode Parquet 加三路 MP4，不能直接进入旧 loader；同时 16D PandaOmron state 和 12D action 的语义及 normalization 尚未在 M2 冻结，误启动训练会产生不可解释输入。

### 新增和修改逻辑

- 增加无 Torch 的 v2.1 episode index：读取 `info.json`、`tasks.jsonl`、`episodes.jsonl`，解析官方路径模板、任务语言、chunk 和 clip 起点。
- 固定时间窗契约：默认九个观测帧、`frame_skip=4`、`action_skip=1`，对应 32 个逐帧 action；边界 clip 不越过 episode。
- 增加原生 runtime adapter：PyArrow 直接读取 `observation.state`、`action`、`frame_index` 和 `episode_index`；Decord 同步读取配置指定的三路 MP4。
- 对 episode 长度、连续帧号、episode ID、有限数值、16D state、12D action、媒体长度和文件存在性进行失败即停校验。
- 输出既有 X-WAM batch key：RGB `[V,T,C,H,W]`、state/action、有效 mask、固定 camera type mask、fps、语言和 episode key；depth 关闭时不检查也不输出任何 depth。
- 增加小型 Parquet/Video LRU cache，并在 DataLoader 序列化时清空运行时 reader cache。
- 训练入口改为 dataset factory，legacy 行为保持；模型和数据 action/proprio 维度不一致时提前报错。
- 原生配置保持 `training_ready=false` 和 `normalization=none`，本轮只允许专用 batch audit，M2 前禁止正式训练。
- 新增持久化 batch audit，报告 commit、依赖版本、解析配置、文件路径、shape/dtype/range、帧序、动作序列、确定性和完整 traceback。
- 新增 PyArrow 16.1.0 依赖并纳入星光约束和环境 manifest；不会替换已验证 Torch/CUDA/FlashAttention/DeepSpeed 栈。

### 涉及文件与验证

- 数据逻辑：`data/robocasa365_index.py`、`data/robocasa365_dataset.py`、`data/dataset_factory.py`、`scripts/train_sft.py`。
- 配置与依赖：`configs/data/robocasa365.yaml`、`requirements.txt`、星光 constraints/manifest。
- 验收入口：`scripts/audit_robocasa365_batch.py`。
- 测试：`tests/test_robocasa365_index.py`，覆盖 prompt/path/chunk、clip 边界、帧/动作 ID 和版本拒绝。
- 本地 22 项 dependency-free 测试和 Python compile 通过；本地没有 Torch/PyArrow runtime，真实 batch 标记为 `cluster-pending`。

### 风险与回滚

- 当前只支持 LeRobot v2.x episode-per-file 布局；v3 分片数据会明确拒绝，不会套用错误路径。
- 当前 state/action 为原始值，不能用于训练；M2 必须依据真实 `modality.json` 建立命名 schema、normalization 和 checkpoint 映射。
- 首次真实 batch 可能暴露 Parquet 列型或视频长度差异，audit 会持久化实际列名与完整异常供修正。
- 回退本次 commit 即恢复旧 dataset 入口；不会修改外部数据、模型或现有 Conda 环境。

## 2026-08-06 — Policy 环境集群验收与 pip check 精确策略

- 分支：`dev/atomic-robocasa365`
- 集群测试 commit：`2da8e02`
- 运行状态：A800 核心环境门禁通过；新版 audit 复跑为 `cluster-pending`

### 问题

从 `abot_m05` clone 后继承了 ABot editable `wam 0.1.0`，其 NumPy/Transformers 精确依赖与 X-WAM upstream 冲突。移除 `wam` 后，`pip check` 仍将 Decord 0.6.0 的旧 manylinux wheel tag 报为当前平台不支持，导致 audit `ok=false`；但 Decord import 和真实 MP4 解码均已通过。安装脚本还会在 `pip check` 非零时提前退出，遗漏安装后 freeze。

### 新增和修改逻辑

- 安装器检测 clone 中残留的 `wam` 并拒绝继续，要求只在独立 X-WAM 环境中显式卸载。
- 安装后先保存 freeze，再执行 `pip check`，因此验收失败也保留安装后环境快照。
- 环境 audit 增加精确 pip-check 策略：仅当 manifest 显式允许且包 runtime import 成功时，才把对应 `is not supported on this platform` 行降级为 warning。
- 当前白名单只有 `decord`；ABot 依赖冲突或任何其他 pip-check 输出仍然阻塞。
- 安装器同样只接受唯一的 Decord 0.6.0 wheel-tag 提示，并额外执行 Decord import。

### 集群证据

- Torch 2.9.0+cu128 在 A800 上完成 BF16 矩阵乘；FlashAttention 2.8.3 完成 BF16 CUDA kernel。
- Decord 0.6.0 解码真实 `CloseFridge` episode 0 左相机 MP4 成功：294 帧、`256x256x3`、`uint8`、像素范围 `[0,255]`。
- Lightning 2.6.5、DeepSpeed 0.19.4 和全部关键模块 import 成功；`ds_report` 返回码为 0。
- `wam` 已从 clone 环境移除，母环境未修改。

### 涉及文件与验证

- 逻辑与契约：`project_tools/starlight_environment.py`、`configs/environment/xwam_starlight.json`、`scripts/install_starlight_dependencies.sh`。
- 测试：`tests/test_starlight_environment.py`，覆盖精确允许与不可隐藏的混合冲突。
- 文档：`docs/ENVIRONMENT_PLAN.md`、`docs/CLUSTER_RUNBOOK.md`、`docs/PROGRESS.md`。
- 本地无 Torch；18 项无依赖测试、shell 语法、Python compile、文档门禁和 Git diff 检查通过。

### 风险与回滚

- Decord 例外只覆盖单一精确 pip 输出，不会忽略缺包、版本冲突或其他平台问题。
- 当前仅验证 CPU 视频解码；M1 真实 batch 仍需检查多视角帧序和动作窗口。
- 回退本次 commit 会恢复严格的原始 `pip check` 判定，不会修改已创建的 Conda 环境。

## 2026-08-06 — 星光 clone 环境增量安装方案

- 分支：`dev/atomic-robocasa365`
- 环境证据 commit：`d5cef4d`
- 运行状态：本地静态验证完成；依赖解析、安装和 GPU audit 为 `cluster-pending`

### 问题

真实环境报告已证明 `abot_m05` 可作为 clone 底座，但原环境的 NumPy/Transformers 越界，并缺少 Lightning、DeepSpeed 等 X-WAM 依赖。直接执行未约束的 `pip install -r requirements.txt` 可能替换已经验证的 Torch/CUDA/FlashAttention 栈，也缺少安装日志和前后环境快照。

### 新增和修改逻辑

- 增加 `abot_m05` 审计版本约束，保护 Torch 2.9.0、torchvision 0.24.0、torchaudio 2.9.0、FlashAttention 2.8.3 及已验证 Python 依赖。
- 将 NumPy 固定为 upstream 测试的 1.23.5，将 Transformers 固定为允许上界 4.51.3。
- Lightning/DeepSpeed 使用稳定主版本区间，首次集群烟测后再依据安装后 freeze 锁定完整精确环境。
- 增加依赖安装脚本，支持 `dry-run` 和 `apply`；强制目标环境名为 `xwam-robocasa365`，明确拒绝修改 `abot_m05`。
- dry-run、apply、pip check 和安装前后 freeze 都写入项目 `logs/cluster/`；不修改代理，并以 `DS_BUILD_OPS=0` 禁止安装阶段预编译 DeepSpeed op。
- 根据真实报告补全中文 clone、安装、验收和反馈命令。

### 涉及文件

- 约束与契约：`configs/environment/xwam_starlight_constraints.txt`、`configs/environment/xwam_starlight.json`。
- 安装入口：`scripts/install_starlight_dependencies.sh`。
- 测试：`tests/test_starlight_environment.py`。
- 文档：`docs/ENVIRONMENT_PLAN.md`、`docs/CLUSTER_RUNBOOK.md`、`docs/PROGRESS.md`。

### 兼容性、风险和验证

- 不修改源环境、外部数据、模型权重或代理变量。
- DeepSpeed 默认安装不预编译全部 CUDA op；真实训练需要的 op 仍可能在首次使用时 JIT 编译。
- 本地验证安装器能够拒绝 `abot_m05`，核心约束存在，shell 语法、无 Torch 单元测试、文档门禁和 Git diff 检查通过。
- pip resolver、DeepSpeed import、`ds_report`、A800 CUDA import 和最终 `ok=true` 均为 `cluster-pending`。

### 回滚

回退本次 commit。若集群 apply 已执行，只删除 clone 出的独立环境才会回滚外部环境；不要修改或删除 `abot_m05`。

## 2026-08-06 — 环境日志持久化与 `abot_m05` 复用结论

- 分支：`dev/atomic-robocasa365`
- 基线 commit：`16ff913`
- 运行状态：A800 环境审计已完成；clone 环境尚未创建

### 问题

环境 audit 将 JSON 写入容器根目录的 `/tmp`。该目录不在项目路径下，换 Pod 后可能消失，用户也容易在当前工作目录中找不到它。同时原报告只有 `ok=false`，无法区分“核心底座不可复用”和“可以 clone 后补包”。

### 新增和修改逻辑

- 默认将报告写入项目下 `logs/cluster/starlight_environment_latest.json`，并支持 `--log-file` 别名。
- 输出改为先原子落盘、再打印；意外 Python 异常也生成包含 traceback 的持久化报告。
- `logs/cluster/` 加入 `.gitignore`，防止完整集群日志进入 Git。
- 增加 `clone_base_ok`、`hard_blockers` 和三态 `reuse_recommendation`。
- 记录 commit `16ff913` 的真实环境证据，并将 `abot_m05` 判定为 `clone_then_patch`。

### 环境结论

- 可复用核心：Python 3.10.20、Torch 2.9.0+cu128、A800 80GB、nvcc 12.8、FlashAttention 2.8.3。
- 原环境没有 DeepSpeed，不存在需要保留的 DeepSpeed 编译产物。
- clone 后需要修正 NumPy/Transformers 并补齐 Lightning、DeepSpeed、SciPy、Decord、OmegaConf 等依赖。
- 原 `abot_m05` 不做任何安装或降级。

### 涉及文件

- 审计逻辑：`project_tools/starlight_environment.py`、`scripts/audit_starlight_environment.py`。
- 测试：`tests/test_starlight_environment.py`。
- 日志隔离：`.gitignore`。
- 文档：`docs/ENVIRONMENT_PLAN.md`、`docs/CLUSTER_RUNBOOK.md`、`docs/PROGRESS.md`。

### 验证

- 环境审计相关测试：7 项通过，包含失败报告持久化测试。
- 既有数据契约测试：6 项通过。
- Python compile 和 Git diff check：通过。
- 新版脚本星光日志路径：`cluster-pending`。

### 回滚

回退本次 commit；不会删除已有日志、修改 Conda 环境或影响外部数据/权重。

## 2026-08-06 — M1 集群证据与星光环境复用审计

- 分支：`dev/atomic-robocasa365`
- 基线 commit：`219916c`
- 运行状态：真实 RoboCasa365 metadata audit 通过；X-WAM runtime 环境待审计

### 问题

星光已在 `abot_m05` 和 A800 上成功审计一份真实 `CloseFridge` 数据，但反馈缺少 Git commit 和完整 Python/Torch/DeepSpeed/FlashAttention 环境信息。直接修改 `abot_m05` 或全量重装可能破坏已有 DeepSpeed/FlashAttention 二进制扩展。

### 新增和修改逻辑

- 记录 `CloseFridge/20250819` 的真实数据契约证据：106 episodes、26888 frames、16D state、12D action、106 个 Parquet、三路相机各 106 个 MP4，`ok=true`。
- 增加版本化的星光 X-WAM policy 环境 manifest，固定 Python/依赖边界、simulator 独立环境契约和 CUDA 12.8 后备版本。
- 增加无侵入环境 audit：检查包版本、隔离 runtime import、Torch CUDA/GPU、`nvidia-smi`、`nvcc`、`pip check`、DeepSpeed `ds_report`、Git commit 和子模块 gitlink。
- 增加 `abot_m05` 先审计、后 clone、只按报告补依赖的中文方案。
- 明确禁止预编译全部 DeepSpeed ops；当前使用 `torch.optim.AdamW` 且未启用 CPU/NVMe offload。
- 修正 M1 数据 audit 示例，使其指向包含日期层的单任务目录。

### 涉及文件

- 环境契约：`configs/environment/xwam_starlight.json`。
- 环境审计：`project_tools/starlight_environment.py`、`scripts/audit_starlight_environment.py`。
- 测试：`tests/test_starlight_environment.py`。
- 文档：`docs/ENVIRONMENT_PLAN.md`、`docs/CLUSTER_RUNBOOK.md`、`docs/IMPLEMENTATION_PLAN.md`、`docs/PROGRESS.md`。

### 兼容性和风险

- audit 不安装包、不修改 Conda 环境，也不主动编译 DeepSpeed op。
- runtime import 在隔离子进程中执行，坏 ABI 不会直接终止主审计进程。
- simulator 依赖继续与 X-WAM policy 环境分离。
- 本地没有 Torch/GPU，真实 runtime 验证保持 `cluster-pending`。
- 数据 audit 的 commit 尚未提供，因此证据先标记为“通过、commit 待补”。

### 验证

- 环境契约/版本比较单元测试：4 项通过。
- 既有 RoboCasa365 数据契约测试：6 项通过。
- 环境 audit 本地负路径：正确返回非零并生成可解析 JSON。
- CLI help、Python compile 和 Git diff check：通过。
- `abot_m05` 环境 audit：`cluster-pending`。

### 回滚

回退本次 commit；不会修改服务器 Conda 环境、外部数据、权重或运行任务。

## 2026-08-05 — 各阶段执行过程中文化

- 分支：`dev/atomic-robocasa365`
- 基线 commit：`1006d72`
- 运行状态：纯文档和 workflow 修改；本地验证通过

### 问题

原计划虽然已经列出中文工作内容和验收条件，但没有逐阶段说明本地修改、超算执行、反馈证据和阶段退出的完整过程。进度表、集群手册和反馈模板仍混有大量英文，不便于持续协作。

### 新增和修改逻辑

- 为 M0～M7 每个阶段增加中文“阶段执行过程”，明确用户输入、Codex 修改、星光运行、反馈证据和进入下一阶段的条件。
- 将项目进度表、当前执行过程和资源/输入状态改为中文。
- 将星光超算操作手册改为中文，并补充 M1 audit 的反馈清单。
- 将超算反馈模板改为中文，保留 `pass`、`fail`、`blocked` 等可检索状态值及中文解释。
- 将项目 workflow skill 的操作说明改为中文，并要求后续变更和进度使用中文记录。

### 涉及文件

- `docs/IMPLEMENTATION_PLAN.md`
- `docs/PROGRESS.md`
- `docs/CLUSTER_RUNBOOK.md`
- `.agents/skills/xwam-robocasa365-workflow/SKILL.md`
- `.agents/skills/xwam-robocasa365-workflow/references/cluster-feedback-template.md`

### 兼容性和风险

- 不修改模型、数据、训练或评测运行逻辑。
- 命令、配置键、代码符号和验证状态标签保持原样，避免翻译导致脚本不可执行或状态不可检索。

### 验证

- Markdown/diff 格式检查：通过。
- Skill 结构验证：通过。
- 文档同步检查：通过。

### 回滚

回退本次文档 commit；不会影响外部数据、模型或超算任务。

## 2026-08-05 — M1 batch 1: 数据契约与 RGB-only 基础

- 分支：`dev/atomic-robocasa365`
- 基线 commit：`bc0e136`
- 运行状态：本地静态验证通过；真实数据和 Torch 集群验证待执行

### 问题

当前训练入口只支持旧 X-WAM JSON+视频格式；缺少 RoboCasa365 官方 LeRobot 元数据门禁。旧 `RobotDataset` 即使模型设置 `use_depth=false`，仍会要求 depth 目录和每个视角的 `depth_path`，augmentation 也强制读取 `depths`。

### 新增和修改逻辑

- 增加官方 Target Atomic-Seen 18 的版本化任务清单。
- 增加无 Torch 的 RoboCasa365 metadata audit，校验 16D state、12D action、三相机、episodes 元数据和 atomic-only 任务归属。
- 同时支持传入直接 LeRobot 路径或包含 `lerobot/` 的任务目录。
- 识别 LeRobot v2 `episodes.jsonl` 和 v3 分块 episodes 元数据，并在尚未支持逐 episode 审计时给出明确提示。
- 修改旧 `RobotDataset`：`use_depth=false` 时不要求 depth 目录、不要求 `depth_path`、不读取 depth 视频，也不向 batch 写入 `depths`。
- 修改视频 augmentation，使深度张量可选但 RGB crop/color jitter 保持可用。
- 将训练和验证数据集的 `use_depth` 与模型配置打通。
- 将完整 M0-M7 计划改为中文版，并在集群手册解释 submodule 操作。

### 兼容性和风险

- `RobotDataset` 默认仍为 `use_depth=true`，保持旧训练行为。
- 本批只建立原生数据契约和审计入口，尚未完成 Parquet 到 X-WAM tensor 的正式 loader。
- 真实官方数据布局和媒体数量必须在星光上验证。
- 本地无 Torch，augmentation 和旧 loader 的运行时验证标记为 `cluster-pending`。

### 涉及文件

- 数据契约与审计：`data/robocasa365_contract.py`、`scripts/audit_robocasa365_dataset.py`。
- 任务范围：`configs/tasks/robocasa365_atomic_seen.json`。
- RGB-only 修复：`data/robot_dataset.py`、`data/augmentation.py`、`scripts/train_sft.py`。
- 测试：`tests/test_robocasa365_contract.py`。
- 计划与进度：`docs/IMPLEMENTATION_PLAN.md`、`docs/ARCHITECTURE.md`、`docs/CLUSTER_RUNBOOK.md`、`docs/PROGRESS.md`。

### 验证

- 纯 Python 数据契约单元测试：6 项通过。
- Audit CLI 合成数据验证：通过，输出可解析 JSON。
- Skill / 文档检查 / Python compile / Git diff check：通过。
- M1 batch-1 commit `83210c2` 已发布到 `origin/dev/atomic-robocasa365`。
- 真实数据 metadata audit：`cluster-pending`。
- RGB-only 旧 loader：`cluster-pending`。

### 回滚

回退本 M1 commit；默认 depth 行为未改变，不涉及外部数据写入。

## 2026-08-05 — M0 repository workflow baseline

- Branch: `dev/atomic-robocasa365`
- Base commit: `72cfb86`
- Runtime status: local static validation passed; cluster validation pending

### Problem

The fork had no repository-specific collaboration rules, implementation roadmap, progress record, cluster feedback contract, or documentation enforcement. The existing global `*.sh` ignore rule would also prevent future Slurm and launch scripts from being versioned.

### Added logic

- Defined atomic-only scope, local/cluster ownership, validation states, and adapter boundaries.
- Added a milestone plan from native RoboCasa365 loading through atomic-only training and evaluation.
- Added a repository workflow skill that guides future modifications and cluster feedback handling.
- Added a deterministic Git-diff check requiring change and progress documentation for implementation/workflow changes.
- Included untracked files in the documentation check so newly created implementation files cannot bypass it.
- Added Starlight model locations, checkout procedure, validation levels, and a structured debug report template.
- Removed the global shell-script ignore rule so future cluster scripts can be committed intentionally.

### Compatibility and risk

- No model, dataset, training, or evaluation runtime behavior is changed in M0.
- The local machine cannot validate Torch/CUDA behavior.
- The repository's third-party submodules are not initialized locally.

### Validation

- Skill structure: passed with the skill-creator validator.
- Documentation enforcement script: passed against the complete branch working diff.
- Python syntax compilation: passed for repository and skill Python sources without importing Torch.
- Git whitespace/error check: passed.
- Development branch publication: passed; `dev/atomic-robocasa365` tracks the fork remote.
- Cluster checkout: pending.

### Rollback

Revert the M0 commit. No external data, checkpoint, or simulator state is modified.
