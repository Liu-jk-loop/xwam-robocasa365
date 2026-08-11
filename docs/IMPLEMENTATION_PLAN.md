# X-WAM × RoboCasa365 Atomic-only 完整实施计划

## 一、项目目标

在星光超算上建立一条可复现的完整链路：

```text
RoboCasa365 原生数据
  → 数据审计与 atomic-only 过滤
  → X-WAM 训练张量适配
  → A100/A800 调试
  → H100 正式训练
  → RoboCasa365 Atomic-Seen 闭环评测
```

当前阶段只处理 atomic 任务，明确排除 composite 任务、复合任务分解和长历史建模。

参考基线：RoboCasa v1.0.1 [数据使用说明](https://robocasa.ai/docs/build/html/datasets/using_datasets.html) 与官方 [dataset registry](https://github.com/robocasa/robocasa/blob/main/robocasa/utils/dataset_registry.py)。任务和数据 schema 发生变化时，先更新版本化清单和审计测试，再修改训练逻辑。

## 二、固定实验范围

- 训练候选池：RoboCasa365 Human Pretraining 中的 65 个 atomic 任务，以版本化任务清单选择。
- 首个目标评测集：Target Atomic-Seen 18。
- 首轮烟测：选择 3 个任务，分别覆盖取放、关节物体操作和电器操作。
- `NavigateKitchen`：属于 atomic，但在 PandaOmron 完整动作链路验证后再加入烟测。
- 主实验初始化：公开的 X-WAM cross-embodiment pretrained checkpoint。
- 消融初始化：只加载 Wan2.2-TI2V-5B，X-WAM 新增模块由代码初始化。
- 首个模态基线：RGB-only。
- 深度：离线生成并缓存，经过小规模试点后才决定是否全量处理。

## 三、职责划分

### Codex / 本地侧

- 编写数据、观测、动作、checkpoint 和 benchmark 适配代码。
- 编写配置、测试、集群命令和文档。
- 完成不依赖 Torch 的语法、schema 和单元测试。
- 每次修改同步更新 `CHANGELOG.md` 和 `PROGRESS.md`。
- 提交并推送到 `dev/atomic-robocasa365`。
- 根据指定 commit 的超算反馈继续修改。

### 用户 / 星光超算侧

- 管理数据、模型权重、RoboCasa assets、环境和调度资源。
- clone 或 pull 指定开发分支。
- 执行仓库中给出的数据审计、训练和评测命令。
- 反馈 commit SHA、命令、配置、GPU、环境版本、Job ID、日志和结果。
- 执行我提供的深度离线生成程序；不需要人工标注深度。

## 四、里程碑与验收门禁

### M0：工程与协作基线

工作内容：

- 配置 fork、官方 upstream 和开发分支。
- 建立仓库规则、架构文档、进度记录、变更记录和集群反馈模板。
- 建立项目级 workflow skill。
- 加入文档同步检查，防止修改逻辑后遗漏记录。
- 记录外部模型路径，不在 Git 中复制权重。

阶段执行过程：

1. Codex 在本地检查 fork、upstream、开发分支和仓库状态。
2. Codex 建立项目规则、文档、workflow skill 和文档同步门禁。
3. Codex 完成本地静态检查，提交并推送开发分支。
4. 用户在星光超算 clone 开发分支，确认 commit 和工作区状态。
5. 用户反馈 clone 结果；Codex 将证据写入进度和变更记录后关闭 M0。

验收条件：

- Skill 结构校验通过。
- 文档同步检查通过。
- Python 文件在不导入 Torch 的条件下完成语法编译。
- 开发分支成功发布，服务器可以 clone。

### M1：RoboCasa365 数据契约与原生 loader

工作内容：

- 读取并校验官方 LeRobot `meta/`：`info.json`、`tasks.jsonl`、episodes 元数据、`modality.json`、`embodiment.json`。
- 固定 Atomic-Seen 18 的版本化任务清单。
- 验证官方 PandaOmron 数据契约：16 维 state、12 维 action、三路 RGB 相机。
- 支持官方数据目录和其 `lerobot/` 子目录两种入口。
- 增加 metadata-only 审计，在加载 Torch、视频或 Parquet 前发现版本和范围错误。
- 实现原生 LeRobot/Parquet adapter，不全量转换为旧 X-WAM JSON。
- 映射 RGB、状态、动作、episode、时间戳、语言和任务 ID。
- 将相机选择改为配置驱动。
- 将 depth 改为真正可选：关闭时不检查目录、不解析路径、不读文件、不进入 augmentation。
- atomic-only 模式下拒绝无法确认任务范围或不在任务清单中的数据。
- 增加 episode 长度、媒体缺失、时序、维度和任务计数审计。

阶段执行过程：

1. 用户提供一个真实 atomic 任务的数据路径，以及最小 episode、元数据和三路 RGB 视频。
2. Codex 先实现不依赖 Torch 的 metadata audit，固定任务范围、目录结构和维度契约。
3. 用户在星光上运行 audit，并反馈 commit、命令、JSON 报告及目录结构。
4. Codex 记录真实数据证据并提供无侵入环境 audit；用户先审计 `abot_m05`，通过后 clone 为独立 X-WAM policy 环境。
5. Codex 根据真实报告实现 Parquet/MP4 到 X-WAM tensor 的原生 adapter，并补充合成数据测试。
6. 用户在 A100/A800 上读取一个真实 RGB-only batch，反馈 shape、dtype、帧序、动作窗口和错误日志。
7. Codex 修正问题并记录集群证据；全部 M1 验收项通过后进入 M2。

需要的超算输入：

- RoboCasa365 数据根目录。
- 至少一个真实任务数据目录；包含 `meta/`、一个 Parquet episode 和它引用的三路 RGB 视频。

验收条件：

- metadata-only 审计通过。
- 一个真实 RGB-only batch 的 shape、dtype、帧序和动作窗口正确。
- 关闭 augmentation 后，同一索引重复读取完全一致。
- `use_depth=false` 时没有任何 depth 目录或 key 依赖。
- composite 数据不能进入 atomic-only 运行。

### M2：观测、动作与 checkpoint 适配

工作内容：

- 建立 `ObservationAdapter` 和 `ActionCodec`。
- 从官方 `modality.json` 冻结 PandaOmron state/action 的命名、切片、范围和版本。
- 正确处理 `base_motion`、`control_mode`、末端增量和 gripper。
- 定义 normalization、clip、控制模式编码和环境动作打包。
- 删除旧 evaluator 只执行前 7 维动作的逻辑。
- 增加 legacy 14D X-WAM action head 到目标 12D schema 的 shape-aware checkpoint loader。
- 同时支持 `xwam_pretrained` 和 `wan_base` 两种初始化。
- 输出 loaded、remapped、initialized、missing 和 unexpected 参数报告。

阶段执行过程：

1. 用户提供真实 `modality.json`、样例 state/action 和已下载 checkpoint 的实际路径检查结果。
2. Codex 依据官方字段建立带名称和切片的 `ObservationAdapter`、`ActionCodec` 与 schema 清单。
3. Codex 实现两种初始化模式和 shape-aware checkpoint adapter，并完成无 GPU 的 schema/映射测试。
4. 用户在 A100/A800 上分别加载 X-WAM pretrained 与 Wan-base，保存参数加载报告。
5. 用户运行一个 batch 的 forward/backward 和动作 round-trip；反馈显存、tensor shape、日志和结果。
6. Codex 根据证据修正映射；动作无静默丢失且两种初始化均可解释后进入 M3。

验收条件：

- 数据动作 → 归一化模型动作 → 反归一化环境动作 round-trip 通过。
- 每个动作分量都有名称、切片、范围和测试。
- checkpoint 不允许静默忽略 shape mismatch。
- A100/A800 上完成一个真实 batch 的 forward/backward。

### M3：RGB-only Atomic 训练烟测

工作内容：

- 将配置拆分为 model、dataset/schema、hardware 和 experiment 四层。
- 提供 A100/A800 debug profile：batch size 1、梯度累积、gradient checkpointing、可选 offload。
- 为只有 120 GiB 主机内存的 A800 门禁提供独立低内存 profile；该 profile 可降低 CPUAdam state 精度，但不得继承到 H100 正式训练。
- 完成显存测量后提供 H100 profile。
- 增加确定性 seed、resume、checkpoint 元数据和简洁日志。
- 先用极小样本确认 loss 有限、监督分支生效并观察短程下降趋势，再进行三个任务的短训练；烟测不冒充完整收敛实验。

阶段执行过程：

1. Codex 准备 A100/A800 debug 配置、确定性 seed、断点恢复和日志记录逻辑。
2. 用户先运行 batch size 1 的单步训练，反馈显存峰值、耗时和首个 loss。
3. Codex 根据 40GB/80GB 显存结果调整 gradient checkpointing、梯度累积或 offload。
4. 用户运行单任务极小样本训练，确认 loss 有限、监督 step 可更新且未读取 depth；若使用低精度 optimizer state，只验收工程恢复能力并显式记录。
5. 用户运行三个 atomic 任务的短训练，并执行一次保存/恢复测试。
6. Codex 记录稳定配置和集群证据；短训练及恢复门禁通过后进入 M4。

验收条件：

- 极小样本训练 loss 有限，action/proprio 监督分支实际出现，短程趋势与参数更新无异常；不以十步烟测宣称完全过拟合。
- checkpoint 保存/恢复后的下一步行为在容差范围内一致。
- 记录 A100/A800 显存峰值和吞吐。
- RGB-only 训练不读取任何 depth 文件。

### M4：RoboCasa365 闭环评测

工作内容：

- policy server 和 RoboCasa simulator 使用独立环境，通过 broker 通信。
- 固定 Atomic-Seen 任务、horizon、seed、layout 和 object split。
- 实现新版 PandaOmron observation 提取和完整动作打包。
- 保存 episode 元数据、成功状态、结束原因、视频、延迟和动作诊断。
- 支持中断恢复、缺失 rollout 检测和结果聚合。

阶段执行过程：

1. M4.0 先在用户已有的 RoboCasa Conda 环境中运行无侵入审计；区分 RoboCasa365 `1.0.1`、只能运行旧任务的 `0.2.x` 和缺少 assets/EGL 的环境。
2. 候选环境必须注册 Atomic-Seen 18，并完成 `CloseFridge(target)` reset、16D state、12D 字典动作、单步和三路 RGB/EGL 渲染；审计只读环境，不安装或升级依赖。
3. M4.1 冻结 Atomic-Seen 18 的官方 horizon 及源码 provenance；runtime 注册值与版本化 manifest 不一致时阻止评测。
4. Codex 实现 dependency-light benchmark adapter：按 schema 打包在线 16D observation，把完整 12D flat action 按名称转换为 Gym 字典，并实现逐 episode 证据与可重算聚合。
5. 用户先在 `robocasa-abot` 运行固定 `CloseFridge(target, seed=0, layout=1, style=1)` 的 20-step 随机门禁，验证环境创建、非零底盘动作、三相机视频和完整结果写入。该短 horizon 只验证工程链路。
6. Codex 审计 M4.1 集群证据；通过后实现版本化 broker 协议，并继续让 policy server 与 RoboCasa simulator client 使用两个独立环境。
7. M4.2 先加载与请求任务严格匹配的 M3 单任务 checkpoint，执行一次三路 RGB + 16D state 请求，返回 32x12 action，并只执行首个 4-action chunk；用户反馈三进程日志、server 加载报告、逐 episode 证据和视频。
8. 单请求通过后增加 M4.3 可恢复长运行：policy response 和每个环境 step 原子落盘；中断后按已记录12D动作重建并逐步校验16D state；视频使用可恢复帧缓存，server 用 fsync JSONL 留存逐请求证据。先故意中断并恢复同一个 run，再完成 `CloseFridge` 官方900-step上限。
9. 用户运行 `NavigateKitchen`，确认底盘动作不是被截断为零。
10. Codex 修正闭环问题并验证聚合可重算；单任务闭环和动作完整性通过后进入 M5/M6。

验收条件：

- 至少一个独立 simulator 环境得到 `reuse_recommendation=reuse_ready`。
- 随机或脚本策略能创建环境，执行所有请求 step，并写出 metadata、逐 episode JSON、三相机视频和可重算 summary；任务随机失败不等于工程门禁失败。
- X-WAM checkpoint 能跑完一个闭环 rollout，不出现动作 shape 错误。
- `NavigateKitchen` 可以产生并执行非零底盘动作。
- 聚合成功率可由 per-episode 记录重新计算。

### M5：离线深度试点

工作内容：

- 使用 `states.npz` 和压缩 MJCF 恢复模拟器状态。
- 对 1～3 个 atomic 任务离线渲染三路 depth。
- 以 episode/frame/camera 为键保存版本化缓存和生成元数据。
- 校验 RGB/depth/action 时序、像素对齐、单位和范围。
- 评估生成速度和存储成本，再决定是否扩展。

阶段执行过程：

1. Codex 编写独立的状态恢复、深度渲染、缓存索引和对齐审计工具，训练 loader 内不调用 MuJoCo。
2. 用户选择 1～3 个 atomic 任务，在星光离线生成少量深度缓存。
3. 用户反馈生成速度、磁盘占用、失败 episode、样例 RGB/depth 和元数据。
4. Codex 检查像素与时间对齐、单位、范围和缺帧，并修正缓存格式。
5. 用户运行一个 RGB-D batch 和短训练烟测。
6. 双方依据质量、速度和存储证据决定是否扩大深度生成；未通过时保持 RGB-only 主线。

验收条件：

- RGB/depth 可视和数值对齐检查通过。
- 缓存索引与 RGB 帧逐帧一致。
- 训练 DataLoader 中不会调用 MuJoCo 渲染。
- 短程 RGB-D 训练烟测通过。

### M6：Atomic-only 正式训练与评测

工作内容：

- 固定 Atomic-Seen 对应18个任务的 `pretrain/atomic` 数据清单，以自然样本比例扫描跨任务 normalization statistics。
- 在单节点4张正式accelerator上以GBS 128、5 epoch、RGB-only训练X-WAM pretrained主实验；当前Clariden目标为4×GH200，保留既有H100兼容配置。
- 资源允许时运行 Wan-base 初始化消融。
- 使用固定 seed 和 checkpoint 选择规则评测全部 Atomic-Seen 任务。
- 汇总 per-task、per-skill、整体成功率、延迟和失败类型。

阶段执行过程：

1. Codex 提供18任务数据审计与全局统计工具；用户在集群生成不可变 manifest/global stats，工具据有效clip数计算 `5*floor(N/128)` 正式step。
2. 用户在目标正式accelerator上先运行2步、保存并恢复到4步；机器审计确认GPU型号/显存、GBS 128、FP32 optimizer state、多卡RNG、有限loss和完整checkpoint。
3. 用户启动正式训练；每次反馈 commit、配置、Job ID、checkpoint、训练曲线和异常日志。
4. Codex 只针对已记录 commit 诊断问题，并将修改推送为新的可追踪 commit。
5. 用户用固定 seed 对 Atomic-Seen 18 执行闭环评测，补跑缺失或明确记录失败 rollout。
6. Codex 汇总 per-task、per-skill、总体指标、延迟和失败类型，明确结果为 atomic-only 设置。

验收条件：

- 训练可断点恢复，所有产物记录 Git commit、完整配置和数据清单。
- 正式RGB主实验精确使用18任务、自然采样、单节点4张同型号正式GPU、GBS 128与5 epoch；当前Clariden运行冻结为GH200，门禁checkpoint不作为正式初始化。
- 所有预期 rollout 均存在，或有明确失败记录。
- 报告明确标注 atomic-only，不冒充 Human300/composite-trained 的标准榜单设置。

### M7：复现与维护

工作内容：

- 为无依赖测试和文档检查增加 CI。
- 在首次集群环境验证后冻结环境依赖。
- 记录 upstream 同步和兼容性检查方法。
- 形成从 clean clone 到最终指标的完整操作手册。

阶段执行过程：

1. Codex 根据已经验证的星光环境冻结依赖、配置模板、命令和 CI 门禁。
2. 用户在新的目录或节点执行 clean clone，不复用旧工作区的手工修改。
3. 用户按手册依次验证数据、权重、单 batch、短训练、恢复和一个闭环 rollout。
4. 用户反馈所有命令、commit、环境和产物位置，Codex 修正文档中的缺口。
5. clean clone 可以仅依赖仓库说明和外部路径复现后，冻结最终复现记录并关闭项目阶段。

验收条件：

- 新的服务器 checkout 仅依赖仓库说明和外部数据/权重路径即可复现已验证烟测。

## 五、主要风险与处理

| 风险 | 处理方案 |
| --- | --- |
| 旧 X-WAM action head 为 14D，RoboCasa365 为另一 schema | 显式 ActionCodec + shape-aware checkpoint loader，禁止静默截断。 |
| 旧 loader 即使关闭深度仍强制读 depth | 让 modality 配置控制目录检查、路径解析、batch 和 loss。 |
| 本地没有 Torch | 本地只完成 static 门禁，Torch/CUDA/模拟器统一标记 `cluster-pending`。 |
| A100 40GB 可能无法容纳调试配置 | batch 1、gradient checkpointing、ZeRO-3/offload，先测量再固定 profile。 |
| 模型和模拟器依赖冲突 | 保留 broker 边界，使用两个 Python 环境。 |
| atomic-only 结果与标准 Human300 结果混淆 | 在数据清单、实验名和报告中固定训练范围。 |
| 深度处理成本失控 | 先做离线小样本试点，缓存结果，再决定是否全量生成。 |

## 六、每次修改的固定流程

1. 只选择一个可验收的里程碑子任务。
2. 在 `dev/atomic-robocasa365` 修改代码并补测试。
3. 同一个 commit 更新 `CHANGELOG.md` 和 `PROGRESS.md`。
4. 本地运行静态门禁，无法运行的部分标记为 `cluster-pending`。
5. 推送 commit，在服务器针对该 SHA 执行仓库给出的命令。
6. 记录超算证据；通过后关闭子任务，否则基于同一记录继续迭代。
