# 变更记录

## 2026-08-21 — 取消M6正式训练的Git clean阻塞门禁

- Job `3133147`的两个节点均在`phase=training`、模型启动前被节点shell中的第二次`git status --porcelain`阻塞；两份train log均为空，因此不是OOM、NCCL、W&B认证或模型配置问题。
- M6正式训练不再要求运行仓库保持clean：外层发现改动时只打印commit、warning和文件列表，节点内重复clean检查已删除。Git commit、dirty状态及具体文件仍由run metadata记录，保留追溯信息但不再消耗正式allocation阻塞运行。
- 正式chunk audit由`clean_git`改为`git_commit_recorded`，只要求有效commit，不因运行时生成的未跟踪文件判失败。该放宽只针对M6正式训练；数据冻结、debug恢复门禁及发布前本地审查规则不变。
- 新增dirty Git正式chunk回归，证明`dirty=true/status!=[]`时审计仍通过；真实2节点8卡重新启动为`cluster-pending`。

## 2026-08-20 — 增加Slurm跨shell变量传递强制审查门禁

- Job `3129585`暴露出原有审查只覆盖sbatch语法和目标脚本静态断言，无法发现外层变量在单引号节点shell中使用却未通过`srun env`传递的问题；这种错误会在排队结束、资源已分配后才由`set -u`触发。
- 项目workflow skill新增无依赖静态检查器，审计全部`deployment/clariden/*.sbatch`中的“外层定义 → `srun env`显式传递 → 节点shell读取”合同。缺少任意变量即返回非零，并报告文件、节点shell行号和变量名。
- 新增正反回归：故意漏传`MILESTONE_STEP`必须失败，显式传递和节点内局部变量必须通过；另对当前全部Clariden作业执行全仓合同检查。
- 发布门禁`check_change_record.py`现自动调用该审计，并把`deployment/`纳入必须同步更新中文CHANGELOG/PROGRESS的material scope。`AGENTS.md`和项目skill明确规定`bash -n`不能替代跨shell变量审查。
- 本地全仓29份sbatch合同检查和4项聚焦测试通过；该机制不执行Torch、Slurm或GPU代码，属于`local-static`，无需消耗超算排队资源。

## 2026-08-20 — 修复Atomic9 RGB-D两节点训练未传递里程碑步数

- Job `3129585`的外层failure report停在`phase=training_srun/line=258`。结合该commit的两节点命令可确定：外层已解析`MILESTONE_STEP=6855`，但`env` 参数没有把它传入节点shell；内层`set -u`在训练入口前读取未定义变量并立即退出。该Job未进入模型加载、forward/backward或checkpoint，不是OOM。
- 两节点`srun` 现显式传递`MILESTONE_STEP`，节点内仍使用已审计的6855步保存第5 epoch永久checkpoint。14,000步停止点、8卡拓扑、depth配置和其他保存逻辑不变。
- 新增静态回归防止外层变量存在但分布式节点未传递。本地测试和sbatch语法通过；修复后首次两节点启动为`cluster-pending`。

## 2026-08-20 — Atomic9 RGB-D改为14000步并永久保留真实第5 epoch

- 集群preflight确认Atomic9有175,514个有效clips，GBS128时每epoch为1,371步；因此真实第5 epoch是step6855，8 epoch是10,968步。之前的8500是人工固定步数，实际约6.20 epoch，不是5 epoch。
- 根据用户对单任务step1500结果的考虑，Atomic9 ratio0 RGB-D正式停止点改为固定14,000步（约10.21 epoch），不再以8 epoch作为停止标准。任务、自然采样、global stats、ratio0、seed42、LR/warmup和2节点8卡配置不变。
- 新的14000-step preflight从PASS JSON导出只包含整数的schedule env。正式停止点严格为14,000，第5 epoch里程碑仍依据真实`steps_per_epoch×5`计算，不把旧step8500误标为5 epoch。
- 常规IOPS滚动checkpoint仍每500步保存、最多5个；Store持久checkpoint逻辑不变。额外增加独立`milestones/epoch5-step=*.ckpt`，在第5 epoch结束时保存完整8-rank训练状态，不受滚动top-k淘汰。
- 通用正式训练合同现支持显式非5-epoch计划；旧RGB/Atomic18默认仍为5 epoch。正式audit在已越过里程碑步数时检查第5 epoch checkpoint唯一且model/八个optimizer shard完整。
- 本地聚焦测试、sbatch语法、Python编译、Ruff和diff检查通过；新14000-step preflight、第5 epoch集群checkpoint及14,000步训练均为`cluster-pending`。

## 2026-08-19 — Atomic9 ratio0 RGB-D全量缓存与8卡正式训练入口

- CloseFridge RGB-D单任务评测已完成：step1000的seed42/seed7均为48%，step1500为88%/84%。这证明训练链路可用，但不能单凭单任务成功率区分更好的收敛与过拟合；根据用户决定，不再为这一点追加单任务训练，直接进入Atomic9 RGB-D对照。
- 新增Atomic9全量depth cache作业：复用已冻结的全局inverse-metric-depth encoding，按现有Atomic9任务清单生成全部episode和三路camera缓存。作业依赖sidecar恢复；已完成的CloseFridge不重生成，12小时到时后可重提同一作业继续补齐。
- 新增Atomic9 RGB-D data/model/hardware/experiment组合：任务、自然比例采样、16D/12D global stats、seed42、LR、warmup、`clean_action_ratio=0`和8500 optimizer steps均与Atomic9 RGB ratio0对照一致。depth不改变state/action normalization，因此复用已审计的RGB manifest/global stats。
- 正式训练固定2节点×4 GH200、单卡batch4、累积4、GBS128、ZeRO-1 GPU AdamW、BF16计算、FP32 optimizer state和gradient checkpointing；使用独立实验/W&B/checkpoint目录，从公开X-WAM pretrained step0启动，不读取CloseFridge单任务权重。
- 通用M6正式入口增加向后兼容的model/depth覆盖。RGB默认行为不变；RGB-D在启动前强制检查缓存、encoding和9任务PASS manifest，正式合同要求`use_depth=true/depth_loss_weight>0`，运行审计要求每个已记录step的depth loss有限且大于0。
- 本地完成sbatch语法和37项聚焦回归；全量depth生成、8卡显存/NCCL、正式训练和恢复均为`cluster-pending`。

## 2026-08-18 — 修正RGB-D评测的experiment/checkpoint分离路径

- Job `3111158`在评测preflight的`test -s "$EXPERIMENT_DIR/config.yaml"`失败；未进入模型加载、server或client阶段。
- 原因是评测脚本误把IOPS滚动checkpoint目录当作experiment目录。训练实际将resolved `config.yaml`保存到Capstor scratch `experiments/xwam/...`，将频繁写入的checkpoint保存到IOPS `xwam_run/.../checkpoints`。
- 评测入口现将`EXPERIMENT_DIR`与`CHECKPOINT_ROOT`显式拆分：前者只读`config.yaml`，后者定位step1000并唯一搜索step1500。训练逻辑和现有checkpoint均不改动。
- 修正后的四路评测仍需Clariden重新提交，状态为`cluster-pending`。

## 2026-08-18 — CloseFridge RGB-D四路checkpoint/seed评测

- 保持CloseFridge RGB-D训练配置和2节8卡训练作业不变；训练由用户手动在期望步数停止，本轮不改trainer/planner/scheduler停止逻辑。
- 新增一个CloseFridge RGB-D四卡评测入口：GPU0/1各加载一次step1000，GPU2/3各加载一次step1500；每张卡各启动一个policy server和一个RoboCasa client。
- 每个checkpoint分别评测连续seed `42..91`和`7..56`（默认50 episodes）；模型侧采样seed继续固定42，target split、1000环境步、replan20和action denoise10与既有比较合同一致。
- 四路broker使用独立端口、topology、控制文件、日志和结果目录；最终生成四个单路summary和一个`comparison.json`，避免并发产物互相覆盖。
- step1000默认使用用户给定的IOPS `epoch=5-step=1000.ckpt`；step1500按`epoch=*-step=1500.ckpt`唯一匹配，避免在脚本中猜epoch编号。
- policy server允许按`use_depth=true`构造训练时RGB-D结构并严格加载checkpoint，但在线评测仍固定`run_depth=false`，client只传三路RGB和16D state。
- 本地无Torch/Clariden模拟器；四卡同时加载、四路闭环和汇总产物均为`cluster-pending`。

## 2026-08-18 — CloseFridge RGB-D正式训练切换2节点8卡

- 用户回报首版4×GH200、单卡batch8的RGB-D正式设置会OOM；本次没有Job ID、峰值显存或完整日志，因此记录为用户反馈，不推断具体OOM阶段或容量。
- 正式硬件层改为2节点×每节点4张GH200、单卡batch4、累积4，保持GBS128、ZeRO-1 GPU AdamW、BF16、FP32 optimizer state和完整gradient checkpointing。训练步数、LR、warmup、ratio0、seed42及depth loss权重不变。
- 正式作业使用torchrun建立8-rank world，增加跨节点master地址/端口、node-rank、节点独立日志和90分钟分布式timeout；planner只接受8-rank完整checkpoint，最终审计同时要求rank 0～7的optimizer实态报告与8个非空ZeRO shard。
- 实验名、W&B run、Capstor metadata、IOPS滚动checkpoint和Store永久checkpoint全部切换到独立`close_fridge_rgbd_ratio00_seed42_8gpu`。它从公开X-WAM pretrained的step 0启动，不扫描、不隔离、不恢复4卡正式实验或P3 smoke checkpoint。
- 本地没有Clariden多节点/Torch运行时；真实跨节点NCCL、显存、首个optimizer update、8-shard checkpoint及恢复仍为`cluster-pending`。

## 2026-08-18 — RGBD-P3门禁关闭与CloseFridge正式训练入口

- 用户回报Clariden Job `3108551`的CloseFridge RGB-D真实batch、有限正depth loss、四卡optimizer update及step 2→4严格恢复全部通过，联合audit为`ok=true/result=pass`。本次反馈没有commit SHA，因此只记录Job和机器结果，不补造Git provenance。
- 核对确认smoke脚本把运行metadata/result/checkpoint保存在Capstor scratch的独立experiment目录，仓库脚本没有`rm`、cleanup或产物迁移逻辑；Store中的batch/audit与日志保持独立。新增回归测试阻止未来误加smoke删除逻辑。
- 新增CloseFridge单任务RGB-D正式data/hardware/experiment层：使用完整106 episode离线depth cache、同步RGB/depth augmentation、公开X-WAM pretrained初始化、ratio0、seed42、LR `1e-5`、warmup200及固定3000步scheduler/停止点。
- 正式硬件层固定4×GH200、`batch8×accum4=GBS128`、ZeRO-1 GPU AdamW、BF16计算、FP32 optimizer state和完整gradient checkpointing。相比RGB-only的micro-batch16主动减半，为新增depth VAE/DiT分支保留显存余量；不继承P3 debug的ZeRO-2 CPU offload。
- 新Clariden作业先要求Job `3108551`联合audit、106 episode/318 depth视频和干净Git，再用双checkpoint根自动选择最新完整四rankcheckpoint恢复。IOPS每500步滚动保留2份，Store每1000步永久保存且最终step 3000另存；Capstor experiment目录只承载resolved config、metadata/result及W&B run ID，不作为唯一恢复来源。
- 作业结束审计固定检查RGB-D/ratio0/GBS128/ZeRO-1/FP32 state、正depth loss、完整checkpoint和W&B身份；运行手册补充smoke目录核对、安全W&B提交、12小时重提恢复、产物路径和OOM回退规则。本地没有Torch/GH200，真实正式训练显存、吞吐、断点恢复及step 3000结果标记为`cluster-pending`。

## 2026-08-17 — RGBD-P3完整CloseFridge缓存、严格loader与短训练恢复门禁

- 用户回报P2全局编码、三任务小缓存及数值/时序/存储三项检查全部通过且无报错；未提供Job ID，因此进度只记录用户反馈。深度生成器现支持`--encoding-input`复用冻结encoding和`--episodes-per-task=0`生成全部episode，P2原三任务入口继续固定任务数3且保持兼容。
- 新增CloseFridge atomic-only完整缓存清单与12小时可恢复作业，复用P2的`robocasa365_inverse_metric_global_q_v1`，目标106 episode/318个三相机视频，写入独立IOPS atomic cache和Store manifest/audit；不重新标定、不改写原始数据。
- RoboCasa365 loader正式支持可选depth。RGB-D模式必须同时提供cache root、encoding和PASS manifest，并逐项验证task/episode/camera、帧数、sidecar和路径；RGB-only模式禁止携带depth配置。depth使用与RGB相同frame ID，三通道uint8映射到`[-1,1]`并以nearest resize，augmentation保持两者空间同步。
- batch审计器增加RGB-D模式，验证`depths[3,9,3,256,320]`、范围、有限性、确定性、DataLoader batch维和真实路径。训练metadata通过单任务Dataset provenance记录encoding digest、manifest和cache root；runner在`use_depth=true`时显式拒绝缺失depth或与RGB shape不一致的batch，不再延迟到loss计算才产生模糊错误。
- 新增RoboCasa365 RGB-D模型/data/四步experiment配置及4×GH200一体化作业：batch PASS后固定8 clip执行step 0→2保存、从精确checkpoint恢复到step 4。复用既有ZeRO-2 CPUAdam debug硬件层，但审计扩展为要求`use_depth=true`及每步depth loss有限且大于0，同时保留四rankFP32 optimizer和严格恢复合同。
- 本地通过171项dependency-light全套测试、Python编译、Ruff、三份sbatch语法和diff检查；本地没有Torch/RoboCasa/GPU，完整318视频、真实depth batch、forward/backward/checkpoint/resume均为`cluster-pending`。

## 2026-08-17 — RGBD-P2全局逆深度编码、三任务缓存与一体化审计

- 根据X-WAM官方RoboCasa发布样例冻结存储合同为三通道灰度、256×256、20 FPS、H.264/yuv420p；公开代码没有披露MuJoCo米制depth到uint8的唯一公式，因此新增项目版本`robocasa365_inverse_metric_global_q_v1`，明确采用跨任务/相机/帧合并采样的全局inverse-metric-depth q01/q99映射，近处更亮、无效值为0，不冒充上游数值公式。
- 新增`CloseFridge/PickPlaceSinkToCounter/OpenDrawer`三任务pilot清单和一体化生成器。第一遍逐episode加载MJCF/metadata/state完成确定性标定并写不可变encoding JSON；第二遍渲染9个完整episode的三路depth，原子生成27个MP4，原始LeRobot目录保持只读。
- 每个缓存视频新增恢复sidecar，绑定encoding、states/MJCF/episode metadata、源RGB和输出视频digest。重提时验证完整相机并跳过，只补缺失相机；没有有效sidecar的旧视频不会被当作完成产物，新文件完整关闭后才原子替换。
- 自动审计帧数/FPS、相机映射、uint8 shape/range、灰度通道、H.264往返MAE、无效/裁剪像素、生成吞吐、总空间和每帧空间，输出独立cache manifest/audit。Clariden单卡作业固定3任务×3 episode×3相机并要求27视频和机器报告PASS。
- P1真实render由用户回报全部通过且无错误，但未提供Job ID，进度仅记录该反馈。为P2复用将环境创建、episode模型加载和state恢复提升为公共helper。本地通过12项聚焦测试、Python编译、Ruff、sbatch语法和diff检查；真实标定边界、缓存生成及审计为`cluster-pending`。

## 2026-08-17 — RGBD-P1逐episode MJCF/state回放与RGB-D渲染门禁

- 修正真实Atomic9结构报告暴露的假失败：MuJoCo展平state宽度由该episode自身MJCF的`nq/nv`决定，不再要求跨episode一致。结构报告现将宽度集合作为信息/警告保留，仍要求每个`states` 的帧数、有限性及三件回放文件完整。
- 新增RoboCasa runtime专用render probe：按任务创建dataset metadata声明的环境，每个抽查episode都先设置`ep_meta.json`、加载自身`model.xml.gz`，再将`states.npz["states"]`的首/中/尾帧写入对应模型。宽度只与当前episode模型比较，并对state写入后回读误差设置`1e-9`门禁。
- 对三路camera同时调用MuJoCo RGB+normalized depth渲染，使用robosuite官方near/far公式转为metric depth。渲染RGB明确纵向翻转后与同episode/帧/camera的原MP4计算MAE/RMSE/PSNR，默认`MAE<=12`；depth必须全部有限、为正且非常量。
- 每个抽查点保存source RGB、rerender RGB、normalized/metric depth的压缩NPZ以及三联对比PNG。PNG中inverse-depth仅用per-frame q01/q99作可视化，机器报告显式标记其不是训练uint8公式，不会在本门禁中猜测或冻结X-WAM depth编码。
- Clariden入口升级为单1卡作业串行完成Atomic9结构审计与真实render probe：每任务默认前3个episode、每episode首/中/尾3帧、每帧3相机，输出独立JSON和作业级artifacts目录。本地9项聚焦测试、Python编译、sbatch语法和diff检查通过；RoboCasa/MuJoCo/EGL及真实视频对齐为`cluster-pending`。

## 2026-08-17 — RGBD-P0/P1官方回放材料结构门禁

- 根据RoboCasa365官方数据说明和转换源码，冻结离线depth的统一实现路线：所有atomic任务都从`dataset_meta.env_args`创建环境，以逐episode的`model.xml.gz/ep_meta.json/states.npz`恢复精确场景和MuJoCo state，再由同一组三路camera离线渲染；不为不同任务编写任务特定depth逻辑，也不在训练DataLoader中调用MuJoCo。
- 明确X-WAM监督目标为inverse depth，公开RoboCasa数据使用三路256×256/20 FPS灰度H.264 depth视频、单通道重复为三通道并映射到`[-1,1]`。公开资料未说明metric/inverse-depth到uint8的完整公式，因此当前只冻结结构合同，禁止猜测min-max或near/far后直接生成全量数据。
- 新增dependency-light单任务/Atomic9审计器：全部episode检查三个官方回放文件，每任务默认解压三个`states`数组并验证二维、有限值、帧数、state width、episode JSON、gzip MJCF和三路camera；输出明确区分`structural pass`与尚待集群执行的render alignment gate。
- 新增Clariden Atomic9结构审计入口和独立Store报告目录；该作业不加载模型、不渲染depth、不修改原始数据。真实Atomic9报告为`cluster-pending`，回滚本次变更不会删除或改写任何数据/模型/实验产物。

## 2026-08-16 — Atomic9 ratio0.5严格对照训练

- 新增Atomic9 `clean_action_ratio=0.5`训练配置和8×GH200入口，其任务清单、global stats、公开pretrained初始化、seed42、LR、warmup、8500步scheduler、自然比例采样、GBS128、ZeRO-1、BF16、checkpoint频率及W&B group与ratio0保持一致；作业目标在7500停止，以便直接对齐已完成的ratio0 step 7500且不改变前7500步学习率轨迹。
- ratio0.5使用独立实验名、W&B run、IOPS滚动checkpoint、Store最终checkpoint及文件锁，不能从ratio0 checkpoint恢复。仅复用ratio0已经冻结且通过审计的Atomic9 manifest/global stats，避免重新统计数据引入第二个变量；独立preflight继续记录相同的8500步scheduler合同，外层planner固定本轮7500停止点。
- ratio0现有step 5500/7000闭环成功率为46.9%/50.9%，但任务数与`clean_action_ratio`同时相对Atomic18发生变化，尚不能把提升单独归因于ratio。新增对照用于隔离监督密度影响；真实ratio0.5 preflight、训练和闭环结果为`cluster-pending`。
- 本地验收覆盖两份实验配置除ratio和实验身份之外保持一致，训练/预检路径隔离、8500步scheduler/7500步作业目标和既有Atomic9门禁。回滚本次修改只移除ratio0.5入口，不删除任何集群产物。

## 2026-08-15 — Atomic9 ratio0固定8500步正式训练入口

- 冻结与FastWAM评测重叠的9个Atomic任务及用户指定顺序，新增独立task manifest、训练manifest、跨任务global stats和preflight路径；不读取或覆盖Atomic18、CloseFridge A/B的统计量、W&B run或checkpoint。
- 新实验与18任务正式设置对齐：公开X-WAM pretrained初始化、RGB-only、seed42、LR `1e-5`、warmup200、自然比例采样、8×GH200、单卡batch16、累积1、GBS128、ZeRO-1、BF16计算和FP32 optimizer state；唯一训练超参改为`clean_action_ratio=0.0`并固定8500 optimizer steps。
- 正式调度合同新增向后兼容的`fixed_steps`模式。原Atomic18仍按5 epoch自动得到16,390步；Atomic9 preflight同时记录8500步、1,088,000次样本抽取及根据真实有效clip折算的epoch数，训练入口不会再用5-epoch计算覆盖8500步。
- 新增Atomic9数据准备和2节点×4卡训练sbatch。滚动checkpoint仍每500步保留5份，Store永久checkpoint每3000步保存，最终8500步另存；12小时超时后重提同一脚本只恢复本实验最新完整8-rank checkpoint。
- 本地验收覆盖固定任务顺序、9任务stats/manifest合同、固定步数调度、独立路径、父训练入口旧默认兼容、Python/sbatch语法及文档门禁。真实Atomic9 clip数、折算epoch、global stats、8卡训练和恢复仍为`cluster-pending`。回滚本次变更不会删除任何集群产物；若已开始新实验，外部目录需单独确认后处理。

## 2026-08-15 — EGL导入物理编号与runtime逻辑编号分阶段切换

- Job `3085601`否定了上一版“旧robosuite会自动把物理`MUJOCO_EGL_DEVICE_ID`映射到逻辑0”的假设：GPU 2 client已通过import断言并连接policy，但在`gym.make`创建offscreen context时明确报告EGL只枚举`0..0`、收到2；X-WAM GPU 3 client同理退出。`robosuite_models`、Mink和mimicgen缺失只是非阻塞warning。
- 对照用户提供、已经成功运行的FastWAM Atomic18脚本，确认该RoboCasa EDF的正式可用合同为“simulator step看见多张GPU，但EGL固定逻辑0”。FastWAM Atomic9修正版因此让全部client共同使用`CUDA_VISIBLE_DEVICES=0,1,2`与三项EGL/render=0，不再给每个client收窄为单个物理卡，也不会暴露预留给X-WAM的GPU 3。
- X-WAM仍保持GPU 3隔离：子进程启动为`CUDA_VISIBLE_DEVICES=3/MUJOCO_EGL_DEVICE_ID=3`，先让旧robosuite import检查通过；`run_robocasa365_m6_client.py`在RoboCasa注册完成后、`gym.make`之前显式切换`MUJOCO_EGL_DEVICE_ID=0`，满足实际单设备EGL context。日志分别记录import物理编号和runtime逻辑编号。
- 同时修正失败报告阶段：client失败在policy回收后会恢复`phase=close_fridge_eval_client`并直接提示`clients/client_05.log`，不再错误显示为`policy_shutdown`。本地聚焦测试、Python/sbatch语法和变更记录门禁通过；GPU 3真实两阶段切换、FastWAM GPU 0/1/2 client和完整共享评测仍为`cluster-pending`。回滚本次commit会恢复已被Job `3085601`证伪的单值EGL设置，不影响既有日志或结果。

## 2026-08-15 — 共享评测物理GPU/EGL编号修复

- Clariden Job `3085477`的X-WAM-B policy已到READY，但CloseFridge simulator client在外层`srun`返回1。同期FastWAM只有GPU 0上的三个client能导入RoboCasa，GPU 1/2 client均在旧版robosuite `binding_utils.py`断言退出：固定的`MUJOCO_EGL_DEVICE_ID=0`不属于各自的`CUDA_VISIBLE_DEVICES=1|2`；X-WAM的`CUDA_VISIBLE_DEVICES=3/EGL=0`属于同一根因。
- X-WAM单任务sbatch现把`MUJOCO_EGL_DEVICE_ID`设置为选中的物理GPU编号；client launcher不再二次覆盖为0，而是取`CUDA_VISIBLE_DEVICES`第一个物理编号。`EGL_DEVICE_ID`和`ROBOSUITE_RENDER_GPU_DEVICE_ID`仍保持单卡namespace中的逻辑0。独立GPU 0评测行为不变，共享GPU 3评测解析为`CUDA=3/MUJOCO_EGL=3/local_EGL=0`。
- client和policy非零返回现在使用条件分支捕获，避免已安装的`ERR` trap在预期收集返回码之前直接退出；这样会先停止并等待policy，再由明确的client/policy阶段生成failure report。每个client的内部Python traceback仍保存在`logs/clients/client_05.log`。
- 本地验收覆盖Python语法、sbatch shell语法、GPU 3物理/逻辑编号合同、错误返回码捕获及完整dependency-light测试。真实GPU 3 EGL初始化、FastWAM GPU 1/2修复和两侧完整50-episode运行仍为`cluster-pending`。回滚本次commit会恢复固定EGL 0和原错误捕获，不会删除Job `3085477`日志或结果。

## 2026-08-15 — FastWAM Atomic9与X-WAM-B共享4卡评测

- 用户已有FastWAM Atomic9脚本申请单节点4卡，但只在逻辑GPU 0/1/2运行6个policy server和9个simulator client。新增共享提交入口，先启动该既有脚本并等待6个动态端口就绪，再在同一allocation中启动CloseFridge X-WAM-B；X-WAM-B默认解释为A/B中的ratio00组，实验目录、checkpoint、episode数和eval ID均可由提交环境覆盖。
- 单任务X-WAM评测新增`XWAM_EVAL_CUDA_DEVICE`与`XWAM_EVAL_STEP_GPUS`。独立作业仍默认`device=0/step_gpus=1`；共享作业让Slurm step看到4卡，但policy launcher和RoboCasa client都只设置`CUDA_VISIBLE_DEVICES=3`，不会在GPU 0/1/2构造Torch模型或EGL client。
- 两边端口不重叠：FastWAM继续使用按Job计算的26000以上连续6端口，X-WAM-B复用单任务topology的12005/13005并在启动前检查占用。FastWAM和X-WAM-B分别写独立driver、模型、client和结果目录；一侧非零退出不会主动终止另一侧，外层等两边结束后再汇总返回码。
- 共享作业申请96 CPU和450G主机内存，X-WAM所有srun阶段增加`--overlap --exact`，避免在FastWAM step存活时等待资源。该设置隔离GPU，但两边仍共享节点CPU、内存和存储带宽；真实Clariden运行必须观察GPU进程映射、主机RSS、EGL和吞吐，当前为`cluster-pending`。
- 本地验收覆盖两个sbatch的shell语法、共享脚本中的GPU/端口/ratio00默认值和原单GPU默认兼容性。回滚本次修改会移除共享入口并让单任务评测恢复写死GPU 0；不会停止现有作业或删除外部评测结果。

## 2026-08-14 — CloseFridge A/B checkpoint容量与慢盘同步修复

- Clariden实跑确认单份4-rank ZeRO-1完整checkpoint约78 GiB。ratio05按250/500/750/1000保存四份后，IOPS目录达到311 GiB；ratio00在step 500保存阶段出现rank间I/O完成时间漂移，随后一个rank进入ALLREDUCE并触发原30分钟NCCL timeout。该失败与`clean_action_ratio`取值无关。
- 两组滚动保存统一改为每500 optimizer steps一次、最多保留最近2份，并继续用`last.ckpt`链接；首轮step 1000只保留500/1000，后续扩展到step 3000也不会在IOPS累计5份约390 GiB的完整状态。
- A/B硬件层将分布式process-group timeout从默认30分钟提高到90分钟；checkpoint callback可配置保存后barrier，A/B显式开启，只有全部rank均从DeepSpeed保存返回后才记录`checkpoint_save_complete`并继续训练。
- 现有ratio05的step 250/750及ratio00的step 250由用户在Clariden按精确路径清理；不在训练脚本中执行通配删除。ratio00重提时planner将先验证并选择现有完整step 500，真实DeepSpeed恢复、step 1000保存及作业PASS仍为`cluster-pending`。

### A组step 1000独立单任务评测

- 新增单GPU `eval_close_fridge_ab_xwam.sbatch`，默认评测ratio05的Store `final-step=1000.ckpt`。合同固定CloseFridge target split、模型/环境seed42、50 episodes、1000步、replan20、action denoise10、逐环境step视频与20 FPS，输出保持精简的`logs/`、`results/CloseFridge/{videos,result.json}`和根级`summary.json`。
- policy pool增加显式单任务checkpoint模式：不把M6跨任务manifest/statistics/allowed-task错误传给单任务模型，并允许单server在单GPUallocation中映射到CUDA device 0；client pool也支持覆盖可见GPU。默认8-server/16-client M6行为保持不变。
- 评测脚本默认要求独立的`src/xwam-robocasa365-eval` clone，并允许通过`XWAM_EVAL_REPO`覆盖；不要求也不建议在ratio00训练作业使用的主仓库中pull/checkout。A组50-episode真实成功率为`cluster-pending`。

## 2026-08-13 — CloseFridge单任务clean-action ratio A/B

- 新增CloseFridge单任务RGB数据层，使用真实任务自己的`meta/stats.json`、三路相机、`256×320`画面和与18任务正式训练一致的RGB增强；不读取多任务manifest或跨任务global statistics。
- 按用户要求将单任务A/B收敛为单节点4×GH200：单卡batch16、累积2、GBS128、ZeRO-1 GPU AdamW、FP32 optimizer state、4 workers/GPU和完整gradient checkpointing。它不启用M6五epochformal guard，但保留4-rank generator state与optimizer dtype记录。
- 新增严格对照的两份实验配置。除实验/W&B名称与`clean_action_ratio=0.5`（A）或`0.0`（B）外配置完全相同：公开X-WAM初始化、seed42、LR `1e-5`、warmup200、3000-step scheduler、首轮trainer目标1000及每250步保存。
- 新增Clariden提交脚本，`XWAM_CF_VARIANT=ratio05|ratio00`选择组别。两组分别使用IOPS `xwam_run/<实验名>`滚动checkpoint、Store `checkpoints/xwam/<实验名>` final checkpoint、独立W&B run ID、日志和文件锁；重提会从本组最新完整4-rank checkpoint恢复。`XWAM_CF_TARGET_STEPS=3000`只用于闭环比较后继续胜出组，不能改变scheduler或从另一组恢复。
- 作业继承`#SBATCH --export=ALL`并继续只接受提交环境的W&B API key；outer/container错误trap保留完整phase、命令和日志路径。真实两组step 0→1000训练、checkpoint和闭环成功率均为`cluster-pending`。

## 2026-08-13 — NavigateKitchen底盘动作诊断

- M6 client的task result新增紧凑的物理量诊断，分别记录base_motion、
  离散后的control_mode和环境实际产生的base-position delta，不保存逐步
  action trace或额外图片。
- Policy/client pool支持重复传入--server-id与--client-id，可以只加载
  server0并运行client8，不必为了一个NavigateKitchen诊断启动8 server /
  16 client。
- 新增单GPU Clariden诊断job与summary脚本，用一个或多个episode区分
  “模型底盘输出接近0”和“存在底盘命令但环境不移动”。
- 修复诊断episode完成后summary阶段在裸计算节点调用不存在的python而以
  exit 127退出；summary现在通过X-WAM EDF中的Python执行。该错误发生在
  result.json写完之后，不会使已完成的NavigateKitchen episode失效。

## 2026-08-13 — M6正式评测与FastWAM配置及产物对齐

- 首次新client集群运行时，broker报告`discard malformed frontend message frame_count=3`且policy始终收不到请求。根因是M6 client误用ZeroMQ `REQ`连接现有`ROUTER` frontend，REQ自动插入空delimiter形成三帧，而项目既有broker协议要求DEALER产生的`[identity,payload]`两帧。M6 client已恢复与M4.2/M4.3相同的`DEALER`类型，并增加静态协议回归测试；该失败发生在policy执行前，不能解释为模型推理或RoboCasa环境问题。
- 修复首次可运行评测在`TurnOnSinkFaucet/seed44`被`M4.3 新 run 必须从干净 Git 工作区启动`中断的问题。该错误不是robosuite模型、Mink或mimicgen warning导致，而是M6 client逐episode嵌套M4.3 runner后重复执行Git门禁；外层作业已经冻结clean commit、checkpoint、RoboCasa/robosuite commit、assets与配置，因此M6改为独立task client，不再重复逐episode门禁。
- 对照用户下载的FastWAM正式client，将RoboCasa场景改为`target` split且不固定layout/style，prompt直接取当前observation的`annotation.human.task_description`；固定环境/模型seed 42起、1000 step、replan20、action denoise10和12D动作裁剪保持一致。原配置`layout=1/style=1`与FastWAM不是同一场景合同。
- 修正视频时间轴：原M6配置每20个环境step录1帧且以5 FPS播放，1000步失败episode只有约10秒并呈现约4倍动作加速；现改为初始帧加每个环境step一帧、20 FPS，跑满1000步约50秒。视频直接流式写MP4，不保存PNG帧。
- 正式结果根目录改为IOPS `x-wam-eval/atomic18/<eval ID>`，结构收敛为`logs/clients`、`logs/servers`、`logs/server_launcher.log`、`results/<Task>/videos`、`results/<Task>/result.json`和根目录`aggregate.json/summary_atomic18.csv`。移除client深层seed/run/progress目录、逐request JSONL和server状态中全部成功request数组；中断重提跳过已完成seed，中断episode从头执行，最多损失一个episode。
- 本地静态验证覆盖Python编译、Ruff、Slurm shell语法、topology与聚合回归；Clariden上新的1000-step时长、target split轨迹、8 server/16 client并发及最终成功率仍为`cluster-pending`。配置与原评测合同不同，必须使用新eval ID，不能把旧目录中的episode混入。

## 2026-08-12 — Clariden M6 Atomic-Seen 18 并行评测

- 新增版本化`8 server / 16 client / 4 GPU`topology，逐项冻结用户给定的client→server→GPU→task映射及FastWAM参考成功率。每卡两个policy server，每server两个client；`client6`串行`OpenStandMixerHead → CloseToasterOvenDoor`，`client7`串行`SlideDishwasherRack → TurnOnElectricKettle`，最终18个Atomic-Seen任务不重不漏。默认每任务50个episode，环境seed从42开始，模型seed固定42，replan为20，ANS action去噪10步并保留50步video scheduler。
- 扩展RoboCasa365 policy server以严格支持M6多任务checkpoint：从不可变训练manifest解析合法任务，使用manifest绑定的跨任务global statistics，并允许每个server进一步限制topology分配任务。单任务M4 checkpoint行为保持兼容；任务越界、统计缺失、manifest/schema漂移仍立即失败。每个server在构造模型前固定Python/NumPy/Torch/CUDA seed，且每次replan显式使用模型seed42。
- 新增可恢复M6 client：每个任务按seed `42..91`逐episode运行既有M4.3完整horizon evaluator；已完成seed跳过，中断seed使用原Git commit和原子progress执行确定性动作回放后继续。每完成一个episode便原子更新client/task成功率，Slurm超时后可用同一eval ID重提。
- 新增八个固定broker/server池、十六client池和最终聚合器。server按顺序加载以避免八份checkpoint同时读取造成Store/主机内存峰值；client按映射设置EGL GPU。聚合必须收齐16份PASS client summary、18任务和每任务完整episode数，才生成overall成功率。
- 新增`eval_m6_atomic18_xwam.sbatch`：单节点4×GH200、450G主机内存、12小时、`#SBATCH --export=ALL`，分别用既有X-WAM与RoboCasa EDF运行policy/simulator并通过Store控制文件协调。评测checkpoint、实验目录和eval ID必须由提交环境显式提供；外层/子进程日志和failure report均持久化。真实8份模型同时加载、16个EGL client、吞吐、12小时中断恢复和成功率聚合均为`cluster-pending`。
- 同一eval ID额外冻结checkpoint、commit、topology、manifest、global stats和episode数；任一字段变化都会在启动client前拒绝，避免断点续评混入另一模型的episode。最终summary显式记录eval ID和checkpoint路径。
- 按用户要求将完整评测根目录迁移到IOPS新目录`/iopsstor/scratch/cscs/zjingchen/terry_nys/x-wam-eval/<eval ID>`；逐episode结果、PNG帧、MP4、client/server日志、恢复状态和summary不再写Capstor/Store。仅Slurm主日志与failure report保留在Store日志目录。
- 首次step-15000正式评测在`client02/PickPlaceCounterToCabinet/seed42`创建环境时中断：0 environment step、0 policy request，容器内`/opt/robocasa/.../Sink025/model.xml`不存在。对照已运行的FastWAM正式sbatch后确认，FastWAM通过`PYTHONPATH`优先使用`$STORE_ROOT/src/robocasa`与`src/robosuite`，其中包含此前下载的完整assets；X-WAM client误只加入自身仓库，因而回退到SQSH内不完整源码。
- X-WAM评测现复用同一已验证Store simulator源码，外层先验证`Sink025`和两份Git源码并将commit/asset hash写入不可变eval合同；RoboCasa EDF内另执行import来源与asset probe，失败时不会加载八个5B policy server。16个client的EGL变量也与FastWAM闭环保持一致：四张GPU对container可见，但当前已验证的单一EGL设备固定为0。修复后的真实18任务评测仍为`cluster-pending`。
- 本地完成topology/聚合/CLI/协议/Slurm shell语法、Python编译、JSON和既有M4回归测试；本地没有Torch、RoboCasa或Clariden runtime，不能据此宣称正式评测可运行。回滚本次变更不会删除IOPS评测结果、Store checkpoint或日志；若已开始评测，外部目录需单独确认后处理。

## 2026-08-11 — Clariden 四卡恢复门禁关闭与 M6 数据预检

- Job `3053264`在干净commit `a2787ded5106f5178c21010d8378a0d2070e7f88`上完成4×GH200全新step 2保存和严格恢复到step 4。两阶段result均为pass，四rank FP32 optimizer实态、有限RGB-only loss、step 2/4完整checkpoint、恢复源一致性及clean Git provenance全部通过；联合audit为`ok=true/result=pass`，最终输出四卡恢复PASS。每rank峰值显存allocated/reserved为30.677/33.039 GiB，Clariden多卡checkpoint/resume工程门禁关闭。
- 新增`prepare_m6_data_xwam.sbatch`作为正式训练前的CPU/I/O门禁：EDF内只读取`pretrain/atomic`，依次生成Atomic-Seen 18 manifest、与其digest绑定的跨任务global stats，并审计GBS 128、5 epoch精确step。产物持久化到Store，临时memmap写Capstor scratch；worktree dirty、任务缺失/多日期歧义、schema漂移、NaN/Inf、digest不一致或任一机器报告非pass都会阻塞。
- 本地通过dependency-light单元测试、shell语法、Python/JSON和变更记录检查。真实18任务目录解析、全Parquet统计和精确step仍为`cluster-pending`；此作业不代表GH200正式hardware profile已经冻结，也不会启动模型训练。
- 回滚本次commit会移除Clariden M6数据作业并恢复环境状态记录，不会删除Store现有manifest/stats或Capstor实验数据；外部产物如需清理必须单独确认。

### M6 数据门禁通过与 GH200 正式 profile 门禁

- Job `3053322`在干净commit `f923c1d27db6ecf9b07e0d4e9258b6c3b50fa7ef`上完成Atomic-Seen 18数据冻结：18个唯一`pretrain/atomic`任务、493,658 frames、419,706 valid clips，manifest digest为`2db4380bcae894abb9ec26d4d1aa16876f076163f3e47a830266abccefdd7977`。16D state/12D action统计与该digest一致，preflight为`ok=true/result=pass`。
- GBS 128、5 epoch精确计划冻结为每epoch 3,278 steps、每epoch使用419,584 clips并丢弃122个尾样本，总计16,390 optimizer steps。自然比例采样保持数据原始任务规模，不把18任务人工均衡。
- 将原H100专用入口泛化为M6 formal accelerator合同，同时保留H100配置兼容。新增Clariden GH200首选`4×4×8=GBS128`与safe `4×2×16=128`profile，均使用ZeRO-2 GPU AdamW、FP32 optimizer state、完整checkpoint和显式4×GH200/≥90 GiB runtime guard。
- 新增GH200正式profile两阶段门禁及通用审计CLI：使用完整18任务和16,390-step scheduler先到step 2，再从精确完整checkpoint恢复到step 4；审计增加clean commit、accelerator一致、model/四rank optimizer shard、resume源及manifest合同。真实首选profile显存、GPU optimizer、完整checkpoint和恢复为`cluster-pending`，通过前禁止启动16,390-step正式训练。

### Clariden GBS128 batch梯度与ZeRO-1合同

- 按用户的正式训练要求，Clariden GH200三档profile及保留的H100正式profile统一改为ZeRO-1；新增`formal_zero_stage`并在训练前要求它与`deepspeed_stage`及正式策略一致。因此命令行意外覆盖为ZeRO-2会直接失败。已验证的debug/CPU-offload smoke仍保留ZeRO-2，不作为正式训练设置。
- 对照FastWAM Clariden的`4卡×单卡batch 16×累积2=GBS128`，确认两项目都是3相机、9个视频帧且训练DiT；FastWAM使用`384×320`画面和128长度缓存文本embedding，X-WAM使用`256×320`画面但文本上下文长度为512，所以不将FastWAM结果直接当成X-WAM显存证据。
- GH200门禁默认候选更新为`16×2`，并提供balanced `8×4`和safe `4×8`两级OOM回退；三档均保持GBS128、BF16、GPU AdamW/FP32 state、无offload和完整checkpoint。联合audit现在还要求初始/恢复两段ZeRO stage一致。
- 本地只能验证配置、合同、审计和Slurm语法；`mb16/ZeRO-1`的真实峰值显存、optimizer update、checkpoint及resume仍为`cluster-pending`。若回滚本次commit，将恢复GH200的ZeRO-2 `mb4/mb2`候选及H100旧profile，不会删除任何集群产物。

### GH200正式profile门禁通过与5-epoch分段作业

- Job `3053436`在干净commit `f4aad5a15f2df428d36639391763debe03280b68`上完整运行默认GH200 profile：4卡均为NVIDIA GH200 120GB，`batch_size_per_gpu=16`、`accumulate_grad_batches=2`、GBS128、BF16、ZeRO-1、无optimizer offload。峰值allocated约78.231 GiB，最高reserved 92.545 GiB，没有OOM。
- Initial到step 2并写入model+四rank optimizer shard，resumed从该精确checkpoint恢复到step 4；两段的18任务manifest digest、16,390-step scheduler、clean commit和ZeRO stage一致。四个rank的AdamW/DeepSpeed optimizer实态全为FP32且非空，loss有限、depth loss为0。联合audit所有checks为true，`errors=[]/ok=true/result=pass`，mb16/ZeRO-1正式profile门禁关闭。
- 新增`train_m6_formal_xwam.sbatch`作为Clariden正式入口：固定Atomic-Seen 18、自然比例、RGB-only、seed42、5 epoch和16,390 steps，不复用门禁checkpoint。由于门禁约为0.033 optimizer step/s，每个12小时作业以1,000步为正常退出边界；重复提交同一脚本自动从最新完整checkpoint推进。
- 新增dependency-light chunk planner和audit：planner严格要求非空model state和rank 0～3 optimizer shard，不完整checkpoint会原子移入按Job分隔的`incomplete-checkpoints/`而不删除；共享文件锁拒绝同实验并发写入。每段audit检查精确resume源、目标global step、完整checkpoint、FP32 optimizer、有限RGB-only metrics、正式manifest/scheduler和clean provenance。中间段关闭final另存，只有step 16,390写入final checkpoint并执行完整formal audit。
- 回滚本次commit会移除分段正式作业、planner/audit及证据记录，不会删除Job `3053436`的Store日志、Capstor checkpoint或未来正式实验目录。首个正式chunk仍为`cluster-pending`。

### Clariden正式训练接入W&B监控

- 正式GH200实验启用Lightning `WandbLogger`，记录训练loss、学习率和trainer指标；project默认`xwam-robocasa365`，checkpoint不上传为W&B artifact，避免重复传输大型DeepSpeed分片。非正式实验默认行为不变。
- 所有12小时chunk共享固定实验目录中的`.wandb_run_id`，logger固定`resume=allow`。首个Job在文件锁内原子创建ID，后续Job和checkpoint resume复用同一ID，避免将16,390步曲线拆成多个W&B run；metadata、chunk audit与final audit都核验online模式、resume合同和预期ID。
- 当前已验收SQSH不重建。新增固定hash的`wandb==0.23.1` IOPS overlay作业，先在临时目录执行真实offline init/log/finish，再原子发布；未来镜像requirements、constraints和image import门禁同时固定同一版本。正式作业在加载5B模型前验证overlay来源、精确版本和在线凭据，失败不会进入训练。
- 按用户要求，正式作业不再接受settings或`~/.netrc`默认账号回退：配置固定`wandb_require_api_key=true`，sbatch和训练入口都要求当前提交环境存在`WANDB_API_KEY`，容器内移除identity-token覆盖并通过`wandb.login(verify=True)`验证该环境凭据。metadata/audit只记录`auth=api_key_env`，绝不记录key；运行手册使用隐藏`read`、Slurm环境继承和提交后立即`unset`，避免密钥进入shell history、进程参数、Git或日志。
- 修复W&B overlay及正式sbatch只有`set -Eeuo pipefail`却没有统一错误上下文的问题。新增可复用的Clariden `ERR` trap，覆盖overlay下载/临时probe/发布、外层preflight/planner/training srun和EDF内planner、W&B preflight、训练、产物检查、chunk/final audit阶段；失败时主日志明确输出phase、exit code、line、command和报告路径，并分别原子写入`wandb-overlay-<JOB_ID>-failure.txt`或`m6-formal-<JOB_ID>-failure.txt`。内层根因报告先写后，外层`srun`失败不覆盖它；命令在日志和报告前都会移除API key并截断，测试同时验证非零退出、阶段定位和密钥不泄漏。
- 本地通过dependency-light测试、Python编译、JSON及shell语法后才发布；aarch64 wheel安装、现有SQSH直接依赖兼容、在线认证和首个W&B正式chunk均为`cluster-pending`。回滚本次commit会关闭正式配置中的W&B并移除overlay/审计接入，不会删除远端W&B run、IOPS overlay、Capstor checkpoint或Store日志；外部产物清理需单独确认。

### Clariden W&B overlay补齐Sentry依赖

- Job `3053810`成功下载`wandb==0.23.1`，但在原子发布前的临时目录offline probe中因`ModuleNotFoundError: sentry_sdk`退出；阶段化failure report准确记录`wandb_overlay_staging_probe`。训练未启动，最终IOPS overlay未发布，临时目录由EXIT trap清理。
- 根因是overlay使用`--no-deps`保证当前SQSH不被解析器改写，但首版哈希清单只列了W&B本体。现固定FastWAM同环境版本`sentry-sdk==2.58.0`及官方PyPI wheel SHA256；未来镜像requirements、constraints、环境manifest和image import门禁同步冻结该版本。
- 发布前probe现在要求W&B和Sentry模块都来自临时overlay，并根据两者的wheel metadata逐项验证当前环境满足全部声明依赖及版本范围，再真实执行offline init/log/finish。依赖缺失仍会阻塞原子发布，不会污染正式overlay。
- 本地只验证固定哈希、shell/Python/JSON、dependency-light测试和文档合同；aarch64安装及offline probe重试仍为`cluster-pending`。回滚本次commit会恢复缺少Sentry的旧清单，不会删除集群overlay、日志、checkpoint或W&B run。
- Job `3053849`确认Sentry修复有效，但完整metadata门禁继续发现SQSH缺少W&B声明的`GitPython`；失败仍位于临时probe，未发布overlay或启动训练。现一次补齐`GitPython → gitdb → smmap`完整链及三个官方wheel哈希；FastWAM环境记录用于确认依赖链版本基线，但GitPython采用W&B允许且PyPI当前无已知漏洞的`3.1.58`，不沿用已列出安全公告的3.1.45。
- probe新增所有五个固定distribution的版本及overlay来源检查，再递归验证它们的声明依赖范围。未来镜像清单和import门禁同步固定`GitPython 3.1.58 / gitdb 4.0.12 / smmap 5.0.3`；Clariden重试仍为`cluster-pending`。

### Clariden正式训练双层checkpoint存储

- 按用户要求，滚动checkpoint迁移到新建的`/iopsstor/scratch/cscs/zjingchen/terry_nys/xwam_run/robocasa365_m6_atomic_seen18_rgb_seed42/checkpoints`：每500 optimizer steps保存一次完整DeepSpeed状态，`save_top_k=5`并保留`last.ckpt`链接，限制IOPS容量。
- 新建永久目录`/capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/checkpoints/xwam/robocasa365_m6_atomic_seen18_rgb_seed42/checkpoints`：独立callback每3,000步保存，`save_top_k=-1/save_last=false`，不做数量删除；不整除3,000的最终step 16,390也明确保存到这里。
- checkpoint事件新增`rolling/durable/final_durable`tier，metadata冻结两层路径、频率和保留数。chunk/final audit不仅要求IOPS为500/5、Store为3000/unlimited且final目录等于Store，还在每个chunk终点核验实际rolling保存、3,000倍数处的durable保存及最终Store保存事件，防止命令行或callback异常静默改变策略。
- chunk planner扩展为同时扫描IOPS和Store，选择global step最大的完整model+四rank optimizer checkpoint；同step优先后声明的Store副本。两层不完整checkpoint分别原子移入同文件系统的`incomplete-checkpoints/m6-formal-<JOB_ID>`，不执行可能因`EXDEV`失败的跨盘rename。
- 本地验证覆盖双root选择、IOPS较新点恢复、同step Store优先、分盘隔离、配置/Slurm/审计wiring；真实Lightning双callback、500/3000重合step、滚动删除与跨Job恢复仍为`cluster-pending`。回滚本次commit恢复单目录1,000步保存，不会删除任何已生成的IOPS或Store checkpoint。

### Clariden正式训练前日志归档

- 正式训练前保留全部历史工程证据，但把build、validation、overlay、单步、四卡恢复、M6数据和profile门禁等已知前缀统一移动到Store的`logs/xwam/debug/`；新增登录节点归档脚本只执行同文件系统`mv`，不删除文件、不匹配未来正式日志，目标重名时拒绝覆盖。已确认停在outer preflight且未训练的Job `3053803`以精确Job ID作为唯一`m6-formal`例外归档。
- 正式训练默认门禁audit路径同步更新为`logs/xwam/debug/m6-gate-3053436-audit.json`，避免归档后outer preflight误报缺文件。正式作业自己的主日志、train log、failure report和audit继续写`logs/xwam/`根目录，便于监控。
- 本地下载的既有M4/四卡/M6证据已统一放入忽略目录`log/debug/`；这些产物不进入Git。集群归档脚本的真实Store移动由用户执行，状态为`cluster-pending`；回滚代码不会自动移回已归档日志，可用普通`mv`从debug目录恢复。

### Clariden首个正式Job的checkpoint与W&B entity修复

- Job `3054130`在干净commit `3258f8fd7d279ff46d540919a2f69d825b3cdfb4`上通过overlay来源、W&B 0.23.1版本、API-key在线认证、4×GH200拓扑和首段step 0→1000 planner；随后在Dataset/5B模型构造前创建滚动callback时退出。Lightning 2.6.5明确拒绝`save_top_k=5, monitor=None`，因此本轮global step仍为0、没有训练checkpoint，不是CUDA、显存、数据或模型错误。
- 新增dependency-light checkpoint monitor解析：当`save_top_k>1`时固定`monitor=step/mode=max`，利用Lightning内置global-step候选保留最大的最近K个；`-1/0/1`保持原语义。滚动storage metadata和chunk/final audit同步要求`step/max`，防止只绕过callback构造却未证明500步滚动保留合同。
- 本轮API key成功认证为`liuwsh25`，提交环境的`WANDB_ENTITY`为空；这不影响API-key身份验证，只表示W&B使用该账号的默认entity。首版修复误将entity设为必填，导致Job `3054165`在outer preflight退出。现恢复为可选覆盖项，正式作业只强制API key，audit记录entity但不要求非空。已创建的`.wandb_run_id`没有训练曲线或checkpoint，重试会安全复用，不应删除。
- 本地通过dependency-light测试、Python/shell/JSON、Ruff和文档门禁；Lightning 2.6.5 callback真实构造、500/1000步保存及5点淘汰仍为`cluster-pending`。回滚会恢复无monitor及可为空entity的旧行为，不会删除持久run ID、W&B项目、日志或checkpoint。

### Clariden正式训练首轮吞吐优化与分段计时

- 用户回报正式训练已能持续运行且step 500滚动checkpoint正常保存，但按当前吞吐完成5 epoch预计约40小时；FastWAM同类实验约14小时。静态核对确认X-WAM每个micro-batch在线执行冻结的5.7B T5，并在batch内任一样本触发CFG text dropout时再次为整批计算空文本；FastWAM则读取预计算文本embedding。两边可训练参数量不能代表这部分额外计算。
- `XWAMRunner`新增按完整prompt字符串键控的冻结T5输出缓存。每个rank只为首次出现的任务prompt和空字符串运行T5，后续直接堆叠已`detach`的同device embedding；缓存是普通运行时属性，不注册buffer、不写入checkpoint，也不改变512-token上下文、T5权重或训练目标，因此现有step-500 checkpoint仍可恢复。缓存采用lazy填充，Atomic-Seen 18稳定后预期最多19项；重启chunk会重新预热。
- Clariden GH200三档正式profile的`num_workers_per_gpu`从2提升到8，保持`prefetch_factor=2`、GBS128和ZeRO-1不变。正式合同现在要求GH200 resolved config同时包含8 workers、T5缓存和正数计时间隔，防止命令行覆盖后无声退回旧吞吐路径。
- 新增低开销CUDA Event分段计时：每20个optimizer step同步并向console/W&B写入data wait、T5、VAE、DiT forward、backward、optimizer和整micro-batch毫秒数，以及T5 cache hit rate/entry/computation计数。正常step不调用CUDA synchronize；20-step边界的一次同步属于测量开销。optimizer区间从Lightning的`on_before_optimizer_step`到batch end，包含该更新后的极少量框架收尾。
- 本地通过AST/配置性能合同测试、M6/Clariden既有dependency-light测试、Python编译、Ruff和diff检查。真实GH200 cache entries是否收敛到19、各阶段耗时、8-worker I/O稳定性及新吞吐均为`cluster-pending`；不能从本地静态检查宣称40小时已缩短。回滚本次commit会恢复在线T5、2 workers并移除计时，不会删除现有IOPS/Store checkpoint、W&B run或日志。

### Clariden吞吐重试的主机OOM修复

- Job `3055021`从既有step-500完整checkpoint成功恢复并至少运行到step 539；用户未单独提供该Job的Git SHA，因此不补造source commit。分段计时在step 539记录data wait约44.9 ms、T5约21.7 ms、VAE约1102.8 ms、DiT forward约1087.5 ms、backward约1251.4 ms、optimizer约56.0 ms和micro-batch总计约3665.9 ms，证明冻结T5缓存已生效，DataLoader等待不是当时的主要耗时。
- Slurm accounting确认训练step `3055021.1`为`OUT_OF_MEMORY`、`ExitCode=0:125`、`MaxRSS=397.06G`、`MaxVMSize=3475.37G`，作业请求主机内存450G；外层Job随后显示`CANCELLED+`。该step被cgroup以SIGKILL终止，进程无法执行Python异常处理或shell `ERR` trap，因此没有failure report是预期行为，不表示作业没有失败。
- 该重试把每卡loader worker从2提高到8，四rank合计32个worker；每个worker会维护独立Parquet/video cache。结合旧2-worker配置已稳定达到step 500、8-worker运行的397.06G RSS以及data wait仅约45 ms，本次将GH200三档正式profile恢复为每卡2个worker，保留GBS128、ZeRO-1、prefetch和分段计时不变。
- 真实数据prompt来自episode语言变体，并非固定18个任务名。冻结T5缓存改为每rank最多128项的LRU；命中项刷新顺序，超过上限时逐项淘汰并记录`timing/t5_cache_evictions`，同时正式合同固定该上限。按BF16 `[512,4096]`估算，128项embedding payload约512 MiB/rank，不随训练步数无限增长；缓存仍不进入checkpoint，step-500训练状态可以原样恢复。
- 本地针对性测试、Python编译和diff检查通过；2-worker/有界缓存从step 500恢复后的主机RSS、吞吐及完整chunk仍为`cluster-pending`。回滚本次修复会重新启用8 workers和无界缓存，存在复现主机OOM的风险；不会修改或删除现有checkpoint、W&B run、日志或Slurm记录。

### Clariden 4-worker折中试验

- 按用户要求，将GH200首选、balanced和safe三档正式profile从2调整为4 workers/GPU，即四rank合计16个worker；GBS128、ZeRO-1、`prefetch_factor=2`、128项T5 LRU和分段计时均不变。正式配置门禁同步要求4 workers，避免resolved config与试验记录不一致。
- 该选择位于已稳定达到step 500的2-worker配置与触发397.06G MaxRSS OOM的8-worker配置之间，但worker内存不保证严格线性，不能据此宣称安全或更快。集群复测必须同时观察训练step的`MaxRSS/AveRSS`和20-step data wait；若RSS持续增长或接近作业内存上限，应取消作业并回退2 workers。
- 本地静态验证完成后发布；4-worker从step 500恢复的主机内存、吞吐及step-1000 checkpoint为`cluster-pending`。回滚本次commit只会把GH200三档恢复为2 workers，不修改checkpoint、W&B run、缓存上限或外部日志。

### Clariden正式训练直接关闭gradient checkpointing

- 4-worker正式重试在step 539/559的稳定吞吐约0.133/0.131 optimizer step/s；data wait与T5稳定开销均约22 ms，VAE、DiT forward和backward分别约1.16/1.15/1.25秒。该证据确认loader和冻结T5不再是主要瓶颈，继续调整worker或扩大文本缓存不会显著缩短训练。
- 按用户明确决定，不再增加独立短门禁，GH200首选、balanced和safe三档正式profile直接设置`use_gradient_checkpointing=false`。正式合同新增同名阻塞检查，确保resolved config不能通过命令行或配置合并静默恢复重计算；4 workers/GPU、文本长度512、128项T5 LRU、GBS128、ZeRO-1、checkpoint与W&B合同保持不变。
- 该开关不改变参数shape、loss定义或checkpoint格式，允许从现有完整checkpoint恢复；它保存DiT各层activation以避免backward重算，预期提高吞吐但增加GPU显存。用户接受直接正式训练验证，首次无checkpointing运行的CUDA峰值、OOM状态和真实速度均为`cluster-pending`，不能从本地静态检查宣称提速已经实现。
- 回滚本次commit会重新启用GH200 DiT gradient checkpointing，不修改或删除现有IOPS/Store checkpoint、W&B run、日志和数据；若正式Job发生CUDA OOM，可回滚该commit后从同一个最近完整checkpoint恢复。

### Clariden恢复BS16/full gradient checkpointing

- 用户确认`use_gradient_checkpointing=false`的直接正式重试发生CUDA OOM；本次反馈没有Job ID、完整日志、显存峰值或`git rev-parse HEAD`，因此只将结果归为用户确认的失败，不推断具体rank、step或commit provenance。该失败没有提供新的完整checkpoint证据，planner继续选择两层存储中最近的既有完整点。
- 按用户选择恢复已验证的原配置：GH200首选仍为单卡BS16/累积2，balanced/safe保留各自batch梯度，但三档全部重新启用DiT gradient checkpointing。正式合同从“必须关闭”改为“必须开启”，避免resolved config误用无checkpointing路径。
- 不采用BS8/累积4/no-checkpointing组合：固定GBS128时它需要每个optimizer step执行4个微批，关闭重计算可能提速、较小微批的GPU利用率可能降速，缺少实测无法确定净收益；用户决定不再为此增加门禁。文本长度512、4 workers/GPU、128项T5 LRU、ZeRO-1、W&B及checkpoint合同均不变。
- 本地静态验证后恢复正式训练；回滚本次commit会再次关闭GH200 gradient checkpointing并复现已确认的CUDA OOM风险，不会修改或删除任何外部checkpoint、W&B run或日志。

### Clariden正式训练取消1,000步主动退出

- 原`CHUNK_STEPS=1000`来自早期约0.033 optimizer step/s的门禁估算，用于在12小时内正常保存、审计和退出；当前稳定实测约0.131 step/s后，每1,000步只需约2.1小时，造成频繁重启和allocation利用不足。
- 按用户明确接受最多回退500步，正式作业将`CHUNK_STEPS`设为总训练步数16,390。每次trainer都以最终step为目标；12小时先到时由Slurm终止，下一次提交由既有planner扫描IOPS/Store两层并选择最新完整checkpoint。500步滚动保存、最多5个，3,000步Store永久保存和最终另存合同均不变。
- 中间超时Job预期不会完成chunk audit或正常W&B finish；checkpoint之后的尾部W&B记录可能与恢复后实际参数轨迹不一致，且最多浪费500步计算。模型、optimizer、scheduler和保存的随机状态从完整checkpoint恢复；写到一半的checkpoint继续由planner隔离。
- 最终某个Job正常达到step 16,390后仍执行chunk/final audit并要求PASS。本地验证只覆盖planner/Slurm/文档合同，真实12小时超时、下一Job跨超时恢复和最终审计为`cluster-pending`。回滚本次commit会恢复每1,000步正常退出，不删除任何外部checkpoint、W&B run或日志。

### Clariden正式sbatch默认继承提交环境

- 正式作业原本只在运行手册的提交命令中写`--export=ALL`，脚本自身没有对应SBATCH指令；功能等价，但调用者直接使用裸`sbatch`时容易漏传已经导出的`WANDB_API_KEY`并在preflight退出。
- `train_m6_formal_xwam.sbatch`现内置`#SBATCH --export=ALL`，运行手册的命令行不再重复该选项。API key仍通过隐藏`read`后在当前shell中`export`，不会把明文写入命令、脚本、Git或日志；提交后立即`unset`，容器内登录和缺失key门禁保持不变。
- 本地Slurm语法和dependency-light合同测试通过；真实Clariden环境继承与W&B登录为`cluster-pending`。回滚本次commit会重新要求调用者在每条提交命令显式添加`--export=ALL`，不影响checkpoint、W&B run或训练数据。

### Clariden 2节点8卡独立正式训练入口

- 新增`train_m6_formal_xwam_8gpu.sbatch`，固定2节点×4张GH200、单卡batch 16、累积1、GBS128、ZeRO-1、BF16和full gradient checkpointing。8卡实验使用独立实验名、W&B run ID、Capstor实验目录、IOPS滚动checkpoint和Store永久checkpoint，从公开pretrained权重的step 0开始，不读取或转换现有4卡checkpoint。
- 正式训练合同、运行时门禁、chunk planner和chunk/final audit从写死4 rank改为读取显式`formal_world_size/formal_num_nodes/formal_devices_per_node`。4卡入口仍要求1节点/4 rank；8卡入口要求2节点/8 rank，checkpoint恢复必须包含rank 0～7全部ZeRO-1 optimizer shard，optimizer dtype审计也必须收齐8份rank报告。
- 共享正式脚本在单节点继续沿用原Lightning本地启动；2节点allocation改为每节点一个EDF launcher，再由`torch.distributed.run`各启动4个rank，并用首节点hostname和Job派生端口建立rendezvous。rank 0保留主训练日志，第二节点写独立`-node1.log`；任一节点失败同时保留node级failure report，随后由外层failure report记录失败阶段。
- 本地已通过4/8-rank planner、checkpoint、optimizer audit、拓扑/配置合同、Slurm静态合同、Python测试、shell语法和diff检查。真实Clariden跨节点NCCL/rendezvous、8卡首步、step-500八分片保存和12小时恢复均为`cluster-pending`；不能从静态验证宣称吞吐已翻倍。回滚本次commit只移除8卡入口及通用rank参数，不删除现有4卡/8卡外部实验、checkpoint、W&B run或日志。

## 2026-08-10 — Clariden 4×GH200 step 2→4 checkpoint/resume 门禁

- 在单卡单步门禁关闭后新增独立的 Clariden 四卡调试层与两阶段实验层；不直接套用M6 H100正式配置。门禁固定CloseFridge前8个clip、4×micro-batch 1、GBS 4、BF16 compute、ZeRO-2 FP32 CPUAdam offload和四步scheduler。
- `smoke_train_resume_xwam.sbatch` 在同一4×GH200 allocation内先训练到step 2并保存，再解析真实`last_checkpoint`并恢复到step 4。两阶段各自保存console log、metadata、result、checkpoint events和四rank optimizer dtype报告；任一`srun`失败会停止后续阶段。
- 每阶段只由`srun`启动一个持有四张GPU的EDF容器任务，延续仓库既有“Lightning按`devices=4`派生四个本地进程”的合同；训练前清除step级单任务Slurm拓扑变量，避免`SLURM_NTASKS=1`与训练world size 4冲突。
- Debug checkpoint显式排除冻结T5/VAE以控制保存体积；既有严格恢复适配只允许这些冻结参数缺失，任何可训练参数缺失、额外参数或shape错误仍会失败。保留两个step checkpoint便于诊断，不把门禁checkpoint用于正式训练。
- 新增dependency-light联合审计器，验证四张GH200/单节点world size 4、相同干净Git commit和数据、固定scheduler、有限RGB-only loss、四rank实际FP32 optimizer state、step 2/4保存完成、非空model state、四个ZeRO optimizer shard、恢复源精确一致及最终`global_step=4`。
- Clariden Job `3046423` 在commit `1b5f350e5e889f12716d380fe00827e784541eab`上完成initial阶段：4×GH200 120GB、world size 4、四rank FP32 optimizer state audit、step 0/1有限loss、step 2 checkpoint保存和`result=pass`均有日志证据。每rank峰值显存为allocated/reserved 30.677/33.039 GiB。
- 该作业未进入resume阶段：EDF训练`srun`结束后，外层Slurm宿主脚本用`python`解析result，Clariden宿主环境报`python: command not found`。现将checkpoint解析和最终联合审计均移入EDF容器，并允许通过`XWAM_INITIAL_JOB_ID=3046423`复用已通过的initial产物，只重跑step 2→4恢复阶段。复用时审计器会用Git验证新旧commit差异严格局限于冻结的Slurm编排、审计、测试和文档路径；模型、数据、训练配置或runner有任何改动仍会失败。
- 首次复用重试Job `3047286`在进入resumed训练前失败：嵌入单引号`srun bash -lc`的Python heredoc仍包含`payload['result']`等单引号，shell quote removal将其变成`payload[result]`并触发`SyntaxError`。现移除内嵌Python，新增独立、可单测的checkpoint解析CLI；它在EDF内严格验证initial `result=pass/global_step=2/error=null`和checkpoint目录后写出路径。该失败没有启动模型加载或训练，不影响Job `3046423` checkpoint。
- Job `3047744`在commit `ac1552b4d85a9d133e5c383da449d84665524fd0`完成一次新的两阶段四卡运行：initial达到step 2并保存，四rank从该精确checkpoint恢复全部状态后达到step 4；两阶段result均为pass，四rank FP32 optimizer dtype audit均通过，step 0～3 loss有限、depth loss为0，每rank峰值显存allocated/reserved为30.677/33.039 GiB。最终联合audit仍为fail：运行时`git status`记录sbatch文件被本地修改，导致两阶段`clean_git=false`；同时两个checkpoint内均只发现model state、未发现审计合同要求的四个optimizer shard。该结果证明四卡forward/backward和恢复路径可运行，但在确认真实checkpoint文件布局及干净provenance前不关闭正式门禁。
- 服务器只读文件清单确认Job `3047744`的step 2和step 4 checkpoint均包含`mp_rank_00_model_states.pt`及rank 0～3四个非缺失optimizer shard；DeepSpeed 0.19.4实际命名为`bf16_zero_pp_rank_<rank>_mp_rank_00_optim_states.pt`。联合audit的checkpoint layout失败是审计glob遗漏`bf16_`前缀的假阴性，不是optimizer state丢失。审计器现同时接受无前缀和dtype前缀命名，并从文件名提取rank，严格要求恰好覆盖0～3及文件非空；干净Git provenance仍需新作业复验。
- 本地已通过聚焦测试、shell语法、Python compile和CLI help；工作站无Torch/Clariden。四卡initial已通过，但严格恢复到step 4和最终联合审计仍为`cluster-pending`。
- 回退本次commit会移除GH200四卡配置、作业、审计器和测试，不会删除服务器现有单卡结果、模型、数据、overlay或未来产生的外部checkpoint。外部实验目录如需清理必须由用户单独确认。

## 2026-08-10 — Clariden DeepSpeed NVTX domain 兼容修复

- 首次 1×GH200 单步训练 Job `3044897` 已进入 Lightning `training_step`，但 DeepSpeed 0.19.4 在执行模型 wrapper 前调用 `nvtx.get_domain`，当前 SQSH 中被 import 的 `nvtx` 不提供该 API，作业以 `AttributeError` 退出。该次 CUDA 峰值 allocated/reserved 为 30.677/40.076 GiB，不是 OOM；真实 X-WAM forward/backward/optimizer update 尚未得到验证。
- 为避免因一个小型 Python 包重建 16.98 GiB SQSH 并重新编译 FlashAttention，新增 `prepare_runtime_overlay_xwam.sbatch`：在计算节点从精确哈希 requirements 安装 `nvtx==0.2.12` 到版本化 IOPS 目录，先在临时目录验证版本、模块来源和 `get_domain`，再原子发布并记录 Store manifest。已存在的合法 overlay 会直接复用，非法目录会失败而不覆盖。
- `smoke_train_xwam.sbatch` 在启动 5B 模型前显式把该 overlay 放到 `PYTHONPATH` 首位，并再次校验模块确实来自 overlay、版本精确且 API 可调用；缺失或 shadowing 会立即退出。
- `requirements-clariden.txt`、constraints、Containerfile、build import 门禁和环境 manifest 同步固定 `nvtx==0.2.12`，因此未来正常重建镜像会内置同一修复；EDF 模板也记录 overlay 优先级，当前已有 EDF 无需为本次 smoke 手工改写。
- 本地通过 shell 语法、dependency-light 单元测试、JSON 解析、文档同步和 diff 门禁。当前工作站没有 Clariden runtime；ARM64 wheel 下载、overlay import、DeepSpeed domain push/pop 和真实单步训练均为 `cluster-pending`。
- 回滚本次 commit 会移除 overlay 安装/前置检查，并恢复旧依赖清单；不会删除服务器已有 SQSH、IOPS overlay、日志、模型、数据或实验目录。若需删除外部 overlay，应单独确认路径后由用户执行。

### `nvtx 0.2.12` 真实重试反馈与精确签名修复

- 0.2.12 overlay 环境门禁已通过，证明新包被正确优先 import，`get_domain` 也存在；但训练 Job `3046110` 在 DeepSpeed 的下一行调用失败：`DummyDomain.push_range(message=msg, category=category)` 报 `TypeError: push_range() takes exactly 2 positional arguments (1 given)`。因此首版门禁只验证 API 存在仍不充分。
- NVIDIA 的 0.2.15 发布说明明确加入“Domain API 接受事件属性关键字参数”。Clariden pin、wheel 哈希、IOPS 路径、EDF、Containerfile、环境 manifest 和测试统一提升到 `nvtx==0.2.15`；Python 3.10 Linux aarch64 wheel SHA-256 固定为 `a4f50832fd90a1b480a9deef6e4cd48015b61869095b54dd1a7afe87b4138c6a`。
- overlay、Containerfile 和训练 preflight 现在真实调用 `domain.push_range(message='probe', category=None)` 后再 `pop_range()`，与 DeepSpeed 0.19.4 的失败路径保持同一参数形式。0.2.15 使用新版本化目录，不覆盖或删除已有 0.2.12 overlay。
- Job `3046110` 的显存峰值仍是 allocated/reserved 30.677/40.076 GiB，失败仍发生在 wrapper 进入模型 forward 前；数据、模型和 optimizer 不能由这次日志重新判错。0.2.15 ARM64 安装与单步训练为 `cluster-pending`。

### `nvtx 0.2.15` 与单步训练关闭证据

- 用户回报 0.2.15 环境门禁和更新后的 Clariden 单步训练均已通过。该训练脚本只有在 checkpoint 初始化报告为 `pass`，且 run result 同时满足 `result=pass`、`global_step=1`、`error=null` 时才输出最终 PASS，因此真实模型 forward、backward 和一次 FP32 CPUAdam optimizer update 门禁关闭。
- 本次没有提供最终 overlay/train Job ID 或 `git rev-parse HEAD` 输出；manifest 明确保存 `null` 和 provenance 备注，不把修复 commit `a942c34` 自动冒充为集群实测 commit。
- `configs/environment/xwam_clariden.json` 将 `nvtx_runtime_overlay` 与 `training` 更新为 `pass`。这只关闭当前单卡 CloseFridge 工程 smoke，不代表多卡 checkpoint/resume、18任务正式训练或策略质量已经通过。

## 2026-08-10 — Clariden 首次构建 dependency resolver 修复

- Clariden 首次真实 build 已证明 Torch 2.9.0/cu126 overlay 安装成功，但在 requirements 解析阶段中止，尚未进入 Decord、FlashAttention 编译和 SQSH 导出。
- 致命冲突为本仓库固定 `safetensors==0.5.3`，而 Diffusers 0.38.0 的发布元数据要求 `safetensors>=0.8.0-rc.0`；现固定 `safetensors==0.8.0`。
- 固定 `huggingface-hub==0.36.0`，同时满足 Transformers 4.51.3 的 `<1.0,>=0.30.0` 和 Diffusers 0.38.0 的 `<2.0,>=0.34.0`，避免 pip 先选 1.x 再回溯。
- Torch overlay 前一并移除 NGC 24.10 中与新 Torch 耦合的 `transformer-engine` / `transformer-engine-cu12` 和 `torch-tensorrt`。它们不在 X-WAM import 路径上；原日志中它们的 pip 警告不是本次退出原因，但保留会产生已知的坏环境。
- 容器内建验证新增 safetensors 精确版本及三个已移除 NGC distribution 不存在检查；真实 aarch64 build、GH200 kernel、SQSH 和训练仍为 `cluster-pending`。

### 第二次真实 build 反馈

- commit `f27ad003dff93b88188ffcfe92bae0e70c0959ac` 已通过 requirements resolver、Decord 源码构建和 FlashAttention 2.8.3 `linux_aarch64` wheel 构建；生成的 FlashAttention wheel 约 116.8 MB，说明 Python/Torch/CUDA/sm90 编译组合可用。
- 新阻塞位于 Containerfile STEP 17 的无 GPU 软件验证：`from modules.attention` 会先执行 `modules/__init__.py`，而 `T5EncoderModel.__init__` 的默认参数在模块 import 时立即调用 `torch.cuda.current_device()`，Podman build 没有 NVIDIA driver 因而退出。
- 将 T5 默认 device 改为构造时延迟解析：未显式传 device 时仍选当前 CUDA device，但纯 import 不再需要 driver。保留 Containerfile 的 X-WAM/FlashAttention import 门禁，真正的 CUDA kernel 仍由 `validate_xwam.sbatch` 在 4×GH200 上验证。
- 迭代构建建议先申请 normal 交互式 allocation，再在同一节点内直接重跑 `build_xwam.sbatch`；Podman 节点本地 layer cache 可复用已完成的 FlashAttention 层。换节点或 allocation 结束后不保证保留该 cache。

### Clariden 容器与 GH200 验收证据

- commit `6594d898a87145101afe5f8a45fa6ba907b2b3c3` 完成 24 GB Podman image 和 17,803,421,177-byte zstd SquashFS；`unsquashfs -s` 确认为有效 SquashFS 4.0，持久化路径为 `/capstor/scratch/cscs/zjingchen/terry_nys/containers/xwam-ngc2410-cu126-6594d898.sqsh`。
- `enroot import` 在已写出有效 SQSH 后返回非零，导致原脚本未写 EDF/manifest；本轮已手工补齐。build 脚本改为仅在 SQSH 存在且 `unsquashfs -s` 通过时允许带警告继续，否则仍保留原错误码退出。
- 用户回报 `validate_xwam.sbatch` 全部通过：4×GH200 可见、Torch CUDA 12.6、NumPy 1.23.5、FlashAttention 2.8.3 BF16 forward/backward kernel、Wan/T5/VAE/tokenizer、X-WAM pretrained 和 CloseFridge 路径发现均通过。本次未提供 validation Job ID，不补写未知 provenance。
- 新增 1-GPU `smoke_batch_xwam.sbatch`，在 EDF 中真实解码 CloseFridge clip 0，检查三路 RGB、16D state、12D action、确定性和 RGB-only 合同。该门禁与训练分开，当前仍为 `cluster-pending`。
- 用户回报上述真实 batch smoke 通过；Job ID 未提供。Clariden `dataset_smoke` 关闭，只剩真实模型训练 step 待验证。
- 新增 Clariden GH200 单卡硬件层、单步实验层和 `smoke_train_xwam.sbatch`。该作业固定 CloseFridge clip 0、batch 1、BF16 compute、ZeRO-2 FP32 CPUAdam offload、RGB-only 和一次 optimizer update；`clean_action_ratio=0` 只为确保该工程门禁实际覆盖 action/proprio 监督，不作为正式训练语义。
- 单步作业不保存巨型 training checkpoint，但必须产生 checkpoint initialization report 和 `result=pass/global_step=1/error=null`；真实 GH200 model load、forward/backward/optimizer update 仍为 `cluster-pending`。

## 2026-08-09 — Clariden aarch64/GH200 X-WAM policy 容器部署基线

- 修改前基线：`c64681a`（M6 H100 atomic training 实现）
- 目标范围：只迁移已有 X-WAM × RoboCasa365 atomic-only policy；复用 Clariden 现有 simulator、数据和 Wan2.2，不合并两套 Python 环境
- 运行状态：Clariden 资源与官方 ARM64 wheel 已盘点；部署文件和 dependency-light 测试为 `local-static`，实际构建与运行均为 `cluster-pending`

### 环境决策与兼容性

- 选择已在 Clariden 成功构建 RoboCasa 的 `nvcr.io/nvidia/pytorch:24.10-py3`，固定 Ubuntu 22.04、Python 3.10 和 CUDA toolkit 12.6.2；不采用 Python 3.12 的 NGC 25.03。
- 从 PyTorch 官方 cu126 index 固定 aarch64 Torch 2.9.0；torchvision 0.24.0 和 torchaudio 2.9.0 在 ARM64 wheel 名中没有 `+cu126` 后缀，不能按 x86 文件名判断为 CPU wheel。
- 保持原 A800 已验证的 NumPy 1.23.5、Transformers 4.51.3、Diffusers 0.38.0、Lightning 2.6.5、DeepSpeed 0.19.4、PyArrow 16.1.0 和 PyZMQ 27.1.0 核心版本。
- FlashAttention 2.8.3 没有匹配该 Python/Torch 的官方 Linux aarch64 wheel，固定官方 tag commit `060c918`，只为 GH200 `sm90` 源码编译并限制八个并行 job。
- Decord 0.6.0 官方 PyPI 只有 Linux x86_64 wheel；固定官方 tag commit `6a3617c`，使用 FFmpeg 开发库构建 CPU decoder，不更改现有 Parquet/MP4 Dataset adapter。
- 两个源码依赖均先 checkout 固定 commit，再按该 commit 更新递归子模块，避免默认分支子模块与目标 tag 漂移；build allocation 放宽到8小时，覆盖 ARM64 编译和同一 allocation 内的 enroot 导出。

### 部署与复用逻辑

- 新增 Clariden Containerfile、精确 requirements/constraints、持久化 bootstrap、Slurm build/export、EDF 模板和 GH200 validation 作业。
- build 作业强制在同一 allocation 中完成 Podman 构建、software import、`enroot import` 和 manifest 写入，避免 `/dev/shm` image 随 allocation 消失。
- EDF 只挂载既有 Store/Capstor scratch/IOPS scratch，workdir 与 PYTHONPATH 指向 Store repo；缓存写 IOPS，并开启 Hugging Face/Transformers offline，阻止运行时隐式下载。
- bootstrap 只链接 FastWAM 已有 Wan2.2 大文件与 UMT5 tokenizer。经 loader 审计确认 FastWAM 目录缺少 `config.json`、`configuration.json` 和 safetensors index，脚本从 Wan 官方 revision `921dbaf` 只补三个小文件。
- X-WAM checkpoint 只从官方 revision `bb6fd16` 下载 `pretrained` 路径，保留 Wan base、X-WAM pretrained 和 RoboCasa post-training 三层语义；不下载或误用官方 RoboCasa SFT。
- checkpoint 下载使用可恢复的 `.partial` 文件和安静日志；连接中断后重跑会续传 38.9 GB model state，不会覆盖已完成文件或在终端刷出巨量进度行。
- 4×GH200 validation 检查四卡可见、Torch/CUDA/NumPy/FlashAttention 精确版本、BF16 FlashAttention forward/backward kernel、全部 Wan/T5/VAE/tokenizer/X-WAM pretrained 文件和真实 CloseFridge atomic 数据发现。

### 验证、风险与回滚

- 新增 dependency-light 测试，检查 policy/simulator 隔离、ARM64 关键 pin、官方源码 commit、EDF cache/offline 约束和三个 shell 脚本语法。
- 本地没有 Clariden container runtime；Podman build、aarch64 native extension、SQSH、EDF、GH200 kernel、真实 model load、dataset batch 和训练仍为 `cluster-pending`，不得从静态检查推断通过。
- 最大构建风险是 FlashAttention/Decord ARM64 源码编译；失败时保留 build log，并先按 Python/Torch/CUDA/arch/ABI 分类，不升级模型语义依赖。
- 回退本次 commit 可删除 `deployment/clariden/`、Clariden environment manifest、测试和文档记录；不会删除 Clariden 既有 RoboCasa/FastWAM SQSH、数据、权重或作业。

## 2026-08-08 — M6 4×H100 RGB-only训练门禁与正式配置

- 修改前基线：`e9c3bf3`（M4.3集群验收记录）
- 目标范围：只训练 Atomic-Seen 对应18个任务的 `pretrain/atomic` 数据；4张H100 80GB、GBS 128、5 epoch、RGB-only
- 运行状态：本地静态、dependency-light单元测试通过；真实18任务数据统计、4×H100训练、checkpoint/resume和optimizer state审计均为 `cluster-pending`

### 加入的逻辑

- 新增M6 manifest构建器：依据版本化18任务清单解析 `pretrain/atomic/<Task>/<date>`，检查真实Parquet/三路RGB视频，按9帧、frame skip 4计算每个episode的有效clip，并用SHA-256冻结任务路径、数量与Git来源。多日期歧义必须显式选择，任何任务缺失都会失败。
- 新增 `natural_proportional` 多任务Dataset，不再使用M3三任务平衡过采样；每个任务仍保留独立modality检查，但18任务训练统一读取与manifest摘要绑定的跨任务16D state/12D action统计。
- 新增精确全局统计脚本。它逐episode读取Parquet并用临时memmap计算所有frame的q01/q99/min/max，不解码视频、不生成depth；临时文件在成功或异常退出后清理。
- 新增5-epoch自动调度：`steps_per_epoch=floor(total_valid_clips/128)`，正式总步数为5倍。H100专用采样器每轮全局shuffle后只丢弃不足128的尾部，再等长分发给四个rank，避免Lightning默认补样导致epoch/step漂移。
- 新增首选 `4×microbatch4×accum8` 与显存回退 `4×2×16` 两个H100 profile，二者GBS均为128。运行时强制4张至少75 GiB的H100、ZeRO-2、无optimizer offload、FP32 optimizer state、通信overlap、完整checkpoint和RGB-only。
- 模型compute继续使用 `bf16-mixed`；只撤销A800 120 GiB调试时的BF16 optimizer state、CPU offload、冻结参数排除和通信让步。四个rank在首个update后分别记录实际optimizer state dtype，不能只依赖配置声明。
- 多卡checkpoint保存四个rank的自定义generator state并禁止跨world-size恢复。rank 0独占公共config/metadata/result/checkpoint事件，四rank共享run ID，避免并发覆盖；正式结束显式生成final checkpoint。
- 新增H100门禁和正式实验配置、preflight/2→4步恢复/正式结束三类机器审计。门禁沿用完整5-epoch scheduler但只运行到step 2，再恢复到step 4；门禁通过后正式实验必须从公开X-WAM pretrained权重新启动。

### 验证、限制与回滚

- 本地通过聚焦Ruff、compileall、全量97项dependency-light单元测试及五个新CLI `--help`；本地没有Torch/H100，不宣称多卡hook、DeepSpeed保存或真实吞吐已经通过。
- 正式数据统计需要额外临时空间，约为 `total_frames×28 bytes` 加NumPy quantile工作空间；建议把 `--work-dir` 放在HDD_POOL。日志、manifest、stats和实验产物均在Git ignore目录，不上传Git。
- H100门禁若首选profile OOM，只允许从头改用safe profile；不能继承A800低内存profile，也不能把门禁step-4 checkpoint接入正式训练。
- 回退本次commit会恢复M3已有单/三任务工程路径；不会删除服务器数据、global stats、checkpoint或实验目录。

## 2026-08-08 — M4.3 900-step可恢复评测实现

- 修改前基线：`8790a06`（M4.2集群验收记录）
- 验收实现：commit `f9e1b6bebde87eeb99cff2151a0eef3bc2501a31`，分支 `dev/atomic-robocasa365`，工作区干净
- 运行状态：本地dependency-light检查和A800真实900-step恢复评测均通过；M4.3正式关闭

### 加入的逻辑

- 新增M4.3完整配置，固定`CloseFridge(target, seed=0, layout=1, style=1)`官方900-step horizon、4-action重规划、RGB-only、900秒单请求超时和每20步视频帧。
- 新增版本化恢复合同。每次policy response先原子保存待执行动作和checkpoint/seed/latency，再在每个环境step后原子保存12D action、16D state及terminal标志；只有最后一个请求允许存在未执行动作。
- 新增可恢复client。中断后必须使用原run目录和原Git commit；它以相同seed重建环境，逐条回放已经执行的动作并比较16D state，默认最大绝对误差容差为`1e-5`。漂移、任务/config/checkpoint变化或工作区dirty都会阻止续跑。
- action chunk在请求完成后持久化，因此中断发生在chunk内部时可以执行剩余动作而不重新调用5B模型。M4.3 request ID包含run id，避免server journal与早先smoke记录冲突。
- 视频改用原子PNG帧缓存：reset、每20步和terminal分别保存，episode完成后统一编码MP4。中断不会留下唯一且不可恢复的损坏流式视频。
- Policy server新增逐请求fsync JSONL、最小请求数和“满足门禁后Ctrl-C正常退出”语义；保留M4.2 `max_requests=1`默认行为。
- 新增长运行审计器，交叉验证client metadata/summary/episode/progress、server JSONL、请求数、每次`[32,12]`、单一checkpoint、完整horizon或提前成功、逐帧缓存和非空视频。

### 验证、限制与回滚

- 新增纯NumPy进度状态机、state漂移阻塞、完整900-step/225请求合成证据和dependency-light CLI测试；本地不导入Torch或RoboCasa。
- A800 run `20260808T023440Z` 完成固定 CloseFridge 官方900/900 step和225次policy request；故意中断后恢复、server journal、单一checkpoint、请求shape、16D/12D合同、完整动作边界和46帧视频证据全部通过。
- 机器审计19项checks全真，`errors=[]`、`ok=true/result=pass`；真实同seed动作回放满足配置的state漂移门禁，M4.3不再是`cluster-pending`。
- 本机归档的M4.3证据为`log/audit.log`；同级其他JSON仍属于M4.2旧run，完整M4.3原始产物保留在超算忽略目录且不上传Git。
- 当前M3 checkpoint只用于工程闭环，900-step结果不解释为有效策略质量；完成M4.3后才进入H100训练测试。
- 本轮900步`success=false`，符合10-step工程checkpoint不具备有效任务能力的预期；它不构成benchmark性能通过，正式成功率必须由H100完整训练后的checkpoint评测。
- 回退本次commit会移除可恢复client/config/audit和server长服务证据逻辑，恢复M4.2单请求入口；不会修改或删除外部checkpoint、数据、实验目录、日志、progress或视频。

## 2026-08-08 — M4.2 X-WAM 单请求闭环集群验收

- 验收实现：commit `b5b6f53da075919cfb7cab4588623dd7d1e5ba76`，分支 `dev/atomic-robocasa365`，server 与 client metadata 均为干净工作区。
- 运行状态：run id `20260808T015103Z` 的 broker、policy server、simulator client 和机器可读证据全部通过；M4.2 正式关闭。

### 集群证据

- A800 policy server 从 M3 `epoch=9-step=10.ckpt` 加载约 10.08 GB 的 DeepSpeed model state；定向严格加载只缺少 438 个保存时主动排除的冻结 T5/VAE 参数，非冻结 missing 为 0、unexpected 为 0。
- Server 使用 `CloseFridge/20250819` 的真实 `stats.json` 和 schema SHA `e95f2b71...f08b4`，模型合同为 16D proprio、12D action、`frame_num=9/action_num=4`，因此单次返回 `[32,12]`。
- Broker 完成一次 request queue、dispatch 和 result forward；server `processed_requests=1`、`failed_requests=0`，固定 inference seed `3928109185`，模型推理耗时 30.566 秒，client round-trip 为 31.593 秒。
- Client 固定 `CloseFridge(target, seed=0, layout=1, style=1)`，一次请求执行首个4-action chunk；episode `1/1` 完成、failed/missing 为 0、total steps 为 4、四步底盘动作均非零，summary 与 episode 均为 `pass`。
- A800 80GB 上 CUDA 峰值 allocated/reserved 为 31.985/32.070 GiB；Torch 2.9.0+cu128、CUDA 12.8、Lightning 2.6.5、PyZMQ 27.1.0。Simulator 使用 RoboCasa 1.0.1、robosuite 1.5.2、MuJoCo 3.3.1、EGL 和独立 `robocasa-abot` 环境。
- 本轮三相机视频为有效 H.264、`768x256`、5 FPS、5帧、66961 bytes；抽帧确认左/右 agentview 与 eye-in-hand 顺序正确且四步运动连续。

### 结论与限制

- `robosuite_models`、GR1 mink 和 mimicgen 缺失警告与 PandaOmron 路径无关，不阻塞本轮。Wan base 构造时打印的新模块初始化提示发生在 M3 checkpoint 严格恢复之前，不表示最终机器人模块未加载。
- `success=false` 符合4步工程门禁预期；该 checkpoint 只经过 M3 单 clip 10-step 工程训练，本结果只证明推理、归一化、网络和完整12D环境动作闭环，不构成900-step成功率或有效策略质量结论。
- 原始 JSON、日志和视频继续只读保存在根级忽略的 `log/` 中，不加入 Git。下一步先完善长 horizon 的运行恢复与证据策略，再进行 `CloseFridge` 900-step 闭环。

## 2026-08-07 — M4.2 X-WAM broker 单请求闭环门禁

- 修改前基线：`a916547`（M4.1 随机闭环实现）
- 运行状态：M4.1 集群 `pass`；M4.2 本地协议、broker 路由和静态测试通过，A800真实 checkpoint 推理为 `cluster-pending`

### M4.1 集群关闭证据

- run id `20260807T152317Z` 在干净 commit `a9165477f0407a1fb1f064ac597a043dce79e5b9` 上完成；固定 CloseFridge scene 执行20步，episode `1/1` 完成、failed/missing 为0，summary 为 `pass`。
- 16D state、完整12D action、20个非零底盘 step、control/gripper 的 `-1/+1` 和 runtime horizon 900 全部匹配配置。
- 三相机 H.264 视频为 `768x256`、5 FPS、6帧、102479 bytes，视觉抽帧正常；随机 success=0 不是工程失败。用户提供的本地 `log/` 目录新增根级忽略规则，本次及后续提交不包含日志或视频。

### 加入的协议和闭环逻辑

- 新增 `xwam.robocasa365.atomic.v1`：request 固定 atomic task/request provenance、三路 `[3,256,256,3] uint8` RGB、16D raw state、prompt 和 cfg；response 必须精确对应请求并返回非空 `[Ta,12]` 动作。
- wire format 使用 `np.savez` 数组和 JSON metadata，解码始终 `allow_pickle=false`，限制单消息大小并拒绝 composite、错误 shape/dtype、NaN/Inf、请求响应 ID 漂移与服务端错误。
- 新增只依赖 PyZMQ 的透明 broker，分离 frontend simulator 和 backend policy 端口，按空闲 server 逐请求派发，并抑制重复 READY。
- 新增 RoboCasa365 policy server：从 M3 实验 `config.yaml` 重建 12D/16D runner，解析 DeepSpeed `last.ckpt`，对排除冻结参数的低内存 checkpoint 复用已验证定向严格加载；首轮关闭 compile/gradient checkpointing。服务端同时冻结 `dataset.task_name`，请求 task 与单任务 checkpoint 不一致时返回显式错误。
- Policy server 从真实单任务 `stats.json` 构造 `PandaOmronTensorCodec`。在线 state 按训练合同归一化；模型输出裁剪训练域后解码完整12D，并只离散 `control_mode`。删除 legacy server 的单臂 padding stats 与 gripper 反转语义。
- RGB preprocessing 与 M3 `augment=false` 对齐：uint8 映射到 `[-1,1]` 后只 resize 到训练 `video_size`，不执行旧 server 硬编码的0.95 center crop。
- 新增 simulator client，继续固定 CloseFridge target/seed/layout/style，设置900秒请求超时；首轮配置最多且至少完成一次请求、预期返回32x12动作、执行前4步，保存 server checkpoint、inference/round-trip latency、动作范围、逐 episode JSON、视频和可重算 summary。即使环境在 reset 时意外报告成功，没有真实模型请求也不能通过 M4.2。
- Gym动作验证新增有限值与 action-space bounds 检查，策略输出越界时不进入 simulator。

### 环境、测试与限制

- Policy 环境清单、constraints 和 requirements 新增 PyZMQ 27.x；安装前先 import，缺少时只补 `pyzmq==27.1.0`，不改 Torch/CUDA/FlashAttention。
- `.gitignore` 新增 `/log/`，与既有 `logs/cluster/` 一起隔离用户回传的JSON、完整日志和视频；本轮只把汇总证据写入中文文档。
- 新增协议/config/checkpoint resolver/单任务隔离/CLI 测试；本地真实回环 broker request/response 路由通过。所有 CLI `--help` 均不需要导入 Torch 或 RoboCasa；server 报告在退出时补写 CUDA allocated/reserved 峰值。
- M4.2 首轮 checkpoint 是单 clip 10-step M3 工程 checkpoint，只用于验证推理 wiring，不代表有效策略训练。4-step结果也不是900-step benchmark成功率。
- 本地没有 Torch/RoboCasa，真实5B构造、低内存 checkpoint 加载、A800显存、模型输出与环境4步执行全部为 `cluster-pending`。

### 回滚

- 回退本次 commit 会移除新版协议、broker/server/client、M4.2配置、PyZMQ policy依赖和测试，并恢复M4.1状态；不会删除或修改服务器 checkpoint、Wan2.2、数据、Conda环境、日志、视频或评测结果。

## 2026-08-07 — M4.1 Atomic 随机闭环评测门禁

- 修改前基线：`f45d57a`（M4.0 simulator 环境门禁关闭）
- 运行状态：本地 dependency-light 测试与语法检查通过；真实 RoboCasa365 20-step rollout 为 `cluster-pending`

### 加入的逻辑

- 在 Atomic-Seen manifest 中冻结 18 个任务的官方 horizon 和 RoboCasa365 源码 commit。评测启动时把 runtime 注册 horizon 与 manifest 对照，版本漂移会立即失败；`CloseFridge` 官方 horizon 为 900。
- 新增 M4.1 版本化配置，固定 `CloseFridge(target, seed=0, layout=1, style=1)`、三路相机及 20-step 工程烟测上限。20 step 不替代官方 900-step 完整评测。
- 新增 dependency-light benchmark adapter：按 PandaOmron schema 把五个具名 observation 分量打包为 16D state；把 schema 顺序的完整 12D flat action 按名称写入五个 Gym action key，不依赖字典顺序、不丢弃底盘或 control 维度。
- 新增不加载 Torch/X-WAM 的随机 rollout CLI。它默认实际采样 `base_motion`，离散化 `control_mode/gripper_close`，验证每一步 state/action/camera 合同，并把三路 `256x256` RGB 横向拼成 `256x768` 视频。
- 每次运行保存 requested/resolved config、Git/环境/runtime metadata、逐 episode JSON、异常 traceback、视频和聚合 summary。summary 从逐 episode 记录重算 success rate，并对 missing/failed rollout 返回 `fail`；随机策略没有完成任务不影响工程链路的 `pass`。
- 新增无 RoboCasa/Torch 单元测试，覆盖 18-task horizon 完整性、固定 smoke 配置、16D/12D 具名映射、三相机顺序、非零底盘随机动作、composite 拒绝及可重算/缺失 episode 聚合。

### 替换的旧逻辑与限制

- 原 `evaluation/robocasa_client.py` 仍基于旧任务表、500-step 上限、手工 padding observation 和旧动作假设，不作为 RoboCasa365 M4 入口；本轮没有删除它，以保留上游复现路径。
- 本轮只关闭随机 simulator 链路，不加载 X-WAM、不经过 broker、不产生 benchmark 指标。正式评测还需要 M4.2 policy adapter、完整 horizon、冻结 seed 集和 checkpoint provenance。
- RoboCasa/robosuite editable 源码仍需在正式 benchmark 前冻结差异；当前 smoke 使用已通过 M4.0 的 `robocasa-abot`，不对第三方源码执行 reset/checkout。

### 回退

- 回退本次 commit 会删除新配置、adapter、随机 CLI 与测试，并恢复 M4.0 文档状态；不会修改服务器 Conda 环境、RoboCasa assets、数据、模型、checkpoint 或既有评测结果。

## 2026-08-07 — M4.0 已有 RoboCasa simulator 环境复用审计

- 分支：`dev/atomic-robocasa365`
- 基线 commit：`0a06beb`
- 运行状态：本地无依赖测试完成；`robocasa-abot` 真实 runtime 为 `reuse_ready`，M4.0 已关闭

### 问题与版本修正

- 用户已有若干 RoboCasa 环境，希望优先复用。仓库旧环境说明仍锁定 RoboCasa `0.2.0`，但该版本是原版 RoboCasa（25 atomic）；RoboCasa365 的 65 atomic、Atomic-Seen 18 和 `gym.make("robocasa/<Task>")` 接口属于 `1.0/1.0.1`。
- 依据官方 `1.0.1` setup/import 契约，将 simulator 基线更正为 RoboCasa `1.0.1`、robosuite `>=1.5.2`、MuJoCo `3.3.1`、NumPy `2.2.5`。旧 `0.2.x` 环境不删除，只分类为 `legacy_robocasa_only`。

### 新增和修改逻辑

- 新增独立 simulator manifest，冻结 Atomic-Seen 18、`CloseFridge(target, seed=0)`、五个 16D state 分量、三路 `256x256 uint8` RGB 和五个 12D Gym 字典动作分量。
- 新增只读环境审计 CLI：先在隔离子进程检查包 import/任务注册，再可选创建 simulator，完成 reset、单步和 EGL render；坏 ABI、MuJoCo 或 OpenGL 崩溃不会终止主审计器，报告始终先原子写入 `logs/cluster/`。
- 报告分类为 `reuse_ready`、`runtime_smoke_required`、`legacy_robocasa_only`、`dependency_blocked`、`registry_blocked` 或 `runtime_blocked`，并记录 Conda/Python、包版本、module 路径和 editable Git provenance。
- 记录 dataset/model 扁平动作与 Gym 字典动作的顺序不同；本轮只验证具名合同，不修改尚未适配的旧 evaluator。M4.1 必须按名称打包，不能把扁平数组直接传给 wrapper。

### 验证、风险与回滚

- dependency-free 测试覆盖 manifest 的 18-task/16D/12D/三相机合同、旧版本拒绝、probe JSON 解析、复用分类和失败日志持久化；Python compile、完整无 Torch 测试和文档门禁在发布前执行。
- 本地没有 RoboCasa365/MuJoCo assets，无法复验真实 runtime；该条初始 `cluster-pending` 状态已由下方集群关闭证据解除。
- 回退本次 commit 可删除 M4.0 审计器并恢复旧文档；不会修改任何服务器 Conda 环境、assets、数据、模型或 checkpoint。

### M4.0 集群关闭证据

- 首轮 registry/runtime 已证明 RoboCasa 1.0.1、robosuite 1.5.2、MuJoCo 3.3.1、NumPy 2.2.5、Atomic-Seen 18、assets 和 EGL 可用；唯一 blocker 是 simulator/policy broker 所需的 PyZMQ。
- 用户在同一 `robocasa-abot` 环境补齐 PyZMQ 27.1.0 后重跑。最终报告为 `ok=true`、`reuse_recommendation=reuse_ready`、`blockers=[]`、`warnings=[]`，registry 和 runtime 子进程返回码均为 0。
- `CloseFridge(target, seed=0)` 成功 reset 和执行单步；online state 合计 16D、Gym 字典动作合计 12D，三路 RGB 与 render 均为 `[256,256,3] uint8`，runtime `errors=[]`。
- `robosuite_models` 和 GR1 mink 缺失只产生与 PandaOmron 无关的 warning；Gym passive checker 的 observation-space warning 未影响真实 state/image/action 合同，M4.1 仍将按具名 schema 自行校验。
- RoboCasa editable 源码仅有 assets 产物未跟踪；robosuite editable 工作区报告大量 tracked 修改。其已验证运行能力可用于 M4 随机 smoke，但正式 benchmark 前必须审计差异并冻结源码 provenance，禁止未经确认执行 reset/checkout。

## 2026-08-07 — M3.2 三个 atomic 任务短训练

- 分支：`dev/atomic-robocasa365`
- 基线 commit：`3bbd621`
- 运行状态：三任务数据、12-step 训练、修正版 audit 和 provenance 全部通过；M3 已关闭

### 方案修正

- 用户确认原 X-WAM 训练应保持 `clean_action_ratio=0.5`，且 M3.1 十步与 M2 单步已经提供 6 个 action/proprio 有效监督更新；因此取消从未在超算运行的 50-step、`clean_action_ratio=0` 单 clip 诊断。
- 删除该诊断专用配置、曲线 parser/audit 和测试。保留 `train/action_proprio_supervision_ratio`，因为它只增加可观测性，不改变训练采样或 loss。
- M3 单任务证据只称为“短程趋势和参数更新 smoke”，不称为严格过拟合或收敛；下一门禁直接进入三个 atomic 任务。

### 新增和修改逻辑

- 新增 atomic-only 三任务 manifest 生成器，默认任务为 `PickPlaceCounterToCabinet`、`OpenCabinet` 和 `TurnOnMicrowave`。工具校验 Atomic-Seen 范围、完整数据/视频合同与唯一日期目录；多个日期目录时拒绝猜测，要求 `--task-path`。
- Dataset 工厂新增 manifest 路径：为每个任务独立构造既有 `RoboCasa365Dataset`，从而保留 task-local PandaOmron normalization；外层 `BalancedRoundRobinDataset` 用 `0→1→2` 虚拟索引布局平衡完整 Dataset，并检查 action/proprio tensor 合同一致，Trainer sampler 可随机重排读取顺序。
- 训练入口将 manifest、任务名、原始/平衡样本数写入 provenance；runner 记录 `train/task_index`。单任务配置和 legacy Dataset 不受影响。
- 新增 12-step FP32/no-checkpoint 配置，保持 `clean_action_ratio=0.5` 和 RGB-only。checkpoint/resume 已由 M3.1 单独通过，本轮隔离多任务数据与训练链路。
- 新增 dependency-free 日志审计，检查 12 个连续 step、三任务覆盖与计数容差、监督比例与 action/proprio loss 对应、两类采样分支、有限 loss、depth=0 和正常退出；不设置固定读取顺序或 loss 降幅阈值。

### 文件、验证、风险和回滚

- 数据边界：`data/robocasa365_multitask.py`、`data/dataset_factory.py`。
- 配置与入口：`configs/data/robocasa365_m3_three_task.yaml`、`configs/experiment/robocasa365_m3_three_task_short.yaml`、`scripts/train_sft.py`、`runners/xwam_runner.py`。
- 审计与测试：`scripts/audit_m3_multitask_dataset.py`、`project_tools/multitask_training.py`、`scripts/audit_m3_multitask_short.py`、`tests/test_m3_multitask.py`、`tests/test_training_run.py`。
- 本地没有 Torch；真实 Parquet/Decord 三任务构造、DeepSpeed FP32 state 内存和 CUDA 训练为 `cluster-pending`。task-local normalization 仅用于 M3 smoke，M6 前必须生成并冻结跨任务正式统计。
- 回退本次 commit 可恢复单任务路径；不会删除或修改服务器数据、权重与 checkpoint。

### 三任务集群数据证据

- commit `3d49c970a00317b3adb466ae8c139d5912946e96` 生成的 manifest 返回 `ok=true/result=pass`、errors 为空，范围为 `atomic_only`，采样为 `balanced_round_robin`。
- `PickPlaceCounterToCabinet`、`OpenCabinet`、`TurnOnMicrowave` 三个日期目录共 322 episodes、75,727 frames；每个任务均满足 16D state、12D action 与三路配置相机合同。
- 数据门禁通过，只关闭 M3.2 的 manifest 子门禁；后续真实训练结论记录在下一小节，不由 manifest 单独推断。

### 12-step 训练反馈与 audit 误判修复

- A800 日志完成 step 0～11、`max_steps=12` 和 run result `pass`；CUDA peak allocated/reserved 为 `30.677/40.076 GiB`。全部指标有限、depth loss 为 0，`clean_action_ratio=0.5` 的两类监督分支及 action/proprio loss 对应关系均通过。
- 实际三个任务计数为 `3/5/4`。旧 audit 要求 task index 严格按 `0,1,2` 循环，但 Lightning/DeepSpeed 会为训练 DataLoader 自动使用随机分布式 sampler；`train_shuffle=false` 只描述构造时 DataLoader，不能保证 Trainer 消费顺序。
- 不修改 Trainer、Dataset 或真实训练 shuffle。audit 改为分别检查 task index 为整数且位于合法范围、三个任务均至少出现一次、12-step 任务计数最大差不超过默认容差 2，并输出具名 `task_counts`。
- CLI 新增 `--max-task-count-spread`，默认 2；非法 task index、任务缺失或超出计数容差仍失败。使用原始日志回归得到计数 `PickPlaceCounterToCabinet=3`、`OpenCabinet=5`、`TurnOnMicrowave=4`，所有检查通过。
- 本次仅修复 dependency-free 审计和记录，不需要重新运行 5B 训练；当时等待的 audit/metadata/result 已由下一节最终证据补齐。

### M3.2 最终验收证据

- run id 为 `20260807T105307Z`；metadata 确认训练 commit `5420c8986836f1ca26fbccc67f5c93db3c263119`、分支正确、工作区干净。修正版 audit 来自 commit `50b11a4`，16 项检查全部为 true，`errors=[]`、`ok=true/result=pass`。
- 环境为单卡 NVIDIA A800 80GB PCIe、Python 3.10.20、Torch 2.9.0+cu128、CUDA 12.8、Lightning 2.6.5、DeepSpeed 0.19.4；拓扑为 world size 1。
- 配置确认 `xwam_pretrained`、ZeRO-2、DeepSpeedCPUAdam FP32 state、CPU offload、RGB-only、`clean_action_ratio=0.5`、12 step 和 checkpoint disabled；三任务 adapter provenance 为 65,423 个原始 clips、102,204 个平衡虚拟 samples。
- result 为 `pass/global_step=12/error=null`，耗时 518.53 秒，CUDA peak allocated/reserved 为 `30.677/40.076 GiB`，没有生成 checkpoint，符合本轮隔离训练链路的设计。
- 进程 VmHWM 为约 108.74 GiB，结束时 cgroup 使用 `119.9987/120 GiB`，只剩约 1.3 MiB；`memory.events.max=4` 但 `oom=0/oom_kill=0`。因此 M3 smoke 通过，但该配置没有主机内存扩展余量，长训练和 H100 正式 profile 必须重新设计资源策略。
- M3 的单 batch、checkpoint/resume、三任务短训练与审计全部关闭；下一阶段为 M4 atomic 闭环评测器，不把本轮 smoke 解释为正式训练或 benchmark 指标。

## 2026-08-07 — M3.2 FP32 单 clip 过拟合曲线门禁

> 状态：已被上方三任务方案取代，未在超算运行；相关专用文件已删除，不应执行本节旧命令。

- 分支：`dev/atomic-robocasa365`
- 基线 commit：`fabaaba`
- 运行状态：M3.1 resume-to-10 通过；本条 M3.2 方案在集群运行前取消

### M3.1 集群关闭证据

- 原 step-8 checkpoint 在修复后成功恢复：定向 loader 只放行 438 个冻结 T5/VAE 参数，非冻结 missing 与 unexpected 均为 0；DeepSpeed 随后报告全部状态恢复，generator state 也明确应用。
- step 8～9 连续执行，Trainer 正常达到 `max_steps=10`；global step 10 的 checkpoint start/complete 均存在，run result 为 `pass`。
- CUDA peak allocated/reserved 为 `30.677/40.076 GiB`，保存期 RSS 约 97.3 GB。M3.1 的低内存保存与 resume 工程门禁通过，但 BF16 optimizer state 不能作为正式数值结论。

### M3.2 新增和修改逻辑

- 新增独立 50-step 单 clip experiment 配置，使用原 `a800_80gb_debug` FP32 CPUAdam profile，并显式关闭 checkpoint；训练阶段曾在 120 GiB 主机通过，已知风险集中在 FP32 state 保存峰值，因此本轮隔离数值收敛与保存问题。
- 诊断配置将 `clean_action_ratio` 临时设为 0，让 batch size 1 的每个 step 都计算 action/proprio loss；该修改只用于过拟合门禁，M3.3 和正式训练恢复模型层默认 0.5。
- Runner 新增 `train/action_proprio_supervision_ratio`，使 loss 为 0 时可以区分设计采样、无效数据 mask 和恢复错误。
- 新增 dependency-free 日志 parser 与 CLI audit：验证 50 个连续 step、必需指标、有限值、监督比例、RGB-only depth=0、Trainer/result 正常退出，并计算前后各 10 step 的均值和相对下降。
- M3.2 通过阈值固定为 video/action/proprio/total 四项后窗均值相对前窗至少下降 10%；audit 失败时不自动放宽。

### 涉及文件、兼容性和回滚

- 配置：`configs/experiment/robocasa365_close_fridge_m3_overfit_curve.yaml`。
- 训练日志：`runners/xwam_runner.py`。
- 审计与测试：`project_tools/training_curve.py`、`scripts/audit_m3_overfit_curve.py`、`tests/test_training_curve.py`、`tests/test_training_run.py`。
- 文档：`docs/ARCHITECTURE.md`、`docs/PROGRESS.md`、`docs/CLUSTER_RUNBOOK.md`。
- 不修改 Dataset、normalization、12D action、16D proprio、checkpoint loader、模型结构或正式默认采样分布；原 M3.1 checkpoint 保留。
- 本地没有 Torch；真实 FP32 CPUAdam 50-step 未运行，相关专用文件已由后续三任务 commit 删除；外部数据、权重与 checkpoint 不受影响。

## 2026-08-07 — M3.1 排除冻结参数 checkpoint 的定向恢复

- 分支：`dev/atomic-robocasa365`
- 基线 commit：`54d1f01`
- 运行状态：本地静态验证完成；复用原 step-8 checkpoint 的 resume-to-10 为 `cluster-pending`

### 问题与集群证据

- 120 GiB profile 已在固定 RGB clip 上完成 step 0～7、step-8 checkpoint 保存和正常退出；`checkpoint_save_start/complete`、result `pass/global_step=8` 均存在，说明上一轮保存期内存修复有效。
- 首次 resume 能解析 `epoch=7-step=8.ckpt`，但在 DeepSpeed `load_module_state_dict` 阶段失败；optimizer、scheduler 和后续训练 step 尚未恢复，CUDA peak 仍为 `30.677/40.076 GiB`，不是本轮 OOM。
- 报错缺失项全部是冻结的 `text_encoder.model.*` 与 `vae.model.*`。保存侧按 `exclude_frozen_parameters=true` 主动省略这些参数，而 Lightning/DeepSpeed 恢复侧默认传入 `strict=true`，两者合同不一致。
- 日志还显示 `on_fit_start` 的 generator 种子初始化早于 checkpoint module load；旧 `on_load_checkpoint` 只缓存状态，不能保证它在 generator 已存在时真正应用。

### 新增和修改逻辑

- 增加无 Torch 的 resume key validator：只允许缺失当前模型中 `requires_grad=false` 的参数；缺少可训练参数或 buffer、出现 unexpected key 时立即失败并给出计数和样例。
- Runner 仅在训练入口同时检测到 `resume_checkpoint` 与 `exclude_frozen_parameters=true` 时启用定向非严格底层加载；其他初始化、普通 resume、M2、upstream 和正式 profile 保持原严格语义。
- 在底层加载前后各验证一次 key 集合；tensor shape mismatch 继续由 PyTorch `load_state_dict` 阻塞，不会被放宽。
- 成功恢复时输出 `missing_frozen/missing_non_frozen/unexpected` 摘要，并把完整允许清单写入 result JSON 的 `resume_module_load`，metadata 同时记录该模式是否启用。
- generator 恢复改为顺序无关：checkpoint hook 先到时在 `on_fit_start` 创建后应用，fit hook 先到时由 checkpoint hook 立即覆盖种子状态。

### 涉及文件

- 恢复合同：`project_tools/training_run.py`、`runners/xwam_runner.py`、`scripts/train_sft.py`。
- 回归测试：`tests/test_training_run.py`。
- 架构、进度和超算验收：`docs/ARCHITECTURE.md`、`docs/PROGRESS.md`、`docs/CLUSTER_RUNBOOK.md`。

### 验证、风险和回滚

- 聚焦无 Torch 测试覆盖冻结缺失通过、可训练缺失拒绝、unexpected key 拒绝和入口 wiring；Python compile、完整无 Torch 测试、文档门禁和 diff 检查在发布前执行。
- 本地没有 Torch/Lightning/DeepSpeed，真实 5B module、optimizer、scheduler、loop 和 generator 完整恢复仍为 `cluster-pending`。
- 原 step-8 checkpoint 不需要重建；它依赖 Wan2.2 路径重新构造冻结 T5/VAE，再由训练 checkpoint 恢复 X-WAM 可训练状态。
- 回退本次 commit 即恢复 DeepSpeed 默认严格加载；不会删除或改写外部 checkpoint、数据和模型权重。

## 2026-08-06 — M3.1 120 GiB checkpoint 低内存门禁

- 分支：`dev/atomic-robocasa365`
- 基线 commit：`cfe86cb`
- 运行状态：本地静态验证完成；A800 step 8 保存与 resume-to-10 为 `cluster-pending`

### 问题与集群证据

- 原 M3.1 运行在固定 RGB clip 上完成 step 0～7，并由 Lightning 正常报告 `max_steps=8 reached`；没有 NaN、CUDA OOM 或 Python traceback。
- step 8 恰好触发完整 DeepSpeed checkpoint，日志间隔由约 30 秒增至约 180 秒；之后没有进入训练入口的 `finally`，缺少 CUDA peak、result JSON 和正常退出，tmux/Pod 同时失效。
- 模型包含 5.0B trainable 与 6.4B frozen 参数，CPUAdam 使用 FP32 master/momentum/variance；星光当前主机内存只有 120 GiB。结论为 checkpoint/teardown 阶段失败，cgroup OOM 概率最高但仍需事件证据确认。

### 新增和修改逻辑

- 新增独立 `a800_80gb_120g_debug` hardware profile：保持 ZeRO-2 CPU offload，但将 CPUAdam optimizer state 设为非 FP32，并在 checkpoint 保存时排除冻结 T5/VAE。
- 原 `a800_80gb_debug`、M2 smoke 和 upstream 配置显式保持 FP32 optimizer state；代码默认也是 FP32，防止正式训练静默继承低内存调试语义。
- CPUAdam state 精度改为配置驱动并在日志、resolved config 和 run metadata 中记录；非布尔配置提前拒绝。
- DeepSpeedStrategy 接入 `exclude_frozen_parameters`，默认 false，只有 120 GiB profile 显式启用。
- checkpoint callback 在实际保存前后 fsync 写入独立 JSONL，记录 step、目标路径、进程 VmRSS/VmHWM、cgroup memory current/peak/max/events；普通异常额外写入 error 事件。
- run metadata/result 增加 optimizer backend/state precision、checkpoint event 路径和内存快照。

### 兼容性、风险与正式训练恢复

- 本次不修改 Dataset、RGB、depth、12D action、16D proprio、loss 或公开 checkpoint 映射。
- BF16 optimizer state 只用于 120 GiB 的 M3 保存/恢复工程门禁，可能改变优化器数值；该结果不能代表正式训练配置。
- H100 短程门禁和正式训练必须显式使用 `deepspeed_fp32_optimizer_states=true`，并重新验证 checkpoint/resume。
- 排除冻结参数依赖 Lightning 2.6.5/DeepSpeed 0.19.4 的真实保存与严格恢复行为，本地无 Torch，保持 `cluster-pending`。
- 回退本次 commit 可恢复原 M3 profile；不会删除当前不完整实验目录或外部 checkpoint。

### 本地验证

- CPUAdam 精度默认/覆盖、DeepSpeed 冻结参数选项和非法类型测试通过。
- cgroup v2/RSS 解析及 fsync JSONL 测试通过。
- 聚焦无 Torch 测试、Python compile、文档同步和 Git diff 检查通过。

## 2026-08-06 — M3.1 分层配置、极小子集与 checkpoint resume 门禁

- 分支：`dev/atomic-robocasa365`
- 基线 commit：`c593036`
- 运行状态：本地静态验证完成；A800 8-step + resume-to-10 为 `cluster-pending`

### 目标与问题

- M2 配置把模型、硬件和一步实验参数放在同一文件中，不能作为 M3/H100 正式配置基础。
- 训练入口没有固定极小子集、显式 resume path、每次调用的 Git/config/result 产物；`num_training_steps` 同时承担 scheduler horizon 和本次停止位置，也无法对分段恢复保持同一学习率计划。
- X-WAM 使用自建 CPU generator 采样 diffusion noise/timestep，Lightning 不会自动理解这个对象的恢复语义；不保存它会让单样本 resume 的随机序列重新从 seed 开始。

### 新增和修改逻辑

- 增加 RoboCasa365 atomic 模型层、A800 80GB 单卡 debug 硬件层和 `CloseFridge` M3 极小样本实验层；训练入口按 model/data/hardware/experiment/CLI 顺序合并。
- `train_subset_size=1`、`train_subset_start=0` 和 `train_shuffle=false` 固定唯一 clip；边界越界或空数据在构造 5B 模型前失败。
- 分离 `num_training_steps=10` 的 scheduler horizon 与首轮 `trainer_max_steps=8`；resume 使用同一 horizon 并把 invocation limit 提升到 10。
- 增加 `resume_checkpoint`，存在性检查通过后传入 `Trainer.fit(ckpt_path=...)`；恢复运行跳过公开 X-WAM checkpoint adapter，完整训练状态由 DeepSpeed checkpoint 接管。
- 单 GPU M3 配置保存/恢复自定义 generator state；resume 缺少该字段或 world size 大于 1 时明确拒绝，避免随机序列静默重置或多卡错误复用 rank 0 RNG。
- checkpoint callback 支持配置 `save_top_k/save_last/save_on_exception`，M3 首轮只在 step 8 保留一个 checkpoint，并以本地 `last.ckpt` symlink 指向它，避免重复复制 5B optimizer state。
- 每次调用生成独立 resolved config、metadata JSON 和 result JSON，自动记录 Git、环境、配置/数据/checkpoint 来源、命令、子集、global step、耗时、进程 max RSS、CUDA peak 与 checkpoint 路径。
- validation 为 0 时不再重复构造完整验证 Dataset；既有配置没有设置新字段时保持原 scheduler、全数据 shuffle、初始化和 Torch AdamW/DeepSpeed 行为。

### 验证、风险和回滚

- 新增 dependency-free schedule/subset/resume/provenance 测试和静态 wiring 测试；本地无 Torch，真实 OmegaConf/Lightning/DeepSpeed checkpoint 保存恢复均为 `cluster-pending`。
- DeepSpeed ZeRO checkpoint 是目录而非普通单文件，完整 optimizer checkpoint 可能占用数十 GiB；两段运行会保留 step 8/10 两个恢复点，运行前必须检查实验盘至少约 200 GiB 可用空间。
- 10 个 diffusion step 的单点 loss 可能因随机 timestep/noise 波动，M3.1 先验收有限值、总体趋势、step 连续和恢复完整性，不要求每一步严格单调下降。
- 回退本次 commit 可移除 M3.1 能力；不会删除外部实验目录。正式 H100 profile 尚未创建，不能把 A800 CPU offload 配置作为正式训练结论。

## 2026-08-06 — M2 单步训练集群验收通过

- 分支：`dev/atomic-robocasa365`
- 测试逻辑：包含 commit `2c107b2` 新增的 `Optimizer backend: deepspeed_cpu_adam`；反馈未单独附 `git rev-parse HEAD`
- 运行状态：`pass`；M2 关闭，下一阶段进入 M3 RGB-only 训练烟测

### 星光验收证据

- 真实数据为 `CloseFridge/20250819`，配置固定 RGB-only、12D action、16D proprio、batch size 1、单 GPU 和一步训练。
- X-WAM checkpoint adapter 报告 `mode=xwam_pretrained, result=pass`；模型为 5.0B trainable、6.4B non-trainable、11.4B total。
- DeepSpeed 正确解析 ZeRO-2 CPU optimizer offload、1e8 allgather/reduce bucket 和关闭 overlap；optimizer backend 为 `deepspeed_cpu_adam`。
- 训练输出 video loss `0.191945`、action loss `1.254097`、proprio loss `1.533597`、depth loss `0.000`、总 loss `2.979639`，分项求和与总 loss 一致。
- `Trainer.fit` 以 `max_steps=1 reached` 正常停止且没有 traceback，证明真实 batch 的 forward、backward 和第一次 optimizer update 均已完成。
- CUDA peak allocated `30.677 GiB`、reserved `40.076 GiB`；相比 GPU AdamW 首次更新失败时的 allocated `77.645 GiB`，单卡 debug 显存策略已验证有效。

### 边界、兼容性和后续

- BF16 model summary 估算、0-worker、冻结模块 eval mode、TF32 deprecated 和 val interval 提示均未阻塞训练；0 worker 是 M2 smoke 的显式设置，T5/VAE 为预期冻结模块。
- 本轮没有显式 `train_exit_code`、`free -h` 或单独的 `git rev-parse HEAD`；正常 `max_steps` 退出和最终 CUDA peak 已足够关闭单步计算门禁，但这些字段在 M3 性能/复现记录中必须补齐。
- 本次只记录集群证据，不修改模型、数据、optimizer 或配置逻辑。正式默认配置仍为 offload false 和 Torch AdamW；H100 多卡策略在 M3/M6 单独冻结。

## 2026-08-06 — M2 offload optimizer 选择 DeepSpeedCPUAdam

- 分支：`dev/atomic-robocasa365`
- 失败配置：与 commit `4bdb272` 的 M2 smoke 一致；反馈未附 `git rev-parse HEAD`
- 运行状态：CPU offload 参数解析通过；DeepSpeed 初始化拒绝 client-provided Torch AdamW，修复后参数更新为 `cluster-pending`

### 问题与诊断

- ZeRO-2 CPU offload 已正确解析，但 runner 的 `configure_optimizers()` 固定返回 `torch.optim.AdamW`。
- DeepSpeed 默认要求 ZeRO-Offload 使用 `DeepSpeedCPUAdam`，因此在 `strategy.setup/deepspeed.initialize` 阶段主动抛出 `ZeRORuntimeException`；本轮尚未读取 batch，也未执行 forward/backward。
- 退出前 CUDA peak 21.294 GiB 只代表初始化阶段，不能作为完整训练显存结论；NCCL cleanup warning 是异常退出的伴随结果。

### 新增和修改逻辑

- 新增无依赖 optimizer backend policy：只有显式设置 `deepspeed_offload_optimizer=true` 时选择 `deepspeed_cpu_adam`，其他运行一律选择 `torch_adamw`。
- CPU offload 分支使用 `DeepSpeedCPUAdam`，显式保持 AdamW mode、FP32 optimizer state、lr、betas、eps 和 weight decay；scheduler 逻辑不变。
- optimizer backend 在启动日志中打印，便于区分单卡调试和正式训练来源。
- M2 单卡 smoke 继续开启 CPU offload；正式默认配置保持 offload false，因此仍使用 Torch AdamW 和 GPU optimizer，不改变正式实验语义。

### 验证、风险和回滚

- 本地没有 Torch；backend policy、静态 wiring、Python compile、完整无 Torch 测试、变更记录门禁和 diff 检查通过后发布，真实 CPUAdam 扩展加载与参数更新为 `cluster-pending`。
- 首次使用 DeepSpeedCPUAdam 可能触发本地扩展编译；若失败，应记录 CPUAdam builder/compiler 日志，不绕过 `zero_force_ds_cpu_optimizer` 保护。
- 将 M2 的 `deepspeed_offload_optimizer` 设为 false 会回到 Torch AdamW，但单卡 80GB 会重现 optimizer state OOM；正式训练是否需要 offload 将在 H100 多卡 profile 中单独确定。

## 2026-08-06 — M2 optimizer-step OOM 与 ZeRO-2 CPU offload

- 分支：`dev/atomic-robocasa365`
- 运行状态：真实 batch forward/backward 通过；首次 optimizer step OOM，CPU offload 复测为 `cluster-pending`

### 问题与诊断

- A800 80GB 在 forward/backward 后达到 allocated 77.645 GiB、reserved 77.748 GiB；AdamW 初始化 `exp_avg_sq` 时还需 18.77 GiB，因只剩约 239 MiB 而退出。
- reserved-but-unallocated 仅约 103 MiB，排除 allocator 碎片为主因；这是约 37.5 GiB 两组 FP32 Adam moment 的真实容量问题。
- 模型构造、checkpoint 迁移、数据读取、12D/16D 契约和反向计算均已越过，不需要修改 RoboCasa365 数据或启用 depth。

### 新增和修改逻辑

- 训练入口把 DeepSpeed stage、optimizer offload、offload device、pin memory、communication overlap 和 bucket size 改为显式配置并在启动时打印解析结果。
- M2 单卡 smoke 使用 ZeRO-2 CPU optimizer offload；bucket 从 5e8 降为 1e8，并关闭 overlap，避免通信 buffer 进一步挤占显存。
- upstream legacy 配置显式保留原 ZeRO-2 GPU optimizer、5e8 bucket 和 overlap 行为，避免改变既有训练实验。
- 增加无 Torch 配置解析和入口 wiring 测试；runbook 要求运行前确认主机可用内存，建议至少 64 GiB。

### 验证、风险和回滚

- 本地没有 Torch；解析、静态 wiring、Python compile、完整无 Torch 单元测试和 diff 检查通过后发布，真实 DeepSpeed offload 参数更新为 `cluster-pending`。
- CPU offload 预计使用约 37.5 GiB optimizer moment 内存，并会降低单步速度；主机内存不足时禁止启动，优先改为多 GPU ZeRO 或后续评估 ZeRO-3，而不是修改数据语义。
- 将 `deepspeed_offload_optimizer` 设回 `false` 并恢复 bucket/overlap 可回退；这会重现单卡 80GB optimizer-step OOM。

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
