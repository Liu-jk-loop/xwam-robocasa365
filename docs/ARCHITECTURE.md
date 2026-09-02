# Architecture decisions

## Adapter boundaries

RoboCasa365 support will be added behind five explicit boundaries:

1. `DatasetAdapter`: native episode, video, metadata, instruction, and task access.
2. `ObservationAdapter`: camera selection, image transforms, state packing, and modality validation.
3. `ActionCodec`: named action components, normalization, model representation, and environment packing.
4. `CheckpointAdapter`: controlled loading between Wan2.2, legacy X-WAM, and the adapted model.
5. `BenchmarkAdapter`: task registry, simulator lifecycle, rollout records, and metric aggregation.

The X-WAM backbone should consume validated tensors and remain free of dataset-path and simulator-specific conditionals.

## Data contract

- Native RoboCasa365 LeRobot/Parquet is the source of truth.
- The adapter must read official metadata rather than infer unnamed dimensions from array length.
- The M1 contract gate expects PandaOmron `observation.state` to be 16D, `action` to be 12D, and the three official RGB camera keys to be present. M2 will freeze the component names and slices from the real `modality.json` before model mapping.
- Atomic-only selection is explicit and auditable.
- Cameras are selected by configured keys and stable ordering.
- RGB is required for the initial path.
- Depth has two legal modes: `disabled` and `cached`. Online simulator rendering in a data-loader worker is forbidden.
- Normalization statistics are generated for an immutable dataset/task manifest and stored with provenance.

### M1 native loader boundary

- `data/robocasa365_index.py` owns dependency-free LeRobot v2.1 episode indexing, prompt parsing, path templates, and temporal windows.
- `data/robocasa365_dataset.py` owns runtime Parquet/MP4 decoding and emits the existing X-WAM batch keys without converting the dataset to legacy JSON.
- The current M1 tensor contract is RGB `[V,T,C,H,W]`, raw state `[T,16]`, raw action `[Ta,12]`, and explicit validity/camera masks. Camera order is configured and never shuffled.
- Raw state/action are intentionally restricted to the batch-audit path. `configs/data/robocasa365.yaml` remains `training_ready: false` until M2 freezes named PandaOmron slices and normalization.
- Only LeRobot v2.x episode-per-file data is accepted by the native loader. A v3 dataset must use a separate indexed shard adapter rather than silently assuming v2 paths.

## Action contract

- Never preserve the legacy evaluator behavior that copies only the first seven predicted values.
- Represent every PandaOmron action component with a name, slice, dtype, range, and environment mapping.
- Read the initial schema from official RoboCasa365 metadata and freeze the validated result in a versioned manifest.
- Keep base motion and control mode available even for manipulation-heavy atomic tasks.
- Validate round-trips using recorded dataset actions before closed-loop policy evaluation.

### PandaOmron versioned schema

- `configs/schemas/robocasa365_panda_omron_v1.json` is the expected 16D/12D contract. Runtime use is legal only after the real task `meta/modality.json` matches every named slice.
- State is `base_position[0:3] + base_rotation[3:7] + end_effector_position_relative[7:10] + end_effector_rotation_relative[10:14] + gripper_qpos[14:16]`.
- Action is `base_motion[0:4] + control_mode[4:5] + end_effector_position[5:8] + end_effector_rotation[8:11] + gripper_close[11:12]`. No action dimension may be discarded.
- Position-like components use dataset `q01/q99`; unit quaternion, controller-range, control-mode and gripper action components preserve their native `[-1,1]` representation. Quantile normalization reports clipping rate and supports an unclipped audit round-trip.
- Model output keeps `control_mode` continuous during denoising, then discretizes it to `-1/+1` only when packing an environment action.
- Legacy X-WAM action semantics are dual-arm 14D. The declared migration copies its left-arm 7D boundary weights into PandaOmron arm slice `[5:12]`, initializes new base/control slice `[0:5]`, and reinitializes proprio boundary layers because the two 16D vectors have different meanings.

### M4 simulator boundary

- RoboCasa365 `1.0.1` 的 Gym wrapper 使用具名字典动作：`end_effector_position(3) + end_effector_rotation(3) + gripper_close(1) + base_motion(4) + control_mode(1)`，总维度仍为 12。
- 数据集/模型的扁平 12D 顺序是 `base_motion + control_mode + end_effector_position + end_effector_rotation + gripper_close`；它与 Gym 字典的展示顺序不同。M4 `BenchmarkAdapter` 必须按名称打包，禁止依赖字典迭代顺序或把扁平数组直接传入 wrapper。
- 在线 observation 使用 Gym wrapper 的五个具名 state 分量和三路 `video.*` RGB key；M4 adapter 在进入 policy 前按版本化 16D schema和相机顺序打包。
- M4.0 环境审计通过独立子进程 import 和创建 simulator，默认 `MUJOCO_GL=egl`，避免坏 ABI 或渲染器崩溃终止主审计进程。版本/注册通过但未执行 runtime smoke 的环境仍需完成最终门禁。

#### M4.1 随机闭环门禁

- `configs/tasks/robocasa365_atomic_seen.json` 固定 Atomic-Seen 18 的官方 horizon，并记录其 RoboCasa365 源码 commit。评测启动时必须再次读取当前 runtime 注册表；runtime horizon 与 manifest 不一致时立即失败，禁止静默采用本地版本。
- 首个门禁固定为 `CloseFridge(target, seed=0, layout=1, style=1)`。官方 horizon 为 900；配置中的 `max_steps=20` 只是快速检查 reset/step/视频/落盘的工程烟测上限，不是完整 benchmark episode，也不能生成可报告的成功率。
- 随机策略先在 schema 顺序中生成完整 12D 动作，再按名称写入五个 Gym action key。`base_motion` 默认实际采样，`control_mode` 和 `gripper_close` 离散为 `-1/+1`；所有 12 维都必须被消费，避免 manipulation task 掩盖底盘维度被截断的问题。
- 每一步都按版本化 schema 检查并打包 16D state；三路 RGB 按固定顺序横向拼接为 `256x768` 视频。该门禁不 import Torch、不加载 Wan2.2/X-WAM，也不经过 broker。
- 每次运行保存 requested/resolved config、运行环境与 Git provenance、逐 episode JSON、三相机视频和聚合 summary。聚合器从逐 episode 记录重算 success rate，并将缺失 rollout、异常退出或不完整记录判为 `fail`。
- 工程门禁的 `result=pass` 表示请求的 episode 完整执行且证据齐全，与随机策略是否完成任务分开；随机策略 `success=false` 是正常结果。正式成功率只能由后续完整 horizon、冻结 seed 集与 X-WAM checkpoint 的评测产生。

#### M4.2 X-WAM broker 闭环门禁

- M4.2 使用独立的 `xwam.robocasa365.atomic.v1` 协议，不复用 legacy RoboCasa client 的旧任务名、手工 16D padding、7D/14D 动作或 gripper 反转。旧入口保留作 upstream 参考，但不能用于 RoboCasa365 指标。
- Simulator request 固定包含 atomic scope、task/episode/step/request ID、语言指令、三路 `[3,256,256,3] uint8` RGB 和一个有限 16D raw state；policy response 必须对应同一个 request，并返回非空 `[Ta,12]` raw environment action。协议通过 NPZ 数组加 JSON metadata 传输，解码时 `allow_pickle=false`。
- Policy server 独占 Torch/CUDA/Wan2.2/X-WAM。它从 M3 `config.yaml` 恢复模型结构和单任务名称，从 DeepSpeed `model_states.pt` 严格加载 trainable 权重；只有配置声明 `exclude_frozen_parameters=true` 时，才复用 M3 已验证的定向规则，允许缺少由 Wan2.2 重建的冻结 T5/VAE 参数。请求 task 必须与 checkpoint 的 `dataset.task_name` 完全一致，不能拿单任务权重静默评测其他 Atomic 任务。
- 在线 raw 16D state 必须使用训练任务的真实 `stats.json` 与版本化 schema 编码；模型 12D 输出先裁剪到训练域 `[-1,1]`，再由同一个 `PandaOmronTensorCodec` 解码，且只对 `control_mode` 做符号离散。禁止沿用 legacy 单臂统计量或反转 gripper。
- RGB 输入执行与 M3 `augment=false` 数据相同的 `uint8 → [-1,1] → video_size` 变换，不增加旧 server 的硬编码 0.95 center crop。首轮关闭 `torch.compile` 与 gradient checkpointing，优先验证可解释加载和单次推理。
- Broker 只透明转发 opaque wire bytes，不加载模型、模拟器或 schema；每个空闲 server 一次接收一个请求。Client 设置显式请求超时，server 将单请求异常编码为 error response，避免两侧无限等待。
- 首个 M4.2 配置仍固定 `CloseFridge(target, seed=0, layout=1, style=1)`，只执行一个 4-action receding-horizon chunk，并要求至少完成一次 policy request。它验证一次真实 X-WAM inference 和四次环境 step，不代表 900-step 完整 rollout 或任务成功率。

#### M4.3 900-step 可恢复闭环

- M4.3 不修改 M3 训练 runner。长运行 client 在每次 policy response 后先原子写入待执行的最多4个12D动作，再在每个 simulator step 后原子更新 `progress.json`；进程在任意两次写入之间退出时，恢复点都对应一个完整环境状态。
- 恢复不依赖不可移植的 MuJoCo 内存快照。Client 使用相同 task/seed/layout/style 重建环境，按顺序回放已经持久化的完整12D动作，并逐步比较16D state；任一步最大绝对误差超过配置容差或 success/terminated/truncated 标志漂移都会阻止续跑。
- 未执行完的最后一个 action chunk 保存在 progress 中，恢复后直接继续执行，不重复请求模型。新的 request ID 包含 run id；恢复进程要求原 Git commit、配置、checkpoint 与已有请求完全一致。
- 视频不再依赖易损坏的长时间流式 writer。初始帧、每个配置 stride 和 terminal 帧分别原子保存为 PNG，episode 完成后再编码 MP4；中断恢复时只补缺失帧。
- Policy server 对每个成功或失败请求 fsync 追加 JSONL。连续服务可在 client 完成后由 `Ctrl-C` 正常退出，并在满足最小请求数且无失败时写出 `pass`；client 进度与 server journal 由独立审计器交叉核对。
- Client中断时可能已有一个请求在server执行；该response未进入progress，恢复后会以相同request ID和确定性seed重试。审计按request ID去重并允许多条一致的成功server记录，但任何关联失败、shape/checkpoint/seed漂移仍阻塞验收；环境动作只按progress执行一次。

#### M6 Atomic-Seen 18 并行评测

- Clariden单节点固定4张GH200，每卡加载两个独立X-WAM进程，共8个policy server；每个server使用独立frontend/backend broker端口，两个指定simulator client不能被动态路由到其他server。模型和模拟器继续分别运行于`xwam.toml`和`robocasa365.toml`，不得合并Python依赖树。Simulator EDF通过`PYTHONPATH`优先使用Store中已下载完整assets的`src/robocasa`和匹配`src/robosuite`，禁止静默回退到SQSH内不完整的`/opt/robocasa`。
- 版本化topology必须恰好包含8个server、16个client、每GPU两个server、每server两个client，并不重不漏覆盖Atomic-Seen 18。`client6/client7`各串行两个任务，其余client各一个任务；server6/server7因此各覆盖三个任务。
- M6 policy server从正式实验`config.yaml`与DeepSpeed model state恢复模型，从不可变M6 manifest获取训练任务集合，并使用与manifest绑定的跨任务16D/12D statistics。每个server再按topology限制允许任务，禁止把单任务M4统计或未训练任务用于正式评测。
- Atomic9 ratio0.5对照复用ratio0的不可变manifest/global stats和8500步scheduler以固定数据与学习率轨迹，但使用独立preflight、实验名、W&B run和checkpoint根。两组不得交叉恢复；外层训练目标在7500停止，使比较区间集中在`clean_action_ratio=0.5`带来的控制监督密度。
- 默认每任务50个episode，环境seed为`42+episode_index`；模型推理seed固定42。场景使用FastWAM一致的`target` split且不固定layout/style。每次请求执行ANS action denoise 10步并在动作可用后early-stop，video scheduler仍为50步；client每20个环境动作replan一次，不改变模型32步action horizon，每个episode最多1000个环境step。
- M6正式client不再嵌套M4.3 runner或逐episode Git门禁。每完成一个episode便原子更新该任务`result.json`；同一Git commit/checkpoint/eval ID重提时跳过连续完成seed，中断episode从头重跑。最终聚合必须收齐18个task result和配置声明的全部episode，缺失记录不能被当作0%成功率静默吞掉。
- 每个环境step把三路原始RGB拼成一帧并直接流式编码20 FPS MP4，跑满1000步的失败episode约50秒；不持久化PNG帧。正式评测目录固定为IOPS的`x-wam-eval/atomic18/<eval ID>`，仅组织为`logs/`、`results/<Task>/videos`、task result及根目录JSON/CSV汇总；policy server正式模式不写逐request journal，也不把全部成功request嵌入状态JSON。Capstor/Store不承载推理结果。
- 在加载八份policy模型前，独立simulator probe必须确认RoboCasa/robosuite实际import路径位于上述Store根目录，并检查已知正式场景资产`Sink025/model.xml`非空；两份源码commit与asset hash进入eval不可变合同。当前RoboCasa EDF经FastWAM验证只枚举单一EGL设备，因此16个simulator client均固定EGL device 0；这不改变policy server的0/1/2/3 GPU映射。

## Model initialization

Two modes remain supported:

- `xwam_pretrained`: load Wan2.2 base components, construct X-WAM modules, then load the public 40k cross-embodiment X-WAM checkpoint through a shape-aware adapter.
- `wan_base`: load Wan2.2 base components and initialize/copy new X-WAM modules using code, without loading the cross-embodiment checkpoint.

Checkpoint loading must emit a machine-readable report of loaded, remapped, newly initialized, missing, and unexpected parameters. Shape mismatches must never be silently ignored.

## Process boundary

The policy server owns Torch, CUDA, X-WAM, and model weights. The benchmark client owns RoboCasa, robosuite, MuJoCo, task creation, and rendering. ZeroMQ remains the boundary so each side can use a compatible environment and can be scaled independently.

## Configuration

Configuration will be layered rather than encoded in source:

```text
model defaults
  + dataset/schema profile
  + hardware profile
  + experiment profile
  + command-line overrides
```

Absolute cluster paths are allowed in cluster-local overrides but not as Python defaults. Every saved experiment must contain the resolved configuration, Git commit, dataset manifest ID, checkpoint source, and environment summary.

### M3 training-run contract

- 配置合并顺序固定为 model → data/schema → hardware → experiment → CLI override；后层只能覆盖前层，不在 Python 中写机器路径。
- `num_training_steps` 表示学习率计划总步数，`trainer_max_steps` 表示本次调用停止位置。初始运行和 resume 必须使用相同的学习率计划总步数。
- 极小样本门禁通过 `train_subset_size/train_subset_start` 选择固定 clip，并显式关闭 shuffle；它只用于验证过拟合和恢复，不能充当正式数据抽样策略。
- Lightning/DeepSpeed 完整恢复统一走 `Trainer.fit(ckpt_path=...)`。恢复时不再重复加载公开 X-WAM checkpoint，模型、optimizer、scheduler、global step 和 loop state 由训练 checkpoint 接管。
- M3 单卡确定性门禁把 X-WAM 自定义 CPU generator state 写入 checkpoint 并在 resume 时恢复。M6 H100 profile 将四个 rank 的 generator state 一并写入 DeepSpeed checkpoint，并禁止改变 world size 后恢复；真实多卡 hook 行为由 2→4 step 门禁验证。
- 每次调用写出独立的 resolved config、run metadata 和 run result JSON。metadata 包含 Git commit/dirty state、命令、配置来源、环境版本、数据/manifest/schema、子集、拓扑、DeepSpeed 和 checkpoint 来源；result 包含 pass/fail、global step、耗时、进程 max RSS、CUDA 峰值和 checkpoint 路径。
- A800 debug profile 可以使用已验证的 ZeRO-2 CPUAdam offload。正式 H100 profile 不继承该决定，必须依据 GPU 数量、显存和吞吐单独冻结。
- `a800_80gb_120g_debug` 是独立的低内存工程门禁：CPUAdam momentum/variance 随 BF16 参数保存，并在 DeepSpeed checkpoint 中排除冻结 T5/VAE。它只验证连续训练和完整恢复 wiring，不作为正式优化器数值配置。
- 低内存 checkpoint 恢复不能全局关闭严格加载。只有同时满足“本次为 resume”及“保存配置排除冻结参数”时，runner 才允许 state dict 缺少当前模型中 `requires_grad=false` 的参数；缺少任一可训练参数、buffer 或出现额外 key 仍立即失败，shape mismatch 继续由 PyTorch 阻塞。
- 单卡 generator state 的恢复必须兼容 Lightning/DeepSpeed 的两种 hook 顺序：`on_load_checkpoint` 先运行时延迟到 generator 创建后应用，`on_fit_start` 先运行时则在 checkpoint hook 中立即覆盖种子初始化状态。
- M3.2 使用三个 Atomic-Seen 任务的显式 manifest。每个子任务由独立 `RoboCasa365Dataset` 保持 task-local normalization，外层 `BalancedRoundRobinDataset` 用 `0→1→2` 虚拟索引布局保证完整数据集中的任务样本总量相等，并记录 task provenance；Lightning/DeepSpeed 的训练 sampler 可以随机重排这些索引，不能把实际 step 顺序解释为固定轮询。这只用于短程工程烟测，M6 正式训练前仍需冻结跨任务统计与正式采样策略。
- M3.2 保持原训练语义 `clean_action_ratio=0.5`，同时记录 action/proprio 监督比例和 task index。12-step FP32/no-checkpoint 门禁要求三个任务均被采到、task index 合法且任务计数最大差不超过 2，并验证两类采样分支、有限 loss 与 RGB-only depth=0；不设置人为 loss 降幅阈值，也不把短烟测解释为收敛。
- CPUAdam 默认和原有 A800/M2/upstream profile 均保持 `fp32_optimizer_states=true`。H100 正式门禁必须显式使用 FP32 optimizer state 并重新验证 checkpoint/resume，禁止从 120 GiB profile 隐式继承 BF16 state。
- 每次 DeepSpeed checkpoint 保存前后向独立 JSONL fsync 写入 process RSS、cgroup memory current/peak/max/events。缺少 `checkpoint_save_complete` 时，结合 `oom_kill` 和 checkpoint 文件结构区分保存期 OOM 与普通 Python 异常。

### Clariden 4×GH200 部署恢复门禁

- 该门禁只验证 Clariden 容器的单节点四卡通信、ZeRO-2 FP32 CPUAdam、分布式 generator state、DeepSpeed checkpoint 和严格恢复，不复用 H100 正式训练的18任务、GBS 128或无 offload 合同。
- 固定 CloseFridge 前8个 clip、micro-batch 1、GBS 4和四步 scheduler。第一次运行到step 2并保存，第二次必须以同一个完整checkpoint恢复到step 4；两阶段使用不同run id但相同干净Git commit、数据路径和scheduler horizon。
- Slurm allocation内每阶段只由`srun`启动一个EDF容器任务，四个本地rank继续由Lightning的`devices=4` launcher派生；进入训练前移除step级`SLURM_NTASKS=1`拓扑变量，避免Lightning把容器任务数误判为训练world size。
- Debug checkpoint 可以排除冻结 T5/VAE以控制空间，但恢复报告只能接受这些冻结参数缺失，任何可训练参数缺失或额外参数仍阻塞。四个rank都必须报告实际FP32 optimizer state，checkpoint必须包含非空model state和四个ZeRO optimizer shard。
- dependency-light审计同时检查四张GH200、有限RGB-only loss、step 2/4 checkpoint完成事件、恢复源一致性及`global_step=4`。通过只关闭部署恢复门禁，不代表M6正式训练配置、吞吐或模型质量。
- 两阶段编排不依赖Clariden宿主机Python：checkpoint result解析和最终联合审计都在EDF内执行。若initial阶段已保存并通过，可用其Slurm Job ID重建实验/run路径并只执行恢复阶段；复用仍会重新验证initial result与checkpoint目录，不会跳过最终联合审计。由于编排修复会改变Git commit，复用模式只在Git diff严格局限于冻结的编排、审计、测试和文档路径时接受commit差异；任何训练runtime、模型、数据或实验配置变化都会阻塞。

### Clariden M6 数据冻结门禁

- Clariden正式profile冻结前，独立EDF作业只扫描`pretrain/atomic`中的Atomic-Seen 18同名任务，生成不可变manifest、与manifest digest绑定的16D state/12D action跨任务统计，以及GBS 128、5 epoch的精确step计划。
- manifest、global stats和preflight统一写入Store的`manifests/xwam/m6`，临时memmap只写Capstor scratch；三份机器产物必须同时为`ok=true/result=pass`，且生成时Git worktree必须干净。
- 该门禁不加载模型、不解码视频、不开始训练。它关闭后才能依据真实总clip数和GH200资源冻结正式hardware profile及step 2→4正式配置门禁。

### M6 正式 accelerator 合同

- M6数据/sampler/scheduler/GBS/FP32 optimizer/full-checkpoint合同与GPU型号解耦；配置必须显式声明`formal_accelerator`、`formal_zero_stage`、`formal_world_size`、节点数、每节点设备数和最低显存。GH200与保留的H100正式profile都严格固定ZeRO-1；Clariden每节点要求4张至少90 GiB的GH200，4卡profile固定1节点/4 rank，8卡profile固定2节点/8 rank。
- Clariden的GBS128候选梯度为默认`4×micro-batch 16×accumulation 2`、balanced `4×8×4`和safe `4×4×8`。三者均使用BF16 model compute、ZeRO-1 GPU AdamW、实际FP32 optimizer state、通信overlap和完整checkpoint，不继承debug的CPU offload或冻结参数排除。默认值来自同为3相机/9视频帧的FastWAM Clariden设置，但X-WAM的更长文本上下文和实现差异仍必须由真实step 2→4门禁验证。
- 4卡正式profile先以完整18任务manifest/global stats和16,390-step scheduler运行到step 2，保存完整model/四rank optimizer shard，再从精确checkpoint恢复到step 4。8卡扩展实验按用户决定从公开pretrained重新开始，不转换4卡ZeRO checkpoint；其后只允许恢复包含rank 0～7八个optimizer shard的本实验checkpoint。
- Clariden正式5-epoch训练使用固定实验目录和绝对global-step目标，trainer始终指向step 16,390，由12小时Slurm上限决定何时重提。完整DeepSpeed checkpoint采用双层存储：IOPS的`xwam_run/<实验名>/checkpoints`每500步保存并滚动保留最近5个；Store的`checkpoints/xwam/<实验名>/checkpoints`每3,000步永久保存且不限数量，step 16,390的final也写入该Store目录。文件锁禁止并发写入；planner按该实验声明的world size跨两层选择最新完整model+全部rank optimizer checkpoint，同step优先Store副本，未完整目录在各自文件系统内原子隔离。
- 正式Clariden分段训练的W&B身份属于完整实验而非单个Slurm Job：固定实验目录原子持久化一个run ID，所有chunk以online模式和`resume=allow`写入同一曲线。每个训练调用仍保留独立本地run ID和metadata用于checkpoint/audit provenance；机器审计要求metadata中的W&B ID等于持久化ID。认证只接受提交环境中的`WANDB_API_KEY`且metadata仅记录`auth=api_key_env`，不读取默认账号作为回退，也不把key写入配置、源码和日志。

### M6 H100 RGB-only 正式训练合同

- 训练任务固定为版本化 Atomic-Seen 清单中的18个同名任务，但数据来源固定为 `pretrain/atomic`；manifest 必须逐任务解析唯一日期目录、检查真实 Parquet/三路视频，并按 `sum(max(episode_length-32, 0))` 记录有效 clip。任何缺失、重复、多日期歧义或 composite 路径都会阻塞。
- 正式采样使用 `natural_proportional`，不再沿用 M3 的三任务等量过采样。18任务共用一份与 manifest SHA-256 绑定的跨任务 q01/q99/min/max；每个任务仍单独验证16D state、12D action与 modality schema。
- `EpochAlignedDistributedSampler` 每个 epoch 先按 `seed+epoch` 全局打乱，只丢弃不足一个 GBS=128 的尾部，再等长切分给4个rank。`steps_per_epoch=floor(total_valid_clips/128)`，总步数固定为 `5*steps_per_epoch`，从而让配置中的5 epoch与实际 optimizer update一致。
- 首选 H100 profile 为 `4 GPU × micro-batch 4 × accumulation 8 = GBS 128`；显存回退仅改为 `4×2×16`，不改变全局 batch、epoch、LR计划或数据清单。正式配置强制单节点4张至少75 GiB且名称为H100的GPU。
- 模型前向保留 `bf16-mixed`。A800 120 GiB工程让步不进入正式训练：正式profile使用ZeRO-1、optimizer不offload、AdamW state要求实际为FP32、通信overlap开启、checkpoint不排除冻结参数。每个rank在首个update后单独写出实际 optimizer-state dtype 审计。
- H100门禁使用完整正式scheduler和数据，仅把本次上限设为step 2，保存后从同一checkpoint恢复到step 4；机器审计联合验证四rank FP32 state、H100拓扑、有限loss、RGB-only depth loss为0、保存完成及resume来源。门禁通过后正式训练必须从公开 X-WAM pretrained权重新建实验，不能接着门禁checkpoint训练。
- 公共 metadata/result/checkpoint事件只由rank 0写入；四个rank共享父进程生成的run ID。正式结束额外保存一个明确的 `final-step=*.ckpt`，审计要求global step等于自动计算的5-epoch总步数。

### CloseFridge clean-action单任务A/B合同

- A/B都从同一个公开X-WAM cross-embodiment checkpoint重新初始化，不允许从18任务正式checkpoint或另一组A/B checkpoint开始。两组使用相同CloseFridge日期目录、task-local `meta/stats.json`、RGB增强、seed42、单节点4×GH200、GBS128、ZeRO-1、FP32 optimizer state、完整gradient checkpointing、学习率和3000-step scheduler；唯一实验变量是`clean_action_ratio=0.5`或`0.0`。
- 首轮两组都训练到global step 1000并每500步向IOPS滚动保存，最多保留最近2个；step 1000额外把完整final checkpoint写入Store。A/B使用90分钟分布式timeout，并在每次DeepSpeed保存后执行全rank barrier，避免慢存储造成某个rank提前进入下一次collective。比较闭环成功率后，胜出组可以保持原scheduler、W&B run ID、world size和optimizer state恢复到step 3000，不能只加载model weights创建新的优化器轨迹。
- 单任务A/B不使用M6 18任务manifest、跨任务statistics或五epochformal guard。它验证CloseFridge机械臂/夹爪动作学习以及clean-action采样语义，不能单独证明NavigateKitchen底盘动作已经恢复。
- 两组闭环比较复用M6已对齐FastWAM的target split和时间轴，但每次只启动一个policy server与一个CloseFridge client。每组固定模型seed42、环境seed42起50 episodes、1000-step horizon、replan20、action denoise10及20 FPS逐步视频。评测源码使用独立clean clone，训练作业所用仓库在作业结束前保持不变。

### RGB-D离线回放合同

- RGB-D首轮是inverse-depth辅助预测监督，不把depth传感器帧加入policy条件。模型输入仍为首帧RGB、16D proprio和语言；depth branch只作为未来空间监督并可在action-only推理时关闭。
- 所有RoboCasa365 atomic任务共用同一depth生成逻辑。dataset-level `env_args`决定环境类，episode-level MJCF/metadata决定场景，`states.npz["states"]`决定逐帧物理状态；任务代码不参与depth编码分支选择。
- 原始LeRobot目录保持只读。MuJoCo只允许出现在独立离线生成/审计进程，训练loader只读取版本化缓存。cache索引必须绑定task/episode/frame/camera、源文件digest和depth encoding版本。
- MuJoCo展平state宽度是per-episode合同，不是dataset/task常量。必须先用当前episode的`ep_meta.json + model.xml.gz`硬重置环境，再要求它与该episode的`states.shape[1]`一致并逐帧恢复；禁止用其他episode已实例化的模型解释state。
- P1 render probe对每个抽查帧直接从同一simulator state渲染三路RGB和depth buffer。MuJoCo bottom-up RGB/depth均先纵向翻转；RGB必须与原LeRobot MP4同帧比较，normalized depth用robosuite模型near/far参数转换为metric depth并保存。
- X-WAM公开合同使用inverse depth、三通道pseudo-RGB和`[-1,1]` VAE输入，但公开代码没有给出MuJoCo depth到8-bit的完整公式。结构门禁与render门禁分开；没有RGB像素/时间对齐和数值映射证据时禁止生成全量缓存。
- 本项目训练缓存固定为`robocasa365_inverse_metric_global_q_v1`：在代表性任务/episode/相机上合并采样`1/depth_m`，用全局q01/q99冻结一个跨任务、跨相机、跨帧范围，再裁剪映射到uint8，近处更亮；无效或非正depth映射为0。该规则兼容X-WAM公开存储/loader合同，但不声称复现上游未公开的数值公式。
- P2缓存是256×256、20 FPS、H.264/yuv420p的三通道重复灰度MP4。每个视频sidecar绑定encoding digest、episode三件回放源文件digest、源RGB digest、帧数、相机key、数值统计、压缩往返误差和文件digest；只有sidecar与视频均通过才可恢复跳过，原始LeRobot数据始终只读。
- RGB-D loader只有在`use_depth=true`且同时提供cache root、冻结encoding和PASS manifest时启用；三者缺一、task/episode/camera集合漂移、sidecar与manifest不一致或任一文件缺失都会在创建Dataset时阻塞。`use_depth=false`时反向禁止携带depth路径，保持RGB实验不触碰缓存。
- loader按RGB完全相同的episode/frame ID解码三路depth，输出`depths[V,T,3,H,W]`，uint8只执行`pixel/127.5-1`且resize使用nearest；RGB继续使用bilinear。随机空间裁剪由同一个augmentation同时作用于RGB/depth，从而保持像素对齐。
- P3工程门禁先生成完整CloseFridge缓存，再用4×GH200固定8个clip执行step 0→2保存和step 2→4恢复。审计必须同时看到有效非零depth loss、RGB/action/proprio有限loss、四rank FP32 optimizer state、完整checkpoint和精确resume来源；该门禁不是正式RGB-D训练。

### PointMap几何流合同

- 现有RGB-D路径是inverse-depth未来辅助监督，depth不是policy输入；把缓存键重命名为PointMap不会自动得到Flex-π式输入流。PointMap先以辅助目标接入，之后才单独增加可选输入、stream dropout和cross-modality forcing。
- PointMap数值合同固定为`robocasa365_camera_xyz_flexpi_v1`：在原生metric depth网格用每路OpenCV内参生成camera-space XYZ；`0.01m < Z < 2m`为有效值，无效XYZ为零，不使用外参。
- XYZ按`x/y=[-0.5,0.5]m、z=[0,1.5]m`裁剪并映射至`[-1,1]`，再以nearest从`256×256`变为`256×320`。该数值定义绑定Flex-π官方实现commit `20c1b2b71ea35a415d5d47c39b04443cfadad7a1`。
- 正式PointMap缓存直接保存未压缩normalized float16 `.npy [T,3,256,320]`，以memory-map按clip读取。训练worker不解码depth、不计算XYZ、不读取相机内参；metric depth只保留在P0少量审计产物中。
- 透明物体的depth来源不能由普通depth→XYZ重投影检查决定。CloseBlenderLid P0.1作业`3259701`在目标像素检测到超过1 mm的depth变化，因此正式策略冻结为`use_forced_opaque_depth_for_pointmap_keep_original_rgb`：只在几何渲染上下文临时把可见半透明geom/material alpha提升为1，退出时恢复；LeRobot RGB不重渲染、不重写。P1入口必须校验该P0.1 JSON的PASS状态、结论和SHA256，并把证据写入缓存合同。
- P0必须在逐episode MJCF/state恢复后，从robosuite simulator读取三路相机内参。报告同时审计RGB对齐、内参稳定性、depth/像素重投影、裁剪比例、无效值、目标shape和float16反量化误差；未通过时禁止全量生成约283 GiB的Atomic9缓存。
- P1缓存以episode/相机为恢复粒度。只有`.npy`和sidecar同时存在，且源MJCF/state/meta/RGB SHA256、PointMap合同摘要、shape/dtype、数组SHA256及逐帧数值审计完全一致时才跳过；其他情况只重建该对。最终manifest与独立audit再次枚举全部episode/相机，训练loader不得把`.partial.npy`或未审计文件视为数据。
- P2训练loader在进程启动时验证P1 manifest/audit绑定、全部sidecar和`.npy` header，但不重复全量数组SHA256；每个worker用read-only mmap和有限LRU按9帧clip读取，避免8个rank各自把完整缓存装入内存。RGB与PointMap使用同一个逐相机时空一致crop；连续XYZ和RGB均bilinear resize，颜色增强只作用RGB。
- `PointMap-Aux`沿用X-WAM已有第二生成模态的VAE/DiT路径，但通过`use_pointmap`和独立loss键与inverse-depth隔离；二者配置互斥。该阶段PointMap只是训练目标，policy server始终以`run_depth=false`加载，闭环推理不读取PointMap缓存、不启动MuJoCo渲染。
- P3先在4×GH200固定8 clip执行step 0→2保存及2→4恢复；只有正`train/pointmap_loss`、有限RGB/action/proprio loss、四rank FP32 optimizer state和完整checkpoint同时PASS，2节点8卡正式入口才被授权。

## Current external paths

These paths are cluster deployment facts, not portable defaults:

```text
Wan2.2 base:
/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/Wan-AI/Wan2.2-TI2V-5B

X-WAM pretrained root:
/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/x-wam/xwam_checkpoints/pretrained/checkpoints/last.ckpt
```
