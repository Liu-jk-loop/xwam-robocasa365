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
- M3 单卡确定性门禁把 X-WAM 自定义 CPU generator state 写入 checkpoint 并在 resume 时恢复。该能力显式限制为 world size 1；多卡 RNG 恢复在正式 H100 profile 中另行设计。
- 每次调用写出独立的 resolved config、run metadata 和 run result JSON。metadata 包含 Git commit/dirty state、命令、配置来源、环境版本、数据/manifest/schema、子集、拓扑、DeepSpeed 和 checkpoint 来源；result 包含 pass/fail、global step、耗时、进程 max RSS、CUDA 峰值和 checkpoint 路径。
- A800 debug profile 可以使用已验证的 ZeRO-2 CPUAdam offload。正式 H100 profile 不继承该决定，必须依据 GPU 数量、显存和吞吐单独冻结。
- `a800_80gb_120g_debug` 是独立的低内存工程门禁：CPUAdam momentum/variance 随 BF16 参数保存，并在 DeepSpeed checkpoint 中排除冻结 T5/VAE。它只验证连续训练和完整恢复 wiring，不作为正式优化器数值配置。
- 低内存 checkpoint 恢复不能全局关闭严格加载。只有同时满足“本次为 resume”及“保存配置排除冻结参数”时，runner 才允许 state dict 缺少当前模型中 `requires_grad=false` 的参数；缺少任一可训练参数、buffer 或出现额外 key 仍立即失败，shape mismatch 继续由 PyTorch 阻塞。
- 单卡 generator state 的恢复必须兼容 Lightning/DeepSpeed 的两种 hook 顺序：`on_load_checkpoint` 先运行时延迟到 generator 创建后应用，`on_fit_start` 先运行时则在 checkpoint hook 中立即覆盖种子初始化状态。
- M3.2 使用三个 Atomic-Seen 任务的显式 manifest。每个子任务由独立 `RoboCasa365Dataset` 保持 task-local normalization，外层 `BalancedRoundRobinDataset` 用 `0→1→2` 虚拟索引布局保证完整数据集中的任务样本总量相等，并记录 task provenance；Lightning/DeepSpeed 的训练 sampler 可以随机重排这些索引，不能把实际 step 顺序解释为固定轮询。这只用于短程工程烟测，M6 正式训练前仍需冻结跨任务统计与正式采样策略。
- M3.2 保持原训练语义 `clean_action_ratio=0.5`，同时记录 action/proprio 监督比例和 task index。12-step FP32/no-checkpoint 门禁要求三个任务均被采到、task index 合法且任务计数最大差不超过 2，并验证两类采样分支、有限 loss 与 RGB-only depth=0；不设置人为 loss 降幅阈值，也不把短烟测解释为收敛。
- CPUAdam 默认和原有 A800/M2/upstream profile 均保持 `fp32_optimizer_states=true`。H100 正式门禁必须显式使用 FP32 optimizer state 并重新验证 checkpoint/resume，禁止从 120 GiB profile 隐式继承 BF16 state。
- 每次 DeepSpeed checkpoint 保存前后向独立 JSONL fsync 写入 process RSS、cgroup memory current/peak/max/events。缺少 `checkpoint_save_complete` 时，结合 `oom_kill` 和 checkpoint 文件结构区分保存期 OOM 与普通 Python 异常。

## Current external paths

These paths are cluster deployment facts, not portable defaults:

```text
Wan2.2 base:
/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/Wan-AI/Wan2.2-TI2V-5B

X-WAM pretrained root:
/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/x-wam/xwam_checkpoints/pretrained/checkpoints/last.ckpt
```
