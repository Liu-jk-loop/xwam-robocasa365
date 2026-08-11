# 星光超算操作手册

## 拉取仓库

开发分支发布后执行：

```bash
git clone --branch dev/atomic-robocasa365 --single-branch \
  https://github.com/Liu-jk-loop/xwam-robocasa365.git

cd xwam-robocasa365
git submodule update --init --recursive
git rev-parse HEAD
git status --short --branch
```

每次反馈都必须包含 `git rev-parse HEAD` 的输出。

`git submodule update --init --recursive` 会下载主仓库锁定版本的第三方代码。本项目包含 RoboCasa、robosuite 和 RoboTwin。通常首次 clone 后执行一次；以后只有主仓库引用的子模块 commit 发生变化时才需要再次执行。M1 metadata audit 不依赖这些子模块，可以暂缓到模拟器或评测环境配置阶段。

## M1 元数据审计

拉取 M1 第一批修改后，将占位路径替换为真实目录并执行：

```bash
python scripts/audit_robocasa365_dataset.py \
  --dataset /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/robocasa/robocasa/datasets/v1.0/pretrain/atomic/CloseFridge/20250819 \
  --task-name CloseFridge \
  --require-data \
  --require-videos \
  --output /tmp/robocasa365_close_fridge_audit.json
```

该命令不会导入 Torch，也不会解码视频。它负责检查官方元数据、16 维 state、12 维 action、三路 RGB 相机键、atomic-only 任务归属，以及 Parquet/MP4 文件是否存在。

执行完成后反馈以下内容：

1. `git rev-parse HEAD` 输出。
2. 完整 audit 命令和返回码。
3. `/tmp/robocasa365_close_fridge_audit.json` 内容。
4. 实际数据路径及 `meta/`、`data/`、`videos/` 的两级目录结构。

任务目录通常包含日期层，例如 `CloseFridge/20250819`。audit 的 `--dataset` 必须指向该日期目录或其中的 `lerobot/`，不能只指向 `pretrain/atomic/` 总目录。

## M1 原生 RGB-only batch 验收

本批新增 PyArrow 16.1.0，用于直接读取官方 LeRobot Parquet。已有独立环境不需要重建；拉取代码后在该环境内重新执行受约束安装器：

```bash
conda activate xwam-robocasa365
export TORCH_EXTENSIONS_DIR=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/.cache/torch_extensions/xwam-robocasa365

cd /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/xwam-robocasa365
git pull --ff-only origin dev/atomic-robocasa365
git rev-parse HEAD

bash scripts/install_starlight_dependencies.sh dry-run
bash scripts/install_starlight_dependencies.sh apply
```

该操作沿用现有约束，不替换 Torch 2.9.0、CUDA、FlashAttention 或 DeepSpeed。随后读取一个真实 `CloseFridge` clip 和 DataLoader batch：

```bash
python scripts/audit_robocasa365_batch.py \
  --dataset /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/robocasa/robocasa/datasets/v1.0/pretrain/atomic/CloseFridge/20250819 \
  --task-name CloseFridge \
  --index 0 \
  --num-workers 0 \
  --log-file logs/cluster/robocasa365_close_fridge_batch.json
```

预期核心证据：

- `video`: `[3, 9, 3, 256, 320]`、`float32`、范围在 `[-1,1]`。
- `proprios`: `[9,16]`；`actions`: `[32,12]`。
- `frame_ids`: `[0,4,8,12,16,20,24,28,32]`；`action_ids`: `[0..31]`。
- `camera_type_mask`: `[0,0,1]`；batch 不含 `depths`。
- 关闭 augmentation 后，同一索引重复读取完全一致。
- 最终 `result=pass`、`ok=true`，完整报告持久化在项目 `logs/cluster/`。

本轮不要运行 `scripts/train_sft.py`。M1 只验证原始 16D state 和 12D action 的读取、时序与维度；`configs/data/robocasa365.yaml` 以 `training_ready: false` 明确阻止在 M2 normalization/action schema 完成前训练。

反馈以下内容：commit SHA、完整命令和返回码、`logs/cluster/robocasa365_close_fridge_batch.json`。如果失败，保留完整 `error` 和 `traceback`，不要只截取最后一行。

## M2 PandaOmron schema 与 checkpoint 契约审计

M1 已在 commit `e4249b9` 通过。拉取 M2 第一批后，在 policy 环境执行：

```bash
conda activate xwam-robocasa365

cd /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/xwam-robocasa365
git pull --ff-only origin dev/atomic-robocasa365
git rev-parse HEAD

python scripts/audit_robocasa365_m2_contract.py \
  --dataset /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/robocasa/robocasa/datasets/v1.0/pretrain/atomic/CloseFridge/20250819 \
  --task-name CloseFridge \
  --checkpoint /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/x-wam/xwam_checkpoints \
  --episode-index 0 \
  --log-file logs/cluster/robocasa365_close_fridge_m2_contract.json
```

该命令完成四项只读检查：

1. 将真实 `meta/modality.json` 的每个名称、切片和 original key 与版本化 schema 对比。
2. 读取 `meta/stats.json` 的 state/action q01、q99、min、max，并报告退化 quantile 维度。
3. 对真实 episode 0 执行 state/action 未裁剪 round-trip，报告 component range、control-mode 值和训练 clip 比例。
4. 使用 FakeTensorMode 和 mmap 只读取公开 DeepSpeed checkpoint 元数据，报告 action/proprio encoder/decoder 边界 shape，不构造 5B 模型、不占用 GPU 权重显存。

预期所有 `checks` 为 `true`，尤其是：

- state/action 分别为 16D/12D。
- `control_mode` 只包含 `-1/+1`。
- unclipped round-trip 最大误差小于 `1e-5`。
- checkpoint action 边界为 legacy 14D，proprio 边界为 16D。
- 最终 `result=pass`、`ok=true`。

如果 `weights_only` 或 `mmap` 读取 checkpoint 失败，不要改为普通全量 `torch.load`；把完整报告反馈回来，避免不必要的 CPU 内存峰值。M2 contract 通过前仍不要启动训练。

## M2 12D checkpoint 加载与单步反传

M2 contract 已在 commit `95808cd` 通过。拉取 M2 第二批 commit 后，先只构造模型并验证公开 X-WAM 权重适配，不要直接启动训练：

```bash
conda activate xwam-robocasa365

cd /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/xwam-robocasa365
git pull --ff-only origin dev/atomic-robocasa365
git rev-parse HEAD

python scripts/audit_xwam_checkpoint_loading.py \
  --mode xwam_pretrained \
  --wan-checkpoint-dir /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/Wan-AI/Wan2.2-TI2V-5B \
  --xwam-checkpoint /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/x-wam/xwam_checkpoints \
  --output logs/cluster/xwam_pretrained_panda_omron_loading.json
```

预期报告应为 `result=pass`、`ok=true`，并满足：

- `action_remap` 恰好包含 action encoder 输入 weight、decoder 输出 weight/bias 三项。
- `reinitialize` 恰好包含 proprio encoder 输入和 decoder 输出的 weight/bias 四项。
- `missing_source`、`unexpected_source`、`shape_errors` 和 `errors` 均为空。
- RGB-only 目标不存在的 depth `extra_blocks/extra_heads` 只能进入 `discard_source`，不能被当作普通 missing 静默跳过。

然后验证不使用公开 X-WAM cross-embodiment 权重的 `wan_base` 消融初始化：

```bash
python scripts/audit_xwam_checkpoint_loading.py \
  --mode wan_base \
  --wan-checkpoint-dir /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/Wan-AI/Wan2.2-TI2V-5B \
  --output logs/cluster/wan_base_panda_omron_loading.json
```

两份加载报告都通过后，再执行一次真实 `CloseFridge` batch 的完整训练 step。该 M2 配置只运行一步、batch size 1、0 worker、RGB-only、gradient checkpointing，并关闭大 checkpoint 保存。

M2 smoke 还显式设置 `enable_tensorboard=false`：只保留控制台/`tee` 日志，不要求环境安装可选的 `tensorboard` 或 `tensorboardX`。正式训练配置仍默认启用 TensorBoard。

第一次 A800 运行已经完成 forward/backward，但 AdamW 在首次创建约 37.5 GiB 两组 FP32 moment state 时 OOM。新版 smoke 因此使用 ZeRO-2 CPU optimizer offload、`DeepSpeedCPUAdam`、1e8 communication bucket 并关闭 overlap；只改变优化器/通信内存位置，不改变数据、12D action、16D proprio、模型输入或 loss。正式默认配置保持 offload false 和 `torch.optim.AdamW`。运行前确认主机可用内存，建议 `available` 至少 64 GiB；不足时不要启动：

```bash
free -h

mkdir -p logs/cluster
set -o pipefail

python scripts/train_sft.py \
  model_config=configs/model/wan22_5b_robocasa365_atomic_m2.yaml \
  wan_checkpoint_dir=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/Wan-AI/Wan2.2-TI2V-5B \
  pretrained_checkpoint=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/x-wam/xwam_checkpoints \
  dataset.dataset_path=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/robocasa/robocasa/datasets/v1.0/pretrain/atomic/CloseFridge/20250819 \
  exp_root=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/experiments/xwam-robocasa365 \
  exp_name=close_fridge_m2_single_step \
  2>&1 | tee logs/cluster/close_fridge_m2_single_step.log

echo "train_exit_code=${PIPESTATUS[0]}"
```

日志中必须出现 `offload_optimizer: True`、`offload_optimizer_device: cpu`、`allgather_bucket_size: 100000000` 和 `Optimizer backend: deepspeed_cpu_adam`。首次运行可能编译 CPUAdam 扩展，需要等待其完成；不要设置 `zero_force_ds_cpu_optimizer=false` 绕过保护。反馈三份 JSON/配置、完整终端日志、返回码、`free -h`、CPUAdam 编译信息、GPU 峰值和首个 loss。若第一条加载 audit 失败，不要继续训练，也不要改为 `strict=False` 或手动删除报错参数。

## M3.1 固定单 clip 训练与 checkpoint resume

第一次运行已经在原 FP32 CPUAdam profile 下完成 step 0～7，但在 step 8 checkpoint/teardown 阶段失去 Pod；当前只有 120 GiB 主机内存。复测改用独立的 `a800_80gb_120g_debug` profile：CPUAdam momentum/variance 使用 BF16，并排除冻结 T5/VAE checkpoint。它只验证 `CloseFridge` 一个固定 clip 的连续训练和恢复 wiring，不代表正式训练数值配置；H100 必须恢复 FP32 optimizer state 并重新验收。

保留原来的 `close_fridge_m3_overfit_gate1` 目录做事后分析，不要删除或覆盖。新门禁使用 fresh experiment；实验盘建议至少 250 GiB 可用。先记录节点内存、cgroup 限额/事件和磁盘：

```bash
conda activate xwam-robocasa365
cd /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/xwam-robocasa365
git pull --ff-only origin dev/atomic-robocasa365
git rev-parse HEAD
git status --short

free -h
mkdir -p /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/experiments/xwam-robocasa365
df -h /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/experiments/xwam-robocasa365
mkdir -p logs/cluster
set -o pipefail

for file in \
  /sys/fs/cgroup/memory.current \
  /sys/fs/cgroup/memory.peak \
  /sys/fs/cgroup/memory.max \
  /sys/fs/cgroup/memory.events \
  /sys/fs/cgroup/cpu.max
do
  if test -r "$file"; then
    printf '%s: ' "$file"
    cat "$file"
  fi
done
```

若要限制 CPUAdam 的 CPU 抢占，只把 `OMP_NUM_THREADS`/`MKL_NUM_THREADS` 设置为 Pod 实际分配的 CPU 数；不要猜测核数。第一段从公开 X-WAM checkpoint 初始化，在固定全局 clip 0 上训练到 step 8，并保存低内存完整训练 checkpoint：

```bash
python scripts/train_sft.py \
  model_config=configs/model/wan22_5b_robocasa365_atomic.yaml \
  hardware_config=configs/hardware/a800_80gb_120g_debug.yaml \
  experiment_config=configs/experiment/robocasa365_close_fridge_m3_overfit.yaml \
  wan_checkpoint_dir=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/Wan-AI/Wan2.2-TI2V-5B \
  pretrained_checkpoint=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/x-wam/xwam_checkpoints \
  dataset.dataset_path=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/robocasa/robocasa/datasets/v1.0/pretrain/atomic/CloseFridge/20250819 \
  exp_root=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/experiments/xwam-robocasa365 \
  exp_name=close_fridge_m3_overfit_120g_gate1 \
  2>&1 | tee logs/cluster/close_fridge_m3_overfit_120g_initial.log

echo "initial_exit_code=${PIPESTATUS[0]}"
```

第一段必须同时满足：

- `Training data selection` 显示 base 23496、selected 1、indices `(0,)`、shuffle false。
- `DeepSpeed options` 包含 `exclude_frozen_parameters: True`，并出现 `CPUAdam options: fp32_optimizer_states=False`。
- 输出 step 0～7 的有限 video/action/proprio/total loss，depth loss 始终为 0；允许随机 diffusion loss 波动，不要求逐步严格单调。
- `Trainer.fit` 正常达到 `max_steps=8`，run result JSON 为 `pass/global_step=8`。
- checkpoint event JSONL 同时包含 `checkpoint_save_start` 和 `checkpoint_save_complete`；若只有 start，立即停止并反馈 JSONL、`memory.events` 和 Pod 退出原因。
- checkpoint 目录存在，run metadata 中 Git commit 与上方 `git rev-parse HEAD` 相同且 `dirty=false`。

确认第一段返回码为 0 后再检查 checkpoint；如果失败或磁盘空间不足，不要启动恢复段：

```bash
XWAM_M3_ROOT=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/experiments/xwam-robocasa365/close_fridge_m3_overfit_120g_gate1
XWAM_M3_CKPT=$XWAM_M3_ROOT/checkpoints/last.ckpt

test -e "$XWAM_M3_CKPT"
du -shL "$XWAM_M3_CKPT"
find "$XWAM_M3_ROOT/runs" -maxdepth 1 -type f -print | sort
find "$XWAM_M3_ROOT/runs" -maxdepth 1 -name '*_checkpoint_events.jsonl' -exec cat {} \;
```

第二段从 `last.ckpt` 恢复到同一 scheduler horizon 的 step 10，并在 step 10 更新 checkpoint：

```bash
python scripts/train_sft.py \
  model_config=configs/model/wan22_5b_robocasa365_atomic.yaml \
  hardware_config=configs/hardware/a800_80gb_120g_debug.yaml \
  experiment_config=configs/experiment/robocasa365_close_fridge_m3_overfit.yaml \
  wan_checkpoint_dir=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/Wan-AI/Wan2.2-TI2V-5B \
  dataset.dataset_path=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/robocasa/robocasa/datasets/v1.0/pretrain/atomic/CloseFridge/20250819 \
  exp_root=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/experiments/xwam-robocasa365 \
  exp_name=close_fridge_m3_overfit_120g_gate1 \
  trainer_max_steps=10 \
  save_interval=10 \
  resume_checkpoint="$XWAM_M3_CKPT" \
  2>&1 | tee logs/cluster/close_fridge_m3_overfit_120g_resume.log

echo "resume_exit_code=${PIPESTATUS[0]}"
```

第二段必须出现：

- `deferred_to_trainer`。
- `Excluded-frozen resume load accepted`，且明确为 `missing_non_frozen=0, unexpected=0`；如果报告可训练参数、buffer、额外 key 或 shape mismatch，立即停止。
- `Restored training generator state`；早先出现一次 seed 初始化日志不代表失败，checkpoint hook 随后必须明确覆盖它。
- step 8～9、`max_steps=10`、第二组 start/complete event 和 result `pass/global_step=10`。
- result JSON 的 `resume_module_load.mode=excluded_frozen_parameters`，并且 `missing_frozen_count` 为正数。

反馈 resume 日志、这次新增的 config/metadata/result/event 文件、`du -shL last.ckpt`、退出码和资源检查结果。原 step-8 checkpoint 可以直接复用，不需要重新训练前 8 step。

该门禁通过后，记录为“120 GiB BF16 optimizer state 工程恢复通过”。正式 H100 profile 必须显式设置 `deepspeed_fp32_optimizer_states=true` 并重新执行短程 checkpoint/resume；禁止将本 profile 用于正式训练或指标对比。

## M3.2：三个 atomic 任务 RGB-only 短训练

M3.1 已验证保存和完整恢复 wiring，也提供了 6 个 action/proprio 有效监督更新。M3.2 不再运行 50-step 单 clip 诊断，保持原训练语义 `clean_action_ratio=0.5`，直接验证取放、关节物体和电器三类 atomic 数据能否进入同一个训练过程。

先拉取指定开发分支，并生成三任务 manifest：

```bash
cd /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/xwam-robocasa365
git pull --ff-only origin dev/atomic-robocasa365
git rev-parse HEAD
git status --short

mkdir -p logs/cluster

python scripts/audit_m3_multitask_dataset.py \
  --dataset-root /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/robocasa/robocasa/datasets/v1.0/pretrain/atomic \
  --output logs/cluster/robocasa365_m3_three_task_manifest.json

echo "dataset_audit_exit_code=$?"
```

默认三个任务是 `PickPlaceCounterToCabinet`、`OpenCabinet`、`TurnOnMicrowave`。audit 会为每个任务寻找唯一的日期目录并检查 Parquet、三路视频、16D state 和 12D action。若某任务存在多个日期目录，命令会返回 `fail`；根据报告显式补充路径，例如：

```bash
python scripts/audit_m3_multitask_dataset.py \
  --dataset-root /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/robocasa/robocasa/datasets/v1.0/pretrain/atomic \
  --task-path PickPlaceCounterToCabinet=/HOME/完整路径/日期目录 \
  --task-path OpenCabinet=/HOME/完整路径/日期目录 \
  --task-path TurnOnMicrowave=/HOME/完整路径/日期目录 \
  --output logs/cluster/robocasa365_m3_three_task_manifest.json
```

不要把示例中的 `/HOME/完整路径/日期目录` 原样执行；用 audit 列出的真实目录替换。manifest 的 `ok/result` 必须是 `true/pass`，三个任务的 warnings/errors 必须为空。

数据审计通过后，在 A800 上运行 12-step FP32/no-checkpoint 短训练：

```bash
set -o pipefail

python scripts/train_sft.py \
  model_config=configs/model/wan22_5b_robocasa365_atomic.yaml \
  data_config=configs/data/robocasa365_m3_three_task.yaml \
  hardware_config=configs/hardware/a800_80gb_debug.yaml \
  experiment_config=configs/experiment/robocasa365_m3_three_task_short.yaml \
  wan_checkpoint_dir=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/Wan-AI/Wan2.2-TI2V-5B \
  pretrained_checkpoint=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/x-wam/xwam_checkpoints \
  exp_root=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/experiments/xwam-robocasa365 \
  exp_name=robocasa365_m3_three_task_short \
  2>&1 | tee logs/cluster/robocasa365_m3_three_task_short.log

echo "train_exit_code=${PIPESTATUS[0]}"
```

解析后的配置必须包含 `deepspeed_fp32_optimizer_states=true`、`deepspeed_exclude_frozen_parameters=false`、`clean_action_ratio=0.5`、`enable_checkpointing=false`、`trainer_max_steps=12` 和 `train_shuffle=false`。日志还应打印三任务 dataset provenance。

训练返回码为 0 后执行机器审计：

```bash
python scripts/audit_m3_multitask_short.py \
  --log logs/cluster/robocasa365_m3_three_task_short.log \
  --manifest logs/cluster/robocasa365_m3_three_task_manifest.json \
  --output logs/cluster/robocasa365_m3_three_task_short_audit.json

echo "training_audit_exit_code=$?"
```

audit 要求 step 0～11 完整、task index 为合法整数、三个任务均至少出现一次，并允许 Lightning/DeepSpeed 随机 sampler 产生短前缀波动；默认任务计数最大值与最小值之差不能超过 2。它还检查监督与 clean-action 两个分支、action/proprio loss 对应关系、有限 loss、depth loss 为 0，以及 result `pass/global_step=12`。不要求固定 `0,1,2` 读取顺序，也不要求短程 loss 达到固定降幅。若旧 audit 仅因 `balanced_round_robin` 失败，拉取新版后直接重审原日志，无需重新训练。反馈 manifest、训练日志、更新后的 audit JSON，以及实验 `runs/` 下的 config/metadata/result JSON。

## X-WAM Conda 环境

先按 `docs/ENVIRONMENT_PLAN.md` 的 E0 步骤审计当前 `abot_m05`，日志写入 `logs/cluster/`。`ok=false` 只表示不能直接运行；当 `clone_base_ok=true` 且 `reuse_recommendation=clone_then_patch` 时，可以 clone 为独立环境后补依赖。不要在 `abot_m05` 中直接运行全量依赖安装。

当前 A800 日志已经满足 clone 条件。执行以下完整命令；它不会取消现有代理：

```bash
conda env list
conda create -n xwam-robocasa365 --clone abot_m05
conda activate xwam-robocasa365

export TORCH_EXTENSIONS_DIR=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/.cache/torch_extensions/xwam-robocasa365

cd /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/xwam-robocasa365
python -m pip uninstall -y wam
bash scripts/install_starlight_dependencies.sh dry-run
bash scripts/install_starlight_dependencies.sh apply

python scripts/audit_starlight_environment.py \
  --require-gpu \
  --include-deepspeed-report \
  --log-file logs/cluster/xwam_robocasa365_environment.json
```

如果同名环境已经存在，跳过 `conda create`，直接激活并从 `dry-run` 开始。不要为了重试自行删除环境。反馈以下文件：

1. `logs/cluster/xwam_dependency_dry-run_*.log`。
2. `logs/cluster/xwam_dependency_apply_*.log`。
3. `logs/cluster/xwam_dependency_apply_*_after.txt`。
4. `logs/cluster/xwam_robocasa365_environment.json`。

安装脚本会拒绝在 `abot_m05` 中执行，也会拒绝保留从 ABot 继承且依赖冲突的 editable `wam`；核心约束文件会阻止 pip 替换已验证的 Torch、torchvision、torchaudio 和 FlashAttention。卸载 clone 中的 `wam` 不会删除 ABot 源码或修改母环境。

当前容器中 `pip check` 会把旧 Decord 0.6.0 wheel tag 报为平台不支持。只有当该提示是唯一输出且 Decord runtime import 成功时，安装器和环境 audit 才将其记录为 warning；其他冲突仍返回失败。真实视频解码验证使用 `CloseFridge` episode 0 左相机 MP4。

## M4.0 复用已有 RoboCasa simulator 环境

本轮只筛选现有环境，不安装、升级或删除任何包。先保存环境清单：

```bash
conda env list | tee logs/cluster/robocasa_conda_env_list.log
```

对每个疑似 RoboCasa 环境分别执行；将 `<候选环境名>` 替换为实际名称，输出文件名也保持一致，便于比较：

```bash
conda activate <候选环境名>

cd /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/xwam-robocasa365
git pull --ff-only origin dev/atomic-robocasa365
git rev-parse HEAD

python scripts/audit_robocasa365_simulator_environment.py \
  --output logs/cluster/robocasa365_simulator_<候选环境名>_registry.json

python scripts/audit_robocasa365_simulator_environment.py \
  --runtime-smoke \
  --runtime-timeout 600 \
  --output logs/cluster/robocasa365_simulator_<候选环境名>_runtime.json
```

第一条审计验证 RoboCasa365 `1.0.1` 依赖和 Atomic-Seen 18 注册。第二条在隔离子进程中以 `MUJOCO_GL=egl` 创建 `CloseFridge(target, seed=0)`，执行 reset 和一条完整 12D PandaOmron 动作，并验证 16D state、三路 RGB 与 render。脚本尊重环境中已有的 `MUJOCO_GL`；如已有值不是 `egl`，报告会记录实际值而不会覆盖。

反馈以下证据：

1. `logs/cluster/robocasa_conda_env_list.log`。
2. 每个候选环境的 registry JSON 和 runtime JSON。
3. 对应 commit SHA，以及失败时完整 `stderr`/traceback（JSON 中已保存）。

最终只接受 `ok=true` 且 `reuse_recommendation=reuse_ready`。`RoboCasa 0.2.x` 会明确标记为 `legacy_robocasa_only`；它能运行旧任务不代表能用于 RoboCasa365 Atomic-Seen。

## M4.1 CloseFridge 随机闭环门禁

这一步只使用 `robocasa-abot`，不加载 Torch、Wan2.2 或 X-WAM。它固定 `CloseFridge(target, seed=0, layout=1, style=1)`，执行最多 20 step，验证具名 16D observation、完整 12D Gym 动作、三路视频和结果落盘。`CloseFridge` 官方 horizon 是 900；20 step 仅是工程烟测，不能当作正式 benchmark。

```bash
conda activate robocasa-abot

cd /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/xwam-robocasa365
git pull --ff-only origin dev/atomic-robocasa365
git rev-parse HEAD
mkdir -p logs/cluster

export MUJOCO_GL=egl
python evaluation/run_robocasa365_random_rollout.py \
  --config configs/evaluation/robocasa365_close_fridge_m4_random_smoke.json \
  --output-root /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/experiments/xwam-robocasa365/evaluation/m4_random_smoke \
  2>&1 | tee logs/cluster/robocasa365_close_fridge_m4_random_smoke.log

echo "rollout_exit_code=${PIPESTATUS[0]}"
```

这里不需要 `/usr/bin/time -v`，也不要求设置 `set -o pipefail`。上面的 `PIPESTATUS[0]` 在 Bash 中直接给出 Python 进程的退出码；必须是 `0`。

运行目录是输出根目录下新生成的 UTC run id，例如 `20260807T120000Z/`。验收以下内容：

1. `summary.json` 为 `result=pass`，且 `episodes_expected/observed/completed=1/1/1`、`episodes_failed=0`、`episodes_missing=0`。
2. `metadata.json` 记录当前 Git commit、dirty 状态、RoboCasa/robosuite/MuJoCo 版本、`MUJOCO_GL=egl` 和 runtime horizon 900。
3. `CloseFridge/episode_000_seed_000000/episode.json` 的 state/action dimension 为 `16/12`，`steps<=20`，`base_nonzero_steps>0`，scene 为 layout/style `1/1`。
4. 同目录 `rollout.mp4` 非空，画面为左 agentview、右 agentview、eye-in-hand 三路 `256x256` RGB 横向拼接，因此视频帧尺寸应为 `768x256`。
5. 随机策略的 `success=false` 很正常，不影响本次工程门禁；只有 rollout 异常、缺失 episode、合同错误或证据未写完整才算失败。

反馈以下文件，不需要重新上传模型或数据：

- `logs/cluster/robocasa365_close_fridge_m4_random_smoke.log`
- 本次 run 的 `metadata.json`、`summary.json` 和逐 episode `episode.json`
- `rollout.mp4` 的文件大小；如画面异常，再提供视频或抽帧

## M4.2 X-WAM broker 单请求闭环门禁

首轮必须在同一个 A800 Pod 中开三个终端。Broker 和 simulator 使用 `robocasa-abot`，policy server 使用 `xwam-robocasa365`；三个进程通过本机 `127.0.0.1` 通信。当前只执行一次 X-WAM inference，模型返回32个12D动作后，client执行前4个动作。它验证真实checkpoint/归一化/网络/环境链路，不是900-step完整评测。

先在任意终端更新代码并检查M3 checkpoint：

```bash
cd /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/xwam-robocasa365
git pull --ff-only origin dev/atomic-robocasa365
git rev-parse HEAD
mkdir -p logs/cluster

M3_EXP=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/experiments/xwam-robocasa365/close_fridge_m3_overfit_120g_gate1
test -s "$M3_EXP/config.yaml"
test -s "$M3_EXP/checkpoints/last.ckpt/checkpoint/mp_rank_00_model_states.pt"
readlink -f "$M3_EXP/checkpoints/last.ckpt"
```

`last.ckpt` 应指向已经通过resume门禁的global step 10 checkpoint，而不是旧step 8。如果链接不存在或指向错误目录，先停止并反馈 `find "$M3_EXP/checkpoints" -maxdepth 2 -type f -o -type l`，不要猜测或覆盖checkpoint。

Policy环境需要PyZMQ。先只检查，不要重装完整requirements：

```bash
conda activate xwam-robocasa365
python - <<'PY'
import zmq
print("pyzmq:", zmq.__version__)
PY
```

只有这里报 `ModuleNotFoundError: zmq` 时，才执行下面的单包安装；保留当前代理，不要unset：

```bash
python -m pip install \
  -c configs/environment/xwam_starlight_constraints.txt \
  'pyzmq==27.1.0'
```

### 终端A：启动broker

```bash
conda activate robocasa-abot
cd /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/xwam-robocasa365

python evaluation/run_robocasa365_policy_broker.py \
  --host 127.0.0.1 \
  --frontend-port 10086 \
  --backend-port 10087 \
  2>&1 | tee logs/cluster/robocasa365_m4_policy_broker.log
```

看到 `RoboCasa365 broker ready` 后保留该终端。完成client后按一次 `Ctrl-C` 正常停止broker。

### 终端B：启动X-WAM policy server

```bash
conda activate xwam-robocasa365
cd /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/xwam-robocasa365

M3_EXP=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/experiments/xwam-robocasa365/close_fridge_m3_overfit_120g_gate1
WAN_DIR=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/Wan-AI/Wan2.2-TI2V-5B
DATASET=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/robocasa/robocasa/datasets/v1.0/pretrain/atomic/CloseFridge/20250819

CUDA_VISIBLE_DEVICES=0 python evaluation/robocasa365_policy_server.py \
  --experiment-dir "$M3_EXP" \
  --wan-checkpoint-dir "$WAN_DIR" \
  --dataset-path "$DATASET" \
  --broker-address 127.0.0.1 \
  --broker-port 10087 \
  --denoise-steps 50 \
  --action-denoise-steps 10 \
  --max-requests 1 \
  --startup-report logs/cluster/robocasa365_m4_policy_server.json \
  2>&1 | tee logs/cluster/robocasa365_m4_policy_server.log

echo "policy_server_exit_code=${PIPESTATUS[0]}"
```

首次不要加 `--compile-model`。模型加载可能需要数分钟；等日志出现 `Policy runtime ready`，并确认终端A出现 `policy server READY` 后，再启动client。server收到一个请求后会自动退出，退出码必须为0。

### 终端C：运行simulator client

```bash
conda activate robocasa-abot
cd /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/xwam-robocasa365
export MUJOCO_GL=egl

python evaluation/run_robocasa365_policy_rollout.py \
  --config configs/evaluation/robocasa365_close_fridge_m4_policy_smoke.json \
  --broker-address 127.0.0.1 \
  --broker-port 10086 \
  --output-root /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/experiments/xwam-robocasa365/evaluation/m4_policy_smoke \
  2>&1 | tee logs/cluster/robocasa365_close_fridge_m4_policy_smoke.log

echo "policy_client_exit_code=${PIPESTATUS[0]}"
```

验收标准：

1. Server startup JSON最终为 `result=pass/ok=true`、`processed_requests=1`、`failed_requests=0`，checkpoint解析到M3 step 10；加载报告不得包含非冻结missing或unexpected key。
2. Server报告 model action/proprio为12/16、`predicted_action_steps=32`、schema SHA、数据stats路径、GPU/CUDA峰值和实际inference耗时。
3. Client summary为 `result=pass`、episode `1/1` 完成、failed/missing为0、`policy_requests=1`、`total_steps=4`。
4. Episode记录中response为32x12，executed action steps为4；state/action dimension为16/12，所有值有限且通过Gym bounds，checkpoint路径与server一致。
5. `rollout.mp4` 为三相机 `768x256` 视频。任务 `success=false` 不阻塞本轮；任何模型加载错误、request超时、server error、动作越界或证据缺失都判失败。

反馈以下文件；模型、checkpoint、完整评测目录和视频仍不得提交Git：

- `logs/cluster/robocasa365_m4_policy_broker.log`
- `logs/cluster/robocasa365_m4_policy_server.log`
- `logs/cluster/robocasa365_m4_policy_server.json`
- `logs/cluster/robocasa365_close_fridge_m4_policy_smoke.log`
- Client本次run的 `metadata.json`、`summary.json`、逐episode `episode.json` 和视频文件大小

## M4.3 CloseFridge 900-step可恢复闭环

本轮继续使用A800，不进入H100训练。完整900步在任务未提前成功时需要225次模型请求；按M4.2实测30.6秒/请求估算，纯推理约1.9小时，另加约6分钟模型加载和模拟器开销。三个进程仍须位于同一个Pod，且不要让Pod在client恢复前被回收。

先更新代码并创建一个全新的控制目录。下面固定使用`m4_full_gate1`；如果目录已经存在，不要覆盖或删除，手工把三个终端中的名字统一改成`m4_full_gate2`：

```bash
cd /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/xwam-robocasa365
git pull --ff-only origin dev/atomic-robocasa365
git rev-parse HEAD
git status --short

M4_CONTROL=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/experiments/xwam-robocasa365/evaluation/m4_full_gate1
test ! -e "$M4_CONTROL"
mkdir -p "$M4_CONTROL/runs"
```

`git status --short`必须为空，`test ! -e`必须成功。M4.3恢复会拒绝dirty工作区和commit漂移。

### 终端A：启动连续broker

```bash
conda activate robocasa-abot
cd /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/xwam-robocasa365
M4_CONTROL=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/experiments/xwam-robocasa365/evaluation/m4_full_gate1

python evaluation/run_robocasa365_policy_broker.py \
  --host 127.0.0.1 \
  --frontend-port 10086 \
  --backend-port 10087 \
  2>&1 | tee "$M4_CONTROL/broker.log"
```

### 终端B：启动连续policy server

```bash
conda activate xwam-robocasa365
cd /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/xwam-robocasa365

M4_CONTROL=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/experiments/xwam-robocasa365/evaluation/m4_full_gate1
M3_EXP=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/experiments/xwam-robocasa365/close_fridge_m3_overfit_120g_gate1
WAN_DIR=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/Wan-AI/Wan2.2-TI2V-5B
DATASET=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/robocasa/robocasa/datasets/v1.0/pretrain/atomic/CloseFridge/20250819

test ! -e "$M4_CONTROL/server_requests.jsonl"

CUDA_VISIBLE_DEVICES=0 python evaluation/robocasa365_policy_server.py \
  --experiment-dir "$M3_EXP" \
  --wan-checkpoint-dir "$WAN_DIR" \
  --dataset-path "$DATASET" \
  --broker-address 127.0.0.1 \
  --broker-port 10087 \
  --denoise-steps 50 \
  --action-denoise-steps 10 \
  --max-requests 0 \
  --minimum-requests 1 \
  --graceful-stop-is-pass \
  --startup-report "$M4_CONTROL/server_report.json" \
  --request-journal "$M4_CONTROL/server_requests.jsonl" \
  2>&1 | tee "$M4_CONTROL/server.log"

echo "policy_server_exit_code=${PIPESTATUS[0]}"
```

等`Policy runtime ready`和broker的`policy server READY`。Server会持续服务；不要在client第一次故意中断时停止它。

### 终端C：新建run并故意中断

```bash
conda activate robocasa-abot
cd /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/xwam-robocasa365
export MUJOCO_GL=egl
M4_CONTROL=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/experiments/xwam-robocasa365/evaluation/m4_full_gate1

python evaluation/run_robocasa365_policy_rollout_resumable.py \
  --config configs/evaluation/robocasa365_close_fridge_m4_full.json \
  --output-root "$M4_CONTROL/runs" \
  2>&1 | tee "$M4_CONTROL/client_initial.log"
```

日志每完成一个4-action chunk会打印`M4.3 progress saved`。第一次看到`steps=8/900 requests=2`后，在终端C按一次`Ctrl-C`。Client应写出`result=interrupted`、`resumable=true`、`completed_steps=8`并以130退出；这是本轮设计内的故意中断，不是失败。

Client打印8步后可能已经发出step 8的新请求；此时server会完成这个孤立请求，恢复后相同request ID会确定性重试，因此server JSONL可能多一条重复的成功记录，`processed_requests`也可能比client多1。审计器按run级request ID去重并要求所有关联记录均成功；这不等于重复执行环境动作。

列出唯一run目录，并把实际run id手工填入`RUN_DIR`：

```bash
find "$M4_CONTROL/runs" -mindepth 1 -maxdepth 1 -type d -print
RUN_DIR="$M4_CONTROL/runs/把这里替换为实际run_id"

jq '{run_id, result, resumable, completed_steps, completed_policy_requests}' \
  "$RUN_DIR/interruption.json"
jq '{result, steps, request_count:(.request_records|length), step_count:(.step_records|length)}' \
  "$RUN_DIR/CloseFridge/episode_000_seed_000000/progress.json"
```

必须看到progress的request/step结构完整后才恢复：

```bash
python evaluation/run_robocasa365_policy_rollout_resumable.py \
  --resume-run-dir "$RUN_DIR" \
  2>&1 | tee "$M4_CONTROL/client_resume.log"

echo "policy_client_resume_exit_code=${PIPESTATUS[0]}"
```

恢复日志必须先出现`M4.3 deterministic replay passed: steps=8`，且`max_state_error <= 1e-5`，然后从未完成位置继续。若回放漂移，立即停止并反馈首个step和误差；禁止增大容差或删除progress重跑来掩盖问题。

Client最终正常退出后，在终端B按一次`Ctrl-C`。由于设置了`--graceful-stop-is-pass`且至少处理过一个请求，server report应为`result=pass/ok=true`且退出码为0。随后可在终端A按`Ctrl-C`。

### 机器审计

在终端C运行：

```bash
python scripts/audit_robocasa365_policy_rollout.py \
  --run-dir "$RUN_DIR" \
  --server-request-journal "$M4_CONTROL/server_requests.jsonl" \
  --output "$M4_CONTROL/audit.json" \
  2>&1 | tee "$M4_CONTROL/audit.log"

ffprobe -v error \
  -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames \
  -show_entries format=duration,size \
  -of json \
  "$RUN_DIR/CloseFridge/episode_000_seed_000000/rollout.mp4"
```

验收标准：

1. Audit为`ok=true/result=pass`，所有checks为true；若任务未提前成功，steps为900、policy requests为225。RoboCasa在官方horizon返回`truncated=true`也视为完整900步。
2. 恢复回放8步逐步通过16D state容差；原progress中的前两个请求不重复调用模型，最终request ID不重复。
3. 每个client请求都能在server JSONL找到同checkpoint、`[32,12]`且`result=pass`的记录；不存在关联失败记录。
4. Episode/summary为pass，完整12D动作通过Gym bounds；success可为false，本轮不把step-10工程checkpoint解释为策略指标。
5. 视频为三相机`768x256`；未提前结束时帧缓存与MP4均为46帧（reset加每20步），文件非空。

反馈`git rev-parse HEAD`、`audit.json`、`server_report.json`、`server_requests.jsonl`、两段client日志、summary/episode/progress、视频ffprobe和三个进程退出码。原始产物继续保留在Git之外。

## M6 Clariden：4×GH200 正式 profile 门禁

数据预检通过后，先在干净worktree运行完整18任务、GBS128、16,390-step scheduler的短门禁；不得直接开始正式5 epoch：

```bash
cd /capstor/store/cscs/swissai/aa004/users/zjingchen/terry_nys/src/xwam-robocasa365
git pull --ff-only origin dev/atomic-robocasa365
git rev-parse HEAD
git status --short

sbatch deployment/clariden/smoke_m6_gate_xwam.sbatch
```

首选profile为`4×micro-batch 4×accumulation 8=GBS128`。如果日志明确是CUDA OOM，使用新Job从公开pretrained权重重新开始safe profile；不能复用首选profile的半成品checkpoint：

```bash
sbatch \
  --export=ALL,XWAM_M6_HARDWARE_CONFIG=configs/hardware/gh200x4_96gb_gbs128_safe.yaml \
  deployment/clariden/smoke_m6_gate_xwam.sbatch
```

只有总日志出现`[PASS] X-WAM Clariden M6 4xGH200 formal-profile step 2 to 4 gate`且audit为`ok=true/result=pass`，才能生成并提交16,390-step正式训练作业。

## M6：4×H100 RGB-only训练（兼容路径）

本阶段只使用 Atomic-Seen 同名18任务的 `pretrain/atomic` 数据，不读取 composite，也不启用depth。模型前向保持BF16 mixed precision；正式optimizer state恢复FP32，并关闭A800调试所用的CPU offload和冻结参数排除。

### 1. 公共路径与18任务数据清单

```bash
conda activate xwam-robocasa365

cd /HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/xwam-robocasa365
git pull --ff-only origin dev/atomic-robocasa365
git rev-parse HEAD
git status --short --branch

DATA_ROOT=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/robocasa/robocasa/datasets/v1.0/pretrain/atomic
WAN_DIR=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/Wan-AI/Wan2.2-TI2V-5B
XWAM_CKPT=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/x-wam/xwam_checkpoints
EXP_ROOT=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/experiments/xwam-robocasa365
STATS_WORK=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/tmp/xwam-m6-stats

python scripts/build_robocasa365_m6_training_manifest.py \
  --dataset-root "$DATA_ROOT" \
  --output logs/cluster/robocasa365_m6_atomic_seen18_manifest.json
```

如果某个任务下存在多个日期目录，脚本会失败而不是猜测。核对数据版本后为每个歧义任务追加一次：

```bash
  --task-path TaskName=/absolute/path/to/TaskName/YYYYMMDD
```

manifest必须为`ok=true/result=pass`、18个唯一任务，且每项包含`valid_clips`。有效clip跨度固定为32，即`(sequence_length-1)*frame_skip=(9-1)*4`。

### 2. 跨任务统计与精确step

```bash
mkdir -p "$STATS_WORK"

python scripts/compute_robocasa365_global_stats.py \
  --manifest logs/cluster/robocasa365_m6_atomic_seen18_manifest.json \
  --work-dir "$STATS_WORK" \
  --output logs/cluster/robocasa365_m6_atomic_seen18_global_stats.json

python scripts/audit_robocasa365_m6_preflight.py \
  --manifest logs/cluster/robocasa365_m6_atomic_seen18_manifest.json \
  --stats logs/cluster/robocasa365_m6_atomic_seen18_global_stats.json \
  --global-batch-size 128 \
  --epochs 5 \
  --output logs/cluster/robocasa365_m6_h100_preflight.json
```

统计脚本不解码视频、不生成depth；它用临时memmap读取全部Parquet frame，临时原始矩阵大小约为`total_frames×28 bytes`。preflight输出以下精确关系：

```text
N = total_valid_clips
steps_per_epoch = floor(N / 128)
num_training_steps = 5 * steps_per_epoch
samples_dropped_per_epoch = N % 128
```

只有preflight为`pass`才申请4×H100训练节点。

### 3. H100门禁：全新运行到step 2

确认当前节点只暴露4张H100 80GB。不要使用`torchrun`；Lightning根据`devices: 4`启动四个本地进程。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1

GATE_NAME=robocasa365_m6_h100_gate_rgb_seed42

python scripts/train_sft.py \
  model_config=configs/model/wan22_5b_robocasa365_atomic.yaml \
  data_config=configs/data/robocasa365_m6_atomic_seen18.yaml \
  hardware_config=configs/hardware/h100x4_80gb_gbs128.yaml \
  experiment_config=configs/experiment/robocasa365_m6_h100_gate.yaml \
  wan_checkpoint_dir="$WAN_DIR" \
  pretrained_checkpoint="$XWAM_CKPT" \
  exp_root="$EXP_ROOT" \
  exp_name="$GATE_NAME" \
  2>&1 | tee logs/cluster/robocasa365_m6_h100_gate_initial.log
```

首选profile是`4 GPU×micro-batch 4×gradient accumulation 8=GBS 128`。预期达到global step 2、完成DeepSpeed checkpoint，并在`runs/<run-id>_optimizer_state/`产生四个rank的FP32实态报告。先保存首轮证据路径：

```bash
GATE_EXP="$EXP_ROOT/$GATE_NAME"
INITIAL_RESULT=$(ls -1t "$GATE_EXP"/runs/*_result.json | head -n 1)
INITIAL_PREFIX=${INITIAL_RESULT%_result.json}
test -e "$GATE_EXP/checkpoints/last.ckpt"
```

### 4. H100门禁：从step 2恢复到step 4

```bash
python scripts/train_sft.py \
  model_config=configs/model/wan22_5b_robocasa365_atomic.yaml \
  data_config=configs/data/robocasa365_m6_atomic_seen18.yaml \
  hardware_config=configs/hardware/h100x4_80gb_gbs128.yaml \
  experiment_config=configs/experiment/robocasa365_m6_h100_gate.yaml \
  wan_checkpoint_dir="$WAN_DIR" \
  pretrained_checkpoint="$XWAM_CKPT" \
  exp_root="$EXP_ROOT" \
  exp_name="$GATE_NAME" \
  trainer_max_steps=4 \
  resume_checkpoint="$GATE_EXP/checkpoints/last.ckpt" \
  2>&1 | tee logs/cluster/robocasa365_m6_h100_gate_resumed.log

RESUMED_RESULT=$(ls -1t "$GATE_EXP"/runs/*_result.json | head -n 1)
RESUMED_PREFIX=${RESUMED_RESULT%_result.json}

python scripts/audit_robocasa365_m6_h100_gate.py \
  --initial-metadata "${INITIAL_PREFIX}_metadata.json" \
  --initial-result "${INITIAL_PREFIX}_result.json" \
  --initial-events "${INITIAL_PREFIX}_checkpoint_events.jsonl" \
  --initial-optimizer-dir "${INITIAL_PREFIX}_optimizer_state" \
  --initial-log logs/cluster/robocasa365_m6_h100_gate_initial.log \
  --resumed-metadata "${RESUMED_PREFIX}_metadata.json" \
  --resumed-result "${RESUMED_PREFIX}_result.json" \
  --resumed-events "${RESUMED_PREFIX}_checkpoint_events.jsonl" \
  --resumed-optimizer-dir "${RESUMED_PREFIX}_optimizer_state" \
  --resumed-log logs/cluster/robocasa365_m6_h100_gate_resumed.log \
  --output logs/cluster/robocasa365_m6_h100_gate_audit.json
```

门禁要求：两段运行分别精确达到step 2/4；第二段明确从checkpoint恢复；四个rank的optimizer浮点state实际均为FP32且非空；H100/显存/GBS/ZeRO合同通过；loss有限、depth loss为0；两次checkpoint保存均有complete事件。

如果首选profile发生CUDA OOM，使用新的`GATE_NAME`从公开X-WAM权重重新开始，并把两条训练命令的hardware层统一替换为：

```text
configs/hardware/h100x4_80gb_gbs128_safe.yaml
```

safe profile为`4×micro-batch 2×accumulation 16`，GBS仍为128。不要把首选profile的半成品checkpoint与safe profile混用，也不要改回A800的BF16 optimizer state、CPU offload或排除冻结参数。

### 5. 正式5-epoch训练

只有门禁audit为`ok=true/result=pass`才启动。正式实验必须是新目录，从公开X-WAM pretrained checkpoint初始化，不能接着step-4门禁训练。

```bash
FORMAL_NAME=robocasa365_m6_atomic_seen18_rgb_seed42

python scripts/train_sft.py \
  model_config=configs/model/wan22_5b_robocasa365_atomic.yaml \
  data_config=configs/data/robocasa365_m6_atomic_seen18.yaml \
  hardware_config=configs/hardware/h100x4_80gb_gbs128.yaml \
  experiment_config=configs/experiment/robocasa365_m6_h100_rgb_formal.yaml \
  wan_checkpoint_dir="$WAN_DIR" \
  pretrained_checkpoint="$XWAM_CKPT" \
  exp_root="$EXP_ROOT" \
  exp_name="$FORMAL_NAME" \
  2>&1 | tee logs/cluster/robocasa365_m6_h100_formal.log
```

若门禁最终采用safe profile，正式命令也使用同一safe hardware层。训练步数不手填，由已审计manifest自动计算；每轮全局shuffle后只丢弃不足128的尾部。正常结束会额外保存`final-step=<num_training_steps>.ckpt`。

正式训练因调度或节点中断时，保留完全相同的model/data/hardware/experiment层和`FORMAL_NAME`，只追加：

```bash
resume_checkpoint="$EXP_ROOT/$FORMAL_NAME/checkpoints/last.ckpt"
```

不得改变world size、GBS、manifest、global stats或总epoch数。完成后定位最新证据并审计：

```bash
FORMAL_EXP="$EXP_ROOT/$FORMAL_NAME"
FORMAL_RESULT=$(ls -1t "$FORMAL_EXP"/runs/*_result.json | head -n 1)
FORMAL_PREFIX=${FORMAL_RESULT%_result.json}

python scripts/audit_robocasa365_m6_formal_training.py \
  --metadata "${FORMAL_PREFIX}_metadata.json" \
  --result "${FORMAL_PREFIX}_result.json" \
  --events "${FORMAL_PREFIX}_checkpoint_events.jsonl" \
  --optimizer-dir "${FORMAL_PREFIX}_optimizer_state" \
  --log logs/cluster/robocasa365_m6_h100_formal.log \
  --output logs/cluster/robocasa365_m6_h100_formal_audit.json
```

如果正式训练经历多次resume，传给审计器的log应为本次最终调用的完整log；metadata/result/events/optimizer目录必须来自同一个run prefix。最终反馈commit、preflight、门禁audit、正式audit、两段门禁日志、正式日志、`nvidia-smi`和实验checkpoint目录清单。所有`logs/`与实验产物保持在Git之外。

## 外部模型路径

复用已有完整 Wan2.2 模型：

```text
/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/Wan-AI/Wan2.2-TI2V-5B
```

使用 X-WAM pretrained checkpoint：

```text
/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/x-wam/xwam_checkpoints/pretrained/checkpoints/last.ckpt
```

在不加载 Torch 的情况下检查文件：

```bash
WAN_DIR=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/Wan-AI/Wan2.2-TI2V-5B
XWAM_CKPT=/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/x-wam/xwam_checkpoints/pretrained/checkpoints/last.ckpt

test -s "$WAN_DIR/config.json"
test -s "$WAN_DIR/Wan2.2_VAE.pth"
test -s "$WAN_DIR/models_t5_umt5-xxl-enc-bf16.pth"
test -s "$XWAM_CKPT/checkpoint/mp_rank_00_model_states.pt"
```

不要把 Wan2.2 复制进 Git 仓库；运行配置会直接指向现有目录。

## 验证级别

### `local-static`：本地静态验证

- 不执行 import 的 Python 语法编译。
- Skill 结构验证。
- 文档与变更记录检查。
- 不依赖 Torch 的单元测试和 schema fixture。

### `cluster-smoke`：超算烟测

- 导入 Torch/CUDA/FlashAttention/DeepSpeed。
- 加载模型组件和 checkpoint。
- 加载一个真实数据 batch。
- 完成一步 forward/backward。
- 创建一个模拟器环境并完成一次 rollout。

### `cluster-train`：超算正式训练

- 多 GPU 初始化。
- 测量显存和吞吐。
- 保存并恢复 checkpoint。
- H100 训练和计划内 atomic 评测。

运行时烟测命令会与对应实现一起加入仓库，避免尚未实现的占位命令被误认为可用。

## 硬件分工

- A100/A800：配置环境、测量显存、单 batch 训练、短程过拟合、推理和 rollout 调试。
- H100：只有 A100/A800 烟测通过后才用于正式训练。
- 必须记录准确显存容量；A100 40GB 和 80GB 使用不同配置。

## 反馈要求

复制 `.agents/skills/xwam-robocasa365-workflow/references/cluster-feedback-template.md`，填写所有适用字段并附上相关日志。不能只反馈最后一行异常。
