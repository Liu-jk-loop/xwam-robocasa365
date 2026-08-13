# X-WAM Evaluation Guidelines

## RoboCasa365 Atomic 适配状态

当前项目的 RoboCasa365 新评测入口是 `run_robocasa365_random_rollout.py`。M4.1 先在独立 simulator 环境中运行一个不加载 X-WAM 的固定随机闭环，用于验证 Atomic-Seen 注册、在线 16D state、完整 12D PandaOmron action、三路 RGB 视频和结果聚合：

```bash
python evaluation/run_robocasa365_random_rollout.py \
  --config configs/evaluation/robocasa365_close_fridge_m4_random_smoke.json \
  --output-root /path/to/evaluation/m4_random_smoke
```

默认任务为 `CloseFridge(target, seed=0, layout=1, style=1)`，只运行 20 step。官方 horizon 为 900，因此这条命令只验证工程链路；随机策略是否成功完成任务不作为门禁。完整星光命令和验收字段见 `docs/CLUSTER_RUNBOOK.md`。

下面的 `robocasa_client.py`、24-task 索引、500-step 假设和旧 policy server 命令保留为上游 X-WAM 参考，目前不是 RoboCasa365 Atomic 的有效入口。新版 M4.2 入口如下。

M4.2 已提供三条 RoboCasa365 专用入口：

- `run_robocasa365_policy_broker.py`：只转发请求，不加载模型或模拟器。
- `robocasa365_policy_server.py`：在 policy Conda环境加载Wan2.2、M3 DeepSpeed checkpoint和真实PandaOmron stats，拒绝与单任务 checkpoint 不一致的任务请求并返回完整12D动作。
- `run_robocasa365_policy_rollout.py`：在 simulator Conda环境发送三路RGB、16D state和prompt，执行具名Gym动作并保存证据。

首轮配置 `configs/evaluation/robocasa365_close_fridge_m4_policy_smoke.json` 只执行一次模型请求和4个动作。三终端精确命令、启动顺序、checkpoint路径和验收字段见 `docs/CLUSTER_RUNBOOK.md`。不要再把下面 legacy policy server/client 命令用于RoboCasa365。

M4.3 完整 horizon 使用 `configs/evaluation/robocasa365_close_fridge_m4_full.json` 和 `run_robocasa365_policy_rollout_resumable.py`。它每个 step 原子保存恢复进度；使用 `--resume-run-dir /path/to/existing/run` 时会从相同 seed 重建环境、回放已执行动作并校验16D state，然后继续未完成的 action chunk。`scripts/audit_robocasa365_policy_rollout.py` 负责把 client 证据与 server request JSONL 交叉审计。精确命令仍以 `docs/CLUSTER_RUNBOOK.md` 为准。

## M6 Atomic-Seen 18 正式评测

Clariden 正式评测使用
`configs/evaluation/robocasa365_m6_atomic18_8server_16client.json` 和
`deployment/clariden/eval_m6_atomic18_xwam.sbatch`：单节点4张GH200，每卡两个独立
X-WAM server，共8个固定server/broker对；16个RoboCasa client按版本化topology固定路由。
`client6`与`client7`各自串行执行两个任务，因此16个client恰好覆盖18个Atomic-Seen任务。

默认每任务50个episode，环境seed为`42..91`，模型侧每次replan固定seed 42；
`replan_steps=20`、action denoise 10步。X-WAM仍保留50步video scheduler，但policy
调用使用ANS `early_stop`，得到动作后在第10次模型前向停止，不继续生成完整视频。
每个episode都沿用M4.3的动作级原子进度和确定性回放。相同Git commit、checkpoint与
`XWAM_EVAL_ID`重新提交时，已完成episode直接跳过，中断episode从原progress恢复。
正式评测产物统一写入IOPS的`/iopsstor/scratch/cscs/zjingchen/terry_nys/x-wam-eval/`
目录，不把视频或逐episode推理结果写到Capstor/Store。
Simulator client与FastWAM正式评测使用同一已验证来源：EDF内将Store的
`src/robocasa`和`src/robosuite`放在`PYTHONPATH`最前，复用此前下载的完整assets，
而不使用SQSH内不完整的`/opt/robocasa`。在加载policy前会写出
`robocasa_eval_runtime_probe.json`并检查模块来源和`Sink025`；当前容器只枚举一个
EGL device，所有client固定device 0，policy server的四卡映射保持不变。

Policy server从M6 manifest读取18个合法任务，并只使用与manifest绑定的跨任务
normalization statistics；每个server还会拒绝topology未分配给自己的任务。最终
`scripts/aggregate_robocasa365_m6_evaluation.py`要求16份client summary、18个任务和
每任务完整episode数全部存在，才写出总体PASS和成功率。完整提交命令见
`docs/CLUSTER_RUNBOOK.md`。

## Installation

Please clone the whole repository with submodules:

```bash
git clone --recurse-submodules https://github.com/sharinka0715/X-WAM.git
cd X-WAM
```

If you have already cloned without submodules:

```bash
git submodule update --init --recursive
```

### Base Environment

Follow the main [README](../README.md) to install the base environment (PyTorch, FlashAttention, etc.).

### RoboCasa

Refer to `third_party/robocasa/README.md` for installation.

### RoboTwin 2.0

Refer to `third_party/RoboTwin/README.md` for installation. You can ignore the `torch` / `huggingface_hub` version requirements in `third_party/RoboTwin/script/requirements.txt`.

### Fix NumPy versions

If you install all evaluation packages in one environment, you should make sure that NumPy version is compatible (we use `numpy==1.23.5` in our experiments).

## Download Checkpoints

Download the checkpoints from Hugging Face:

```bash
hf download sharinka0715/X-WAM-checkpoints --local-dir checkpoints
```

You also need the Wan2.2-TI2V-5B base weights. Specify the path via `--wan_checkpoint_dir` when launching the policy server.

## Evaluation

The evaluation system uses a broker-server-client architecture:
- **Policy Broker**: middleware that dispatches inference requests from clients to servers
- **Policy Server**: loads the model and performs inference
- **Client**: runs the simulation environment and sends observations to the broker

### Step 1: Start the Policy Broker

```bash
python evaluation/policy_broker.py \
    --frontend_port 10086 \
    --backend_port 10087
```

### Step 2: Start the Policy Server(s)

Launch one or more policy servers (each on a separate GPU):

```bash
# RoboCasa
CUDA_VISIBLE_DEVICES=0 python evaluation/policy_server.py \
    --exp_path checkpoints/robocasa_sft \
    --wan_checkpoint_dir /path/to/wan22_5b \
    --broker_port 10087 \
    --denoise_steps 50 \
    --action_denoise_steps 10

# RoboTwin 2.0
CUDA_VISIBLE_DEVICES=0 python evaluation/policy_server.py \
    --exp_path checkpoints/robotwin_sft \
    --wan_checkpoint_dir /path/to/wan22_5b \
    --broker_port 10087 \
    --denoise_steps 50 \
    --action_denoise_steps 10
```

You can launch multiple servers on different GPUs to parallelize inference:

```bash
CUDA_VISIBLE_DEVICES=1 python evaluation/policy_server.py --exp_path checkpoints/robocasa_sft --wan_checkpoint_dir /path/to/wan22_5b --broker_port 10087 &
CUDA_VISIBLE_DEVICES=2 python evaluation/policy_server.py --exp_path checkpoints/robocasa_sft --wan_checkpoint_dir /path/to/wan22_5b --broker_port 10087 &
```

### Step 3: Start the Evaluation Client(s)

#### RoboCasa

Each client evaluates one task. There are 24 tasks in total, indexed 0–23. Launch one client per task:

```bash
python evaluation/robocasa_client.py \
    --env_global_rank 0 \
    --world_size 24 \
    --num_evals_per_worker 5 \
    --server_port 10086 \
    --save_root_dir ./eval_results/robocasa/
```

To evaluate all 24 tasks in parallel:

```bash
for i in $(seq 0 23); do
    python evaluation/robocasa_client.py \
        --env_global_rank $i \
        --world_size 24 \
        --num_evals_per_worker 100 \
        --server_port 10086 \
        --save_root_dir ./eval_results/robocasa/ &
done
wait
```

#### RoboTwin 2.0

If you are using your own RoboTwin installation (not the submodule), modify `ROBOTWIN_ROOT` at the top of `evaluation/robotwin_client.py`:

```python
ROBOTWIN_ROOT = "/path/to/your/RoboTwin"
```

Then launch evaluation for each task:

```bash
python evaluation/robotwin_client.py \
    --task_name adjust_bottle \
    --task_config demo_randomized \
    --num_evals_per_worker 10 \
    --server_port 10086 \
    --save_root_dir ./eval_results/robotwin/
```

To evaluate all 50 tasks:

```bash
TASKS=(adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size click_alarmclock click_bell dump_bin_bigbin grab_roller handover_block handover_mic hanging_mug lift_pot move_can_pot move_pillbottle_pad move_playingcard_away move_stapler_pad open_laptop open_microwave pick_diverse_bottles pick_dual_bottles place_a2b_left place_a2b_right place_bread_basket place_bread_skillet place_burger_fries place_can_basket place_cans_plasticbox place_container_plate place_dual_shoes place_empty_cup place_fan place_mouse_pad place_object_basket place_object_scale place_object_stand place_phone_stand place_shoe press_stapler put_bottles_dustbin put_object_cabinet rotate_qrcode scan_object shake_bottle shake_bottle_horiz stack_blocks_three stack_blocks_two stack_bowls_three stack_bowls_two stamp_seal turn_switch)

for task in "${TASKS[@]}"; do
    python evaluation/robotwin_client.py \
        --task_name $task \
        --task_config demo_randomized \
        --num_evals_per_worker 100 \
        --server_port 10086 \
        --save_root_dir ./eval_results/robotwin/ &
done
wait
```

## Results

Evaluation results (success rates and rollout videos) are saved to the `--save_root_dir` directory.
