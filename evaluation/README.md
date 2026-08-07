# X-WAM Evaluation Guidelines

## RoboCasa365 Atomic 适配状态

当前项目的 RoboCasa365 新评测入口是 `run_robocasa365_random_rollout.py`。M4.1 先在独立 simulator 环境中运行一个不加载 X-WAM 的固定随机闭环，用于验证 Atomic-Seen 注册、在线 16D state、完整 12D PandaOmron action、三路 RGB 视频和结果聚合：

```bash
python evaluation/run_robocasa365_random_rollout.py \
  --config configs/evaluation/robocasa365_close_fridge_m4_random_smoke.json \
  --output-root /path/to/evaluation/m4_random_smoke
```

默认任务为 `CloseFridge(target, seed=0, layout=1, style=1)`，只运行 20 step。官方 horizon 为 900，因此这条命令只验证工程链路；随机策略是否成功完成任务不作为门禁。完整星光命令和验收字段见 `docs/CLUSTER_RUNBOOK.md`。

下面的 `robocasa_client.py`、24-task 索引、500-step 假设和旧 policy server 命令保留为上游 X-WAM 参考，目前不是 RoboCasa365 Atomic 的有效入口。M4.2 会在随机门禁通过后把新版 adapter 接入 broker 和 X-WAM checkpoint。

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
