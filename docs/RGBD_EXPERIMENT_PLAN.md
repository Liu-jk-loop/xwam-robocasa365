# RoboCasa365 RGB-D 实验方案与数据合同

更新时间：2026-08-17

## 当前结论

- 后续新训练默认使用 `clean_action_ratio=0`；ratio 0.5 完整结果仅作为负对照保留。
- 第一轮 RGB-D 指 X-WAM 的“RGB 输入 + inverse-depth 辅助预测监督”，不是把实时 depth 作为 policy 输入。
- RoboCasa365 官方 LeRobot 数据没有训练用 depth 视频，但每个 episode 的 `extras/` 保存了 `states.npz`、`ep_meta.json` 和 `model.xml.gz`。它们用于精确恢复 MuJoCo scene/state 并离线重新渲染观测。
- Atomic 任务不需要分别实现深度逻辑。所有任务共享同一回放路径：用 dataset-level `env_args` 创建任务环境，用 episode MJCF/metadata 恢复场景，再逐帧写入 `states` 并从三路相机渲染 RGB/depth。任务差异已经固化在 MJCF、episode metadata 和 state vector 中。

## 官方实现依据

RoboCasa365 官方数据说明把 `states.npz` 标为 raw MuJoCo states for replay，并明确列出同目录的 episode metadata 和压缩 MJCF：

- https://github.com/robocasa/robocasa/blob/main/docs/datasets/using_datasets.md

官方 LeRobot 转换实现保存的 `states.npz` 只有 `states` 键，同时保存 `ep_meta.json` 和 `model.xml.gz`；读取接口同样按这三个文件恢复：

- https://github.com/robocasa/robocasa/blob/main/robocasa/utils/lerobot_utils.py

官方 HDF5→LeRobot 流程先用 model/episode metadata 恢复初始场景，再对每个 `states[t]` 调用 simulator reset 并提取观测。因此离线 depth 生成应复用 state playback，而不是重放 action：

- https://github.com/robocasa/robocasa/blob/main/robocasa/scripts/dataset_scripts/convert_hdf5_lerobot.py
- https://github.com/robocasa/robocasa/blob/main/robocasa/scripts/dataset_scripts/playback_dataset_hdf5.py

RoboCasa 环境原生支持对每个 camera 设置 `camera_depths=True`，三路相机名称与现有 RoboCasa365 RGB 数据一致：

- https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/kitchen.py

## X-WAM depth 表示

X-WAM 论文明确规定监督目标为 inverse depth；单通道 depth 复制为三通道 pseudo-RGB，再由 Wan VAE 编码。depth branch 是后部 DiT block 的辅助分支，推理时可以关闭：

- https://arxiv.org/pdf/2604.26694

X-WAM 官方 RoboCasa 数据和 loader 的公开合同为：

- 三路 camera 各有独立 depth MP4；
- depth 与 RGB 使用相同 `start/end/fps`；
- depth 文件由 Decord 解码为三通道 8-bit 图像；
- loader 执行 `pixel / 127.5 - 1.0`；
- RoboCasa 配置 `normalize_depths_per_view=false`；
- 官方发布样例为 256×256、20 FPS、H.264/yuv420p 灰度视频。

来源：

- https://github.com/sharinka0715/X-WAM/blob/main/README.md
- https://github.com/sharinka0715/X-WAM/blob/main/data/robot_dataset.py
- https://github.com/sharinka0715/X-WAM/blob/main/configs/data/robocasa.yaml
- https://huggingface.co/datasets/sharinka0715/X-WAM-RoboCasa

公开论文和代码没有给出“MuJoCo metric depth → inverse depth → uint8”的完整数值公式，因此不把任何项目自定义映射描述为上游原公式。P1真实回放已通过后，本项目固定版本`robocasa365_inverse_metric_global_q_v1`：在三个代表性任务各前三个episode、三路相机的抽样帧中合并采样`1/depth_m`，以全局q01/q99作为固定边界，裁剪并映射到uint8，近处更亮、无效值为0。它匹配X-WAM公开的存储和loader合同，但不保证与上游未公开数值分布完全相同。

## RGBD-P0/P1 本轮实现

`scripts/audit_robocasa365_depth_replay_inputs.py` 是 dependency-light 只读门禁，不导入 Torch、RoboCasa 或 MuJoCo。它完成：

1. 解析单任务或 Atomic9 任务清单及唯一日期目录。
2. 检查全部 episode 都存在 `states.npz/ep_meta.json/model.xml.gz`。
3. 每任务默认解压前三个 episode，验证 `states` 为有限二维数值数组且帧数与 `episodes.jsonl` 完全一致。
4. 验证 MJCF gzip 可解析且根标签为 `mujoco`。
5. 验证 dataset-level `env_args.env_name` 和 X-WAM 三路 camera 合同。
6. 报告 state width、缺失文件和下一阶段 render probe 要求。state width是per-episode MJCF属性，不再用跨episode一致性阻塞。

`scripts/probe_robocasa365_depth_render.py` 是RoboCasa runtime门禁，不导入Torch也不加载X-WAM模型。它对每个抽查episode执行：

1. 从`dataset_meta.env_args` 创建任务环境。
2. 设置`ep_meta.json`，硬重置该episode自身`model.xml.gz`。
3. 只将`states.shape[1]`与当前MJCF的展平state宽度比较，写入首/中/尾帧并验证回读偏差。
4. 对三路camera同时渲染RGB和normalized depth，纵向翻转后与原MP4同帧比较。
5. 使用robosuite的near/far公式保存metric depth，验证有限、正值和非常量。
6. 保存source/rerender/depth NPZ和三联PNG，供人工复核。PNG的per-frame inverse-depth归一化只是诊断预览，不是训练公式。

当前Clariden入口一次性执行结构审计和上述render probe：Atomic9每任务前3个episode，每episode三个时间点，每点三路camera。默认RGB MAE门限为12；用户已回报这轮全部检查通过且无错误，但未提供Job ID，因此只关闭P1功能门禁，不补造作业编号。

## RGBD-P2 编码、缓存与审计

`scripts/build_and_audit_robocasa365_depth_cache.py`在同一作业内依次完成：

1. 对`CloseFridge`、`PickPlaceSinkToCounter`、`OpenDrawer`各前三个episode执行逐MJCF/state标定，默认每10帧、每相机采4096个有效逆深度像素，写入不可变encoding JSON。
2. 以冻结范围逐帧渲染三个任务的全部9个episode和三路相机，生成27个256×256、20 FPS、H.264/yuv420p三通道灰度MP4；不改写原始LeRobot目录。
3. 每个视频写sidecar，绑定encoding、states/MJCF/episode metadata、源RGB和输出视频digest。中断重提会验证并跳过完整相机，只原子补写缺失相机；无sidecar的半成品不会被接受。
4. 自动检查帧数、FPS、相机key、uint8 shape/range、灰度通道一致性、H.264往返MAE、无效/裁剪像素、生成吞吐、总字节数和每帧字节数，统一输出cache manifest和audit JSON。

Clariden入口为`deployment/clariden/build_atomic3_rgbd_pilot_cache_xwam.sbatch`。用户已回报三项P2任务全部通过且无报错；本轮未提供Job ID，因此不补造编号。

## RGBD-P3 Loader、完整CloseFridge缓存与短训练

顺序不能颠倒：

1. 先运行`deployment/clariden/build_close_fridge_rgbd_cache_xwam.sbatch`。它复用P2冻结encoding，不重新计算q01/q99；对CloseFridge全部106个episode生成318个相机视频，支持按episode/camera恢复，输出独立manifest和audit。
2. 完整缓存PASS后运行`deployment/clariden/smoke_close_fridge_rgbd_train_resume_xwam.sbatch`。作业先解码一个真实batch，检查RGB/depth均为`[1,3,9,3,256,320]`、有限且在`[-1,1]`，再加载公开X-WAM pretrained执行4×GH200 step 0→2。
3. step 2必须保存完整model/四rank optimizer checkpoint，并从该精确目录恢复到step 4。机器审计要求depth loss每步有限且大于0、action/proprio监督比例为1、全部loss有限、四rank optimizer为FP32且resume来源一致。

P3只使用固定8个CloseFridge clip、GBS4、ZeRO-2 CPUAdam和debug checkpoint策略，目的是关闭loader/forward/backward/resume工程门禁；不作为正式RGB-D超参或性能结论。完整缓存及短训练真实结果均为`cluster-pending`。

## 后续顺序

1. `P1-structure + P1-render`：已由用户回报全部通过。
2. `P2-encoding + pilot cache + audit`：用户回报三项检查全部通过。
3. `P3-full CloseFridge cache + loader + batch + short resume`：代码已完成，等待按上述两个作业顺序运行。
4. CloseFridge ratio0 RGB-D 单任务试验；通过后再运行 Atomic9 ratio0 RGB-D。

任何 P1 render 对齐失败都优先修复 scene/state/camera 恢复，不进入全量缓存生成。
