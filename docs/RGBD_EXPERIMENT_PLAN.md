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

公开论文和代码没有给出“MuJoCo metric depth → inverse depth → uint8”的完整数值公式。当前不能自行假定 per-frame min-max、per-view min-max 或固定 near/far 映射。正式生成前必须用实际 state replay probe 冻结该公式；否则虽能训练，depth pretrained 权重的输入分布可能不匹配。

## RGBD-P0/P1 本轮实现

`scripts/audit_robocasa365_depth_replay_inputs.py` 是 dependency-light 只读门禁，不导入 Torch、RoboCasa 或 MuJoCo。它完成：

1. 解析单任务或 Atomic9 任务清单及唯一日期目录。
2. 检查全部 episode 都存在 `states.npz/ep_meta.json/model.xml.gz`。
3. 每任务默认解压前三个 episode，验证 `states` 为有限二维数值数组且帧数与 `episodes.jsonl` 完全一致。
4. 验证 MJCF gzip 可解析且根标签为 `mujoco`。
5. 验证 dataset-level `env_args.env_name` 和 X-WAM 三路 camera 合同。
6. 报告 state width、缺失文件和下一阶段 render probe 要求。

结构门禁通过只说明离线回放材料齐全，不等于 RGB/depth 已对齐。下一阶段必须在 RoboCasa simulator 环境运行实际 reset/render probe。

## 后续顺序

1. `P1-structure`：Atomic9 全 episode 文件覆盖 + 每任务三个 state 数组抽查。
2. `P1-render`：三个代表任务各一个 episode、三个时间点，恢复 MJCF/state 并渲染三路 RGB/depth。
3. 对原 RGB 计算像素对齐指标，并确定 vertical flip、near/far、inverse-depth 和 uint8 编码公式。
4. 生成少量可恢复 depth cache，统计速度、无效像素和磁盘占用。
5. 接入 loader、checkpoint depth branch 和单 batch/resume 门禁。
6. CloseFridge ratio0 RGB-D 单任务试验；通过后再运行 Atomic9 ratio0 RGB-D。

任何 P1 render 对齐失败都优先修复 scene/state/camera 恢复，不进入全量缓存生成。
