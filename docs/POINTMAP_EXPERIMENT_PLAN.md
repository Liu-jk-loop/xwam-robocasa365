# X-WAM × RoboCasa365 PointMap 实验计划

## 目标边界

- 只处理 Atomic9，不涉及 composite。
- PointMap 使用相机坐标系米制 XYZ，不使用相机外参。
- 分两层实现：先做 `X-WAM-PointMap-Aux` 辅助监督，再做带可选 PointMap 输入和 cross-modality forcing 的 `X-WAM-PointMap-Flex`。
- 第一层推理保持 RGB-only；第二层分别评测 RGB-only 和 RGB+PointMap。
- 现有 inverse-depth H.264 缓存不作为 PointMap 正式来源。

## 冻结数值合同

- 在原生 `256×256` metric depth 上使用每路相机内参反投影：`X=(u-cx)/fx*Z`、`Y=(v-cy)/fy*Z`、`Z=depth_m`。
- 有效深度严格为 `0.01m < Z < 2.0m`；无效 XYZ 为零。
- XYZ 分别裁剪到 `[-0.5,0.5]`、`[-0.5,0.5]`、`[0,1.5]` 米，再映射到 `[-1,1]`。
- 反投影后以 nearest resize 到 `256×320`。
- 正式缓存为每 episode、每 camera 一个未压缩 `.npy`，张量为 `[T,3,256,320] float16` normalized XYZ，可 memory-map 随机读取。
- Atomic9 按 206,042 帧估算约 283 GiB；第一阶段只写少量 NPZ 审计产物，不生成全量缓存。

数值定义对齐 Flex-π 官方 `PointmapEncoder` commit `20c1b2b71ea35a415d5d47c39b04443cfadad7a1`。本项目不复制 Flex-π 的完整 DINO/MoT 架构，也不把第一层辅助监督描述为完整 Flex-π。

## 待办流程

- [x] P0：CloseFridge 前3个episode、首/中/尾帧，审计三路相机内参、RGB对齐、depth↔XYZ重投影、范围和float16误差；Clariden job `3254392`通过。
- [x] P0.1：CloseBlenderLid 3个episode首/中/尾帧完成同状态对照；job `3259701`确认`forced_opaque_required`，原始RGB保持不变。
- [x] P1：用户回报CloseFridge 106 episode/318数组生成与最终审计全部PASS；Job ID未提供，不补造。
- [ ] P2：memory-map loader、RGB/PointMap同步augmentation及真实batch耗时报告已实现；等待Clariden门禁证据。
- [ ] P3：`PointMap-Aux`损失、单batch、optimizer update和step 2→4恢复门禁已实现；等待4×GH200运行。
- [ ] P4：以相同初始化、ratio0、GBS和评测seed完成CloseFridge inverse-depth/PointMap对照。
- [ ] P5：生成Atomic9全量PointMap缓存并正式训练、评测。
- [ ] P6：增加 `PointMap-Flex` 真输入流、stream dropout和cross-modality forcing。

## P0 验收证据

- 报告必须记录Git commit和dirty状态，但dirty状态只用于provenance，不阻塞作业。
- 1个CloseFridge任务、3个episode、每episode 3帧、每帧3路相机均通过。
- 三路相机各自的内参在全部抽样episode/frame中稳定，且相机数量严格为3。
- PointMap含有效像素、无NaN/Inf，depth和像素重投影误差在阈值内。
- normalized float16 PointMap shape为`[3,256,320]`，范围在`[-1,1]`，反量化米制误差不超过2 mm。
- 输出JSON、每帧NPZ和RGB/重渲染RGB/inverse-depth/PointMap对比图；本地只能做静态测试，真实MuJoCo/EGL证据为`cluster-pending`。

## P0.1透明表面验收

- 只在诊断期间把`0 < alpha < 1`的可见geom/material临时提升为1；alpha为0的隐藏/碰撞geom保持隐藏，诊断结束必须恢复模型。
- 优先通过`base_env.blender.blender_lid`实体解析目标geom，仅在实体不可用时使用严格blender-lid名称正则兜底，并要求目标像素在forced-opaque segmentation中真实可见。
- 原始与forced-opaque两路PointMap数值合同都必须PASS；比较仅统计目标像素中超过1 mm的新增有效、变近或变远depth。
- `forced_opaque_required`表示正式PointMap应使用forced-opaque depth而RGB保持原始；`original_depth_matches_forced_opaque`表示原始depth可直接使用。其他状态均为证据不足，不进入P1。

## P1完整缓存验收

- 正式生成必须先读取并校验P0.1 PASS JSON及SHA256，且结论必须为`forced_opaque_required`、策略必须为`use_forced_opaque_depth_for_pointmap_keep_original_rgb`。随后每个episode加载自己的MJCF、state和metadata；每帧只恢复一次state，再为三路相机渲染forced-opaque metric depth。原始RGB文件不得改写。
- 每个episode/相机原子发布一个未压缩`.npy [T,3,256,320] float16`。`.npy`和sidecar均完整且源摘要、合同摘要、shape、dtype和数组SHA256一致时才允许断点跳过。
- 缺失或不一致的缓存只重建对应episode/相机；中断生成的`.partial.npy`不视为有效产物。重提同一作业必须复用已通过的完整文件。
- 最终manifest严格覆盖CloseFridge 106 episode、318数组；独立audit以memory-map重新打开每个文件，检查有限值、`[-1,1]`、帧覆盖、三相机覆盖、sidecar一致性及SHA256。

## P2/P3单任务训练门禁

- Dataset启动只验证PASS manifest/audit、sidecar和全部`.npy` header，不对约40 GiB缓存重复做全量SHA256；worker按9帧clip使用read-only mmap读取并转float32。
- RGB和PointMap共用逐相机、跨时间一致的crop参数；RGB和连续XYZ使用bilinear，颜色抖动只作用于RGB。
- PointMap复用X-WAM第二生成模态作为辅助目标，独立记录`train/pointmap_loss`；`use_depth`与`use_pointmap`互斥，PointMap不是policy在线输入。
- 4×GH200门禁固定8个clip，必须完成step 0→2保存和step 2→4严格恢复，并验证正PointMap loss、四rank FP32 optimizer实态和完整checkpoint。
- 门禁通过后从公开X-WAM pretrained重新初始化正式实验：2节点×4 GH200、单卡batch4、累积4、GBS128、ratio0、1500 optimizer steps；滚动每250步最多5个，Store永久保存500/1000/1500步。
- 正式入口还要求P3审计中的训练commit与当前checkout完全一致，旧commit的PASS报告不能解锁新代码训练。
