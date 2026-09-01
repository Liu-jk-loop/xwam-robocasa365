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

- [ ] P0：CloseFridge 前3个episode、首/中/尾帧，审计三路相机内参、RGB对齐、depth↔XYZ重投影、范围和float16误差。
- [ ] P1：实现可恢复的完整CloseFridge PointMap `.npy` 生成器和manifest/sidecar。
- [ ] P2：实现memory-map loader、同步augmentation、真实batch和吞吐审计。
- [ ] P3：实现 `PointMap-Aux`，完成单batch、optimizer update和resume门禁。
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
