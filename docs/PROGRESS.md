# 项目进度

更新时间：2026-08-05

## 当前状态

- 当前分支：`dev/atomic-robocasa365`
- 当前阶段：M1——RoboCasa365 数据契约与原生 loader
- 本地运行能力：没有可用 Torch，只执行静态验证
- 超算运行状态：尚未验证
- 任务范围：只包含 atomic，排除 composite

## 阶段状态

| 阶段 | 当前状态 | 超算状态 | 下一门禁 |
| --- | --- | --- | --- |
| M0 工程与协作基线 | 已完成 | 主仓库已 clone | 模拟器阶段开始时确认第三方子模块 |
| M1 原生 RoboCasa365 loader | 第一批已发布（`83210c2`） | 待验证 | 对一个真实 atomic 数据集运行 metadata audit |
| M2 动作与 checkpoint 适配 | 未开始 | 待验证 | 冻结官方 PandaOmron schema |
| M3 RGB-only 训练烟测 | 未开始 | 待验证 | A100/A800 单 batch forward/backward |
| M4 闭环评测器 | 未开始 | 待验证 | 完成一个 atomic 闭环 rollout |
| M5 离线深度试点 | 未开始 | 待验证 | 完成 1～3 个任务的对齐缓存 |
| M6 Atomic 正式训练与评测 | 未开始 | 待验证 | 通过 H100 正式训练门禁 |
| M7 复现与维护 | 未开始 | 待验证 | clean clone 完整复现 |

## 已确认资源

- 已有 Wan2.2-TI2V-5B 路径可以复用，不再重复下载。
- X-WAM 公开 cross-embodiment pretrained checkpoint 已下载或正在约定的 X-WAM 模型目录下完成下载。
- A100/A800 用于调试，H100 用于正式训练。

## 待提供输入

- 按 `docs/CLUSTER_RUNBOOK.md` 中的路径检查确认 X-WAM pretrained 文件。
- 数据可用后提供 RoboCasa365 根目录。
- M1 超算验证需要一个最小真实 episode、对应元数据及其引用的三路 RGB 视频。

## 当前执行过程

1. 用户在星光超算拉取最新开发分支，并记录 `git rev-parse HEAD`。
2. 用户对一个真实 RoboCasa365 atomic 数据集运行 metadata audit。
3. 用户反馈 audit 命令、JSON 报告和实际目录结构。
4. Codex 根据报告实现原生 Parquet tensor adapter，并同步测试、变更记录和进度。
