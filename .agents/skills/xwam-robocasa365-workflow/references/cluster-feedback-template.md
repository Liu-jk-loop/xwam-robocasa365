# 星光超算反馈模板

## 运行标识

- 日期、时间和时区：
- Git commit SHA：
- 分支：
- Slurm/Kubernetes Job ID 或 Pod ID：
- 执行人：

## 本次目标

- 阶段和子任务：
- 预期验收证据：
- 结果：`pass`（通过）、`fail`（失败）或 `blocked`（受阻）

## 命令与配置

- 工作目录：
- 完整命令：
- 解析后的配置路径或随附配置：
- 数据集根目录及 manifest/任务子集：
- Wan2.2 路径：
- X-WAM checkpoint 路径和初始化模式：
- 输出目录：

## 运行环境

- 主机或容器镜像：
- Python:
- PyTorch:
- CUDA runtime：
- NVIDIA driver：
- GPU 型号、数量和显存：
- DeepSpeed:
- FlashAttention:
- 模拟器任务使用的 RoboCasa/robosuite/MuJoCo 版本：

## 实际运行情况

- 到达的运行阶段：
- 与问题相关的 tensor shape：
- GPU 已分配/预留显存峰值：
- 吞吐或单步延迟：
- loss/指标样例：
- 最后一个成功操作：
- 第一个失败操作：

## 日志

条件允许时附上完整调度日志；否则提供完整 traceback 和第一次异常前至少 100 行日志。不要包含密码、token 或其他敏感信息。

## 产物

- 保存的 checkpoint、结果和视频路径：
- 随附的小型诊断文件：
- 使用相同命令和 commit 是否可以复现：
- 是否在超算 checkout 中做过未提交的手工修改：
