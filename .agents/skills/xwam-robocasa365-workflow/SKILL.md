---
name: xwam-robocasa365-workflow
description: Manage X-WAM adaptation, debugging, training, evaluation, documentation, and Git handoff for RoboCasa365 atomic tasks. Use when planning or changing dataset adapters, action/state schemas, checkpoint loading, RGB/depth handling, cluster configurations, Slurm runs, simulator evaluation, experiment records, or when diagnosing feedback returned from the Starlight cluster.
---

# X-WAM × RoboCasa365 工作流

## 建立上下文

1. 阅读仓库 `AGENTS.md`。
2. 阅读 `docs/IMPLEMENTATION_PLAN.md`、`docs/ARCHITECTURE.md`、`docs/PROGRESS.md` 和 `docs/CHANGELOG.md` 最新记录。
3. 检查当前分支、commit、remote、工作区状态和相关 diff，保留用户已有的无关修改。
4. 每次只选择一个阶段子任务，修改前明确它的验收证据。

除非用户明确扩大范围，否则始终保持 atomic-only；composite 数据进入运行时必须视为错误。

## 划分验证级别

每项验收检查必须归入以下一种：

- `local-static`：不需要导入 Torch 或使用 GPU。
- `cluster-smoke`：需要 Torch/CUDA、真实数据、checkpoint 或模拟器。
- `cluster-train`：需要多 GPU 正式训练或计划评测。

本地工作站没有可用的 Torch runtime。未执行的运行时检查必须标记为 `cluster-pending`，不得从语法验证推断运行成功。

## 通过 adapter 隔离实现

- RoboCasa365 数据访问必须封装在 dataset 和 observation adapter 后面。
- 动作分量名称、归一化和环境打包必须封装在 action codec 后面。
- legacy 到目标参数的处理必须封装在 checkpoint adapter 后面，并输出明确报告。
- 任务注册、rollout 记录和指标聚合必须封装在 benchmark adapter 后面。
- depth 模式只能明确设为 `disabled` 或 `cached`，训练 worker 内禁止渲染深度。
- 固定 action、state、camera、时序或任务假设前必须读取官方元数据。

## 验证并记录

1. 每次逻辑修改都增加或更新聚焦的测试/fixture。
2. 在本地运行无依赖检查，适用时包含 Python 语法编译。
3. 运行 `python .agents/skills/xwam-robocasa365-workflow/scripts/check_change_record.py --base main`。
4. 在 `docs/CHANGELOG.md` 中用中文记录问题、逻辑、文件、兼容性、验证、超算状态、风险和回滚方法。
5. 在 `docs/PROGRESS.md` 中用中文记录阶段状态、证据、阻塞项和下一步。
6. 只有对应实现已经存在时，才能加入精确的超算命令。

## 审查分布式作业边界

任何包含 `srun ... env ... bash -lc '...'` 的作业都必须把节点shell读取的外层变量逐项显式传入，不能依赖提交shell的隐式继承。

1. 修改 `deployment/clariden/*.sbatch` 后，运行 `python .agents/skills/xwam-robocasa365-workflow/scripts/check_slurm_env_contract.py`。
2. 新增外层变量时，核对“外层定义 → `srun env` 传递 → 节点内读取”三处名称完全一致。
3. 对影响训练入口、恢复、停止步数、checkpoint或产物路径的变量增加回归测试；测试必须证明漏传时检查失败、显式传递后通过。
4. `bash -n` 只能作为语法检查，不能替代变量边界审计。
5. 发布前的 `check_change_record.py` 会再次运行该检查；任何失败都禁止提交超算正式作业。

超算反馈使用 `references/cluster-feedback-template.md`；只针对模板记录的 commit 和解析后配置进行诊断。

## 对高成本作业执行双遍审查

第一遍按用户约束核对任务集、初始化、RGB/RGB-D、节点/GPU/GBS、训练步数、checkpoint保留、分支和路径，不能凭上一轮记忆补值。第二遍从最终diff重新审查进程边界、配置与checkpoint分离、恢复来源、停止条件、持久化目录及失败报告；此遍不得把“已有测试覆盖”当作已核对。正式启动命令只能在两遍审查和全部`local-static`门禁通过后交付，仍需集群验证的内容明确标记为`cluster-pending`。

## 发布与交接

1. 检查完整 diff，确认只包含本次预期文件。
2. 在 `dev/atomic-robocasa365` 创建聚焦的 commit。
3. 推送到 `origin`，禁止推送到 `upstream`。
4. 交接 commit SHA、超算命令、预期证据和已知 `cluster-pending` 检查。
5. 收到反馈后先把证据写入变更/进度记录，再宣布阶段完成。
