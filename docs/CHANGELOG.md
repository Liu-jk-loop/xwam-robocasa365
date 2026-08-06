# 变更记录

## 2026-08-06 — M1 集群证据与星光环境复用审计

- 分支：`dev/atomic-robocasa365`
- 基线 commit：`219916c`
- 运行状态：真实 RoboCasa365 metadata audit 通过；X-WAM runtime 环境待审计

### 问题

星光已在 `abot_m05` 和 A800 上成功审计一份真实 `CloseFridge` 数据，但反馈缺少 Git commit 和完整 Python/Torch/DeepSpeed/FlashAttention 环境信息。直接修改 `abot_m05` 或全量重装可能破坏已有 DeepSpeed/FlashAttention 二进制扩展。

### 新增和修改逻辑

- 记录 `CloseFridge/20250819` 的真实数据契约证据：106 episodes、26888 frames、16D state、12D action、106 个 Parquet、三路相机各 106 个 MP4，`ok=true`。
- 增加版本化的星光 X-WAM policy 环境 manifest，固定 Python/依赖边界、simulator 独立环境契约和 CUDA 12.8 后备版本。
- 增加无侵入环境 audit：检查包版本、隔离 runtime import、Torch CUDA/GPU、`nvidia-smi`、`nvcc`、`pip check`、DeepSpeed `ds_report`、Git commit 和子模块 gitlink。
- 增加 `abot_m05` 先审计、后 clone、只按报告补依赖的中文方案。
- 明确禁止预编译全部 DeepSpeed ops；当前使用 `torch.optim.AdamW` 且未启用 CPU/NVMe offload。
- 修正 M1 数据 audit 示例，使其指向包含日期层的单任务目录。

### 涉及文件

- 环境契约：`configs/environment/xwam_starlight.json`。
- 环境审计：`project_tools/starlight_environment.py`、`scripts/audit_starlight_environment.py`。
- 测试：`tests/test_starlight_environment.py`。
- 文档：`docs/ENVIRONMENT_PLAN.md`、`docs/CLUSTER_RUNBOOK.md`、`docs/IMPLEMENTATION_PLAN.md`、`docs/PROGRESS.md`。

### 兼容性和风险

- audit 不安装包、不修改 Conda 环境，也不主动编译 DeepSpeed op。
- runtime import 在隔离子进程中执行，坏 ABI 不会直接终止主审计进程。
- simulator 依赖继续与 X-WAM policy 环境分离。
- 本地没有 Torch/GPU，真实 runtime 验证保持 `cluster-pending`。
- 数据 audit 的 commit 尚未提供，因此证据先标记为“通过、commit 待补”。

### 验证

- 环境契约/版本比较单元测试：4 项通过。
- 既有 RoboCasa365 数据契约测试：6 项通过。
- 环境 audit 本地负路径：正确返回非零并生成可解析 JSON。
- CLI help、Python compile 和 Git diff check：通过。
- `abot_m05` 环境 audit：`cluster-pending`。

### 回滚

回退本次 commit；不会修改服务器 Conda 环境、外部数据、权重或运行任务。

## 2026-08-05 — 各阶段执行过程中文化

- 分支：`dev/atomic-robocasa365`
- 基线 commit：`1006d72`
- 运行状态：纯文档和 workflow 修改；本地验证通过

### 问题

原计划虽然已经列出中文工作内容和验收条件，但没有逐阶段说明本地修改、超算执行、反馈证据和阶段退出的完整过程。进度表、集群手册和反馈模板仍混有大量英文，不便于持续协作。

### 新增和修改逻辑

- 为 M0～M7 每个阶段增加中文“阶段执行过程”，明确用户输入、Codex 修改、星光运行、反馈证据和进入下一阶段的条件。
- 将项目进度表、当前执行过程和资源/输入状态改为中文。
- 将星光超算操作手册改为中文，并补充 M1 audit 的反馈清单。
- 将超算反馈模板改为中文，保留 `pass`、`fail`、`blocked` 等可检索状态值及中文解释。
- 将项目 workflow skill 的操作说明改为中文，并要求后续变更和进度使用中文记录。

### 涉及文件

- `docs/IMPLEMENTATION_PLAN.md`
- `docs/PROGRESS.md`
- `docs/CLUSTER_RUNBOOK.md`
- `.agents/skills/xwam-robocasa365-workflow/SKILL.md`
- `.agents/skills/xwam-robocasa365-workflow/references/cluster-feedback-template.md`

### 兼容性和风险

- 不修改模型、数据、训练或评测运行逻辑。
- 命令、配置键、代码符号和验证状态标签保持原样，避免翻译导致脚本不可执行或状态不可检索。

### 验证

- Markdown/diff 格式检查：通过。
- Skill 结构验证：通过。
- 文档同步检查：通过。

### 回滚

回退本次文档 commit；不会影响外部数据、模型或超算任务。

## 2026-08-05 — M1 batch 1: 数据契约与 RGB-only 基础

- 分支：`dev/atomic-robocasa365`
- 基线 commit：`bc0e136`
- 运行状态：本地静态验证通过；真实数据和 Torch 集群验证待执行

### 问题

当前训练入口只支持旧 X-WAM JSON+视频格式；缺少 RoboCasa365 官方 LeRobot 元数据门禁。旧 `RobotDataset` 即使模型设置 `use_depth=false`，仍会要求 depth 目录和每个视角的 `depth_path`，augmentation 也强制读取 `depths`。

### 新增和修改逻辑

- 增加官方 Target Atomic-Seen 18 的版本化任务清单。
- 增加无 Torch 的 RoboCasa365 metadata audit，校验 16D state、12D action、三相机、episodes 元数据和 atomic-only 任务归属。
- 同时支持传入直接 LeRobot 路径或包含 `lerobot/` 的任务目录。
- 识别 LeRobot v2 `episodes.jsonl` 和 v3 分块 episodes 元数据，并在尚未支持逐 episode 审计时给出明确提示。
- 修改旧 `RobotDataset`：`use_depth=false` 时不要求 depth 目录、不要求 `depth_path`、不读取 depth 视频，也不向 batch 写入 `depths`。
- 修改视频 augmentation，使深度张量可选但 RGB crop/color jitter 保持可用。
- 将训练和验证数据集的 `use_depth` 与模型配置打通。
- 将完整 M0-M7 计划改为中文版，并在集群手册解释 submodule 操作。

### 兼容性和风险

- `RobotDataset` 默认仍为 `use_depth=true`，保持旧训练行为。
- 本批只建立原生数据契约和审计入口，尚未完成 Parquet 到 X-WAM tensor 的正式 loader。
- 真实官方数据布局和媒体数量必须在星光上验证。
- 本地无 Torch，augmentation 和旧 loader 的运行时验证标记为 `cluster-pending`。

### 涉及文件

- 数据契约与审计：`data/robocasa365_contract.py`、`scripts/audit_robocasa365_dataset.py`。
- 任务范围：`configs/tasks/robocasa365_atomic_seen.json`。
- RGB-only 修复：`data/robot_dataset.py`、`data/augmentation.py`、`scripts/train_sft.py`。
- 测试：`tests/test_robocasa365_contract.py`。
- 计划与进度：`docs/IMPLEMENTATION_PLAN.md`、`docs/ARCHITECTURE.md`、`docs/CLUSTER_RUNBOOK.md`、`docs/PROGRESS.md`。

### 验证

- 纯 Python 数据契约单元测试：6 项通过。
- Audit CLI 合成数据验证：通过，输出可解析 JSON。
- Skill / 文档检查 / Python compile / Git diff check：通过。
- M1 batch-1 commit `83210c2` 已发布到 `origin/dev/atomic-robocasa365`。
- 真实数据 metadata audit：`cluster-pending`。
- RGB-only 旧 loader：`cluster-pending`。

### 回滚

回退本 M1 commit；默认 depth 行为未改变，不涉及外部数据写入。

## 2026-08-05 — M0 repository workflow baseline

- Branch: `dev/atomic-robocasa365`
- Base commit: `72cfb86`
- Runtime status: local static validation passed; cluster validation pending

### Problem

The fork had no repository-specific collaboration rules, implementation roadmap, progress record, cluster feedback contract, or documentation enforcement. The existing global `*.sh` ignore rule would also prevent future Slurm and launch scripts from being versioned.

### Added logic

- Defined atomic-only scope, local/cluster ownership, validation states, and adapter boundaries.
- Added a milestone plan from native RoboCasa365 loading through atomic-only training and evaluation.
- Added a repository workflow skill that guides future modifications and cluster feedback handling.
- Added a deterministic Git-diff check requiring change and progress documentation for implementation/workflow changes.
- Included untracked files in the documentation check so newly created implementation files cannot bypass it.
- Added Starlight model locations, checkout procedure, validation levels, and a structured debug report template.
- Removed the global shell-script ignore rule so future cluster scripts can be committed intentionally.

### Compatibility and risk

- No model, dataset, training, or evaluation runtime behavior is changed in M0.
- The local machine cannot validate Torch/CUDA behavior.
- The repository's third-party submodules are not initialized locally.

### Validation

- Skill structure: passed with the skill-creator validator.
- Documentation enforcement script: passed against the complete branch working diff.
- Python syntax compilation: passed for repository and skill Python sources without importing Torch.
- Git whitespace/error check: passed.
- Development branch publication: passed; `dev/atomic-robocasa365` tracks the fork remote.
- Cluster checkout: pending.

### Rollback

Revert the M0 commit. No external data, checkpoint, or simulator state is modified.
