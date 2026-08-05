---
name: xwam-robocasa365-workflow
description: Manage X-WAM adaptation, debugging, training, evaluation, documentation, and Git handoff for RoboCasa365 atomic tasks. Use when planning or changing dataset adapters, action/state schemas, checkpoint loading, RGB/depth handling, cluster configurations, Slurm runs, simulator evaluation, experiment records, or when diagnosing feedback returned from the Starlight cluster.
---

# X-WAM RoboCasa365 workflow

## Establish context

1. Read repository `AGENTS.md`.
2. Read `docs/IMPLEMENTATION_PLAN.md`, `docs/ARCHITECTURE.md`, `docs/PROGRESS.md`, and the latest `docs/CHANGELOG.md` entry.
3. Inspect the current branch, commit, remotes, status, and relevant diff. Preserve unrelated user changes.
4. Select one milestone task and state its acceptance evidence before editing.

Keep work atomic-only unless the user explicitly expands scope. Treat composite data entering a run as an error.

## Classify validation

Classify every acceptance check as one of:

- `local-static`: no Torch import or GPU required.
- `cluster-smoke`: Torch/CUDA, real dataset, checkpoint, or simulator required.
- `cluster-train`: multi-GPU training or scheduled evaluation required.

The local workstation has no usable Torch runtime. Mark unexecuted runtime checks `cluster-pending`; never infer success from syntax validation.

## Implement behind adapters

- Keep RoboCasa365 access behind dataset and observation adapters.
- Keep named action components, normalization, and environment packing behind an action codec.
- Keep legacy-to-target parameter handling behind a checkpoint adapter with explicit reports.
- Keep task registration, rollout records, and metric aggregation behind a benchmark adapter.
- Keep depth mode explicit as `disabled` or `cached`; never render depth inside training workers.
- Read official metadata before fixing action, state, camera, timing, or task assumptions.

## Validate and document

1. Add or update focused tests/fixtures with each logic change.
2. Run dependency-free checks locally, including Python compilation where applicable.
3. Run `python .agents/skills/xwam-robocasa365-workflow/scripts/check_change_record.py --base main`.
4. Update `docs/CHANGELOG.md` with problem, logic, files, compatibility, validation, cluster status, risks, and rollback.
5. Update `docs/PROGRESS.md` for milestone state, evidence, blockers, and next action.
6. Add exact cluster commands only when the corresponding implementation exists.

For cluster feedback, use `references/cluster-feedback-template.md`. Diagnose only against its recorded commit and resolved configuration.

## Publish and hand off

1. Review the complete diff and confirm only intended files are included.
2. Commit on `dev/atomic-robocasa365` with a focused message.
3. Push to `origin`; never push to `upstream`.
4. Hand off the commit SHA, cluster command, expected evidence, and known `cluster-pending` checks.
5. On feedback, append evidence to the change/progress records before declaring the milestone complete.
