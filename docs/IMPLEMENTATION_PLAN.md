# X-WAM × RoboCasa365 atomic-only implementation plan

## Objective

Build a reproducible path from native RoboCasa365 atomic-task data to X-WAM training and closed-loop evaluation on Starlight. Use A100/A800 for debugging and H100 for training. Composite tasks are explicitly out of scope.

## Fixed experiment scope

- Training pool: the 65 RoboCasa365 atomic tasks from the human pretraining data, selectable by task manifest.
- Target evaluation: Atomic-Seen 18.
- First smoke subset: three tasks covering pick/place, articulated-object interaction, and appliance interaction.
- Navigation: `NavigateKitchen` remains part of the atomic target, but enters after the full PandaOmron action path is validated.
- Primary initialization: X-WAM cross-embodiment pretrained checkpoint.
- Ablation initialization: Wan2.2-TI2V-5B with newly added X-WAM modules initialized by code.
- First modality baseline: RGB-only. Cached depth is a later, separately gated phase.

## Ownership split

### Local/Codex responsibilities

- Implement adapters, configuration, tests, cluster commands, and documentation.
- Perform dependency-free validation and review every diff.
- Commit and push tested changes to the development branch.
- Diagnose cluster feedback against the exact tested commit.

### Cluster/user responsibilities

- Maintain datasets, assets, model weights, Python environments, and scheduler access.
- Run the commands committed for each validation gate.
- Return the commit SHA, command, environment, GPU information, job ID, logs, and outcome.
- Run offline data scans and depth generation jobs written by Codex. No manual depth annotation is expected.

## Milestones and acceptance gates

### M0 — Engineering and collaboration baseline

Tasks:

- Establish branch/remotes and repository rules.
- Add architecture, plan, progress, changelog, cluster runbook, and feedback template.
- Add a repository workflow skill and deterministic documentation check.
- Record external model locations without duplicating model data.

Acceptance:

- Skill validation passes.
- Documentation check passes for the branch diff.
- Python sources compile without importing Torch.
- Development branch is published to the fork.

### M1 — RoboCasa365 dataset contract and native loader

Tasks:

- Capture the exact versioned `meta/` files from one official RoboCasa365 dataset checkout.
- Implement a native LeRobot/Parquet dataset adapter; do not convert the full dataset to legacy X-WAM JSON.
- Map video keys, timestamps, states, actions, task IDs, episode IDs, and language instructions.
- Make camera selection configuration-driven.
- Make depth truly optional in directory validation, media loading, batching, and augmentation.
- Add an atomic-only task manifest and reject composite samples when atomic-only mode is enabled.
- Add a dataset audit command for episode length, missing media, timing, dimensions, and task counts.

Cluster inputs required:

- RoboCasa365 dataset root.
- A minimal sample containing metadata, one Parquet episode, all referenced RGB videos, and its extras directory.

Acceptance:

- Metadata-only audit succeeds without loading videos.
- One real RGB-only batch has documented shapes and dtypes.
- Repeated indexing is deterministic when augmentation is disabled.
- Missing depth does not cause file or key errors in RGB-only mode.
- Composite samples cannot enter an atomic-only run.

### M2 — Observation/action schemas and checkpoint adaptation

Tasks:

- Introduce `ObservationAdapter` and `ActionCodec` interfaces.
- Derive PandaOmron state/action dimensions and slices from official metadata, then freeze a versioned schema manifest.
- Preserve base motion, control mode, end-effector delta, and gripper components.
- Define normalization, clipping, control-mode encoding, and environment action packing.
- Replace hard-coded single-arm padding and first-seven-dimension execution.
- Add shape-aware checkpoint loading for the legacy 14D X-WAM action head and the RoboCasa365 target schema.
- Support `xwam_pretrained` and `wan_base` initialization modes with explicit load reports.

Acceptance:

- Dataset action → normalized model action → denormalized environment action round-trip passes on recorded samples.
- Every action component is named, dimension-checked, and covered by tests.
- Checkpoint loading reports loaded, remapped, and initialized parameters; unexpected mismatches fail.
- A one-batch forward/backward smoke test passes on one A100/A800.

### M3 — RGB-only atomic training smoke test

Tasks:

- Refactor configuration loading into model, dataset, hardware, and experiment layers.
- Add A100/A800 debug profiles with batch size 1, gradient accumulation, gradient checkpointing, and optional offload.
- Add H100 training profile after memory measurement.
- Add deterministic seeding, resume semantics, checkpoint metadata, and concise logging.
- Run a tiny overfit experiment on one task, then a three-task smoke experiment.

Acceptance:

- The model overfits a deliberately tiny sample and loss decreases consistently.
- Checkpoint save/resume reproduces the next-step behavior within documented tolerance.
- Memory peak and throughput are recorded for A100/A800.
- No depth file is read in RGB-only mode.

### M4 — RoboCasa365 closed-loop evaluator

Tasks:

- Keep policy inference and simulator clients in separate environments connected through the broker.
- Add a versioned Atomic-Seen task registry, horizons, seeds, layouts, and object splits.
- Implement current PandaOmron observation extraction and full action packing.
- Save per-episode metadata, success, termination reason, video, latency, and action diagnostics.
- Add restart-safe result aggregation and missing-rollout detection.

Acceptance:

- A random or scripted policy validates environment creation and result writing.
- One X-WAM checkpoint completes a closed-loop rollout without action-shape errors.
- `NavigateKitchen` exercises nonzero base commands.
- Aggregate success rates reproduce directly from per-episode records.

### M5 — Offline depth pilot

Tasks:

- Reconstruct simulator state from the official episode extras.
- Render selected camera depths offline for one to three tasks.
- Store depth in a versioned cache with episode/frame/camera identifiers and generation metadata.
- Validate RGB/depth/action temporal alignment and depth units/range.
- Benchmark storage and generation throughput before expansion.

Acceptance:

- Pixel-aligned RGB/depth samples pass visual and numeric checks.
- Cached depth indexing matches RGB indexing for every audited frame.
- Training data loading never invokes MuJoCo rendering.
- A short RGB-D training smoke test runs successfully.

### M6 — Atomic-only training and evaluation

Tasks:

- Scan normalization statistics for the selected atomic training set.
- Train the X-WAM-pretrained primary run on H100s.
- Run the Wan-base initialization ablation when budget allows.
- Evaluate all Atomic-Seen tasks with fixed seeds and checkpoint selection rules.
- Report per-task, per-skill, aggregate, latency, and failure-category metrics.

Acceptance:

- Training is restartable and all artifacts reference an immutable Git commit and config snapshot.
- All expected evaluation rollouts are present or explicitly marked failed.
- Atomic-only data provenance is documented; results are not presented as Human300/composite-trained leaderboard results.

### M7 — Reproducibility and maintenance

Tasks:

- Add CI for dependency-free tests and documentation enforcement.
- Add environment lock files after the first validated cluster installation.
- Document upstream synchronization and compatibility checks.
- Produce a final runbook from clean clone to reported metrics.

Acceptance:

- A clean cluster checkout can reproduce the validated smoke run using only committed instructions and external data/weight paths.

## Cross-cutting risks

| Risk | Mitigation |
| --- | --- |
| Legacy X-WAM action head is 14D while RoboCasa365 differs | Use explicit schema adapters and shape-aware loading; never truncate silently. |
| Existing loader requires depth even when disabled | Make modality selection drive validation, loading, batching, and loss construction. |
| Local workstation cannot run Torch | Separate local static gates from cluster runtime gates and record `cluster-pending`. |
| A100 40GB may not fit the debug path | Start with batch 1, gradient checkpointing, ZeRO-3/offload, and measure before fixing a profile. |
| RoboCasa simulator and model dependencies conflict | Preserve brokered inference with separate environments. |
| Atomic-only results are confused with standard Human300 training | Encode dataset scope in manifests, experiment names, and reports. |
| Depth preprocessing becomes a bottleneck | Gate full generation behind a small offline pilot and cache all outputs. |

## Change lifecycle

1. Select one milestone task and define its acceptance evidence.
2. Implement on `dev/atomic-robocasa365` with focused tests.
3. Update `CHANGELOG.md` and `PROGRESS.md` in the same commit.
4. Run local static gates and mark runtime work `cluster-pending`.
5. Push the commit and run the committed cluster command against that SHA.
6. Attach cluster evidence and either close the task or iterate from the same recorded state.
