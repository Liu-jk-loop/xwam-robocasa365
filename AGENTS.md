# X-WAM RoboCasa365 repository instructions

## Scope

- Adapt X-WAM to RoboCasa365 atomic tasks only.
- Exclude composite-task training, evaluation, history mechanisms, and task decomposition until the user expands the scope.
- Treat the 65 atomic human-data tasks as the atomic-only training pool and Atomic-Seen 18 as the initial target evaluation set. Keep the exact task lists configurable.

## Development workflow

- Develop on `dev/atomic-robocasa365`; keep `main` stable.
- Preserve `upstream` as fetch-only. Never push to the official X-WAM repository.
- Keep data, model weights, generated depth, checkpoints, videos, and full logs out of Git.
- Do not hard-code machine paths in Python. Put cluster paths in configuration or command-line overrides.
- Keep model inference and RoboCasa simulation separable so they can use different Python environments.

## Required documentation

- Read `docs/IMPLEMENTATION_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PROGRESS.md` before changing implementation logic.
- Update `docs/CHANGELOG.md` for every source, configuration, workflow, or dependency change.
- Update `docs/PROGRESS.md` whenever a milestone, validation state, blocker, or next action changes.
- Mark changes that cannot be executed locally as `cluster-pending`; do not call them complete before cluster evidence is recorded.

## Validation

- The local workstation has no usable Torch runtime. Run syntax, schema, documentation, and dependency-free tests locally.
- Run Torch, CUDA, DeepSpeed, simulator, dataset, training, and rollout validation on the Starlight cluster.
- Record the tested Git commit, command, configuration, environment, GPU, logs, and outcome for every cluster validation.
- Prefer a small deterministic smoke test before any multi-GPU training job.
- Before publishing any `deployment/clariden` change, run the Slurm environment-contract checker from the repository workflow skill. `bash -n` alone does not validate variables crossing an explicit `srun env` boundary.

## Implementation boundaries

- Add explicit dataset, observation, action, checkpoint, and benchmark adapters instead of spreading RoboCasa365 conditionals through the model.
- Read action/state/camera schemas from official dataset metadata where available. Validate dimensions and fail loudly on mismatch.
- Keep depth modes explicit: `disabled` or `cached`. Never render depth inside the training data loader.
- Do not silently discard PandaOmron base or control-mode actions.
- Preserve both initialization tracks: X-WAM cross-embodiment pretrained and Wan2.2-base-only ablation.
