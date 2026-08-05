# Project progress

Updated: 2026-08-05

## Current state

- Active branch: `dev/atomic-robocasa365`
- Active milestone: M0 — Engineering and collaboration baseline
- Local runtime: no usable Torch; static validation only
- Cluster runtime: not yet validated
- Scope: atomic tasks only; composite tasks excluded

## Milestone status

| Milestone | Status | Cluster status | Next gate |
| --- | --- | --- | --- |
| M0 Engineering baseline | Published | Pending | Verify clean cluster checkout |
| M1 Native RoboCasa365 loader | Not started | Pending | Obtain one minimal official dataset sample and metadata |
| M2 Action/checkpoint adaptation | Not started | Pending | Freeze official PandaOmron schema |
| M3 RGB-only smoke training | Not started | Pending | One-batch A100/A800 forward/backward |
| M4 Closed-loop evaluator | Not started | Pending | One complete atomic rollout |
| M5 Offline depth pilot | Not started | Pending | One-to-three-task aligned cache |
| M6 Atomic training/evaluation | Not started | Pending | H100 training readiness |
| M7 Reproducibility | Not started | Pending | Clean-clone reproduction |

## Confirmed resources

- Existing Wan2.2-TI2V-5B path is reusable and should not be duplicated.
- X-WAM public cross-embodiment pretrained checkpoint has been downloaded or is completing under the agreed X-WAM model root.
- A100/A800 will be used for debug; H100 will be used for training.

## Open inputs

- Confirm the downloaded X-WAM pretrained file with the path check in `docs/CLUSTER_RUNBOOK.md`.
- Provide the RoboCasa365 dataset root when available.
- For M1 cluster validation, provide one minimal real episode plus its metadata and referenced RGB media.

## Immediate next actions

1. Clone the published development branch on Starlight and capture the environment inventory.
2. Verify the external checkpoint paths without loading Torch.
3. Start M1 with metadata/schema inspection before implementing tensor assumptions.
