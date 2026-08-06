# Architecture decisions

## Adapter boundaries

RoboCasa365 support will be added behind five explicit boundaries:

1. `DatasetAdapter`: native episode, video, metadata, instruction, and task access.
2. `ObservationAdapter`: camera selection, image transforms, state packing, and modality validation.
3. `ActionCodec`: named action components, normalization, model representation, and environment packing.
4. `CheckpointAdapter`: controlled loading between Wan2.2, legacy X-WAM, and the adapted model.
5. `BenchmarkAdapter`: task registry, simulator lifecycle, rollout records, and metric aggregation.

The X-WAM backbone should consume validated tensors and remain free of dataset-path and simulator-specific conditionals.

## Data contract

- Native RoboCasa365 LeRobot/Parquet is the source of truth.
- The adapter must read official metadata rather than infer unnamed dimensions from array length.
- The M1 contract gate expects PandaOmron `observation.state` to be 16D, `action` to be 12D, and the three official RGB camera keys to be present. M2 will freeze the component names and slices from the real `modality.json` before model mapping.
- Atomic-only selection is explicit and auditable.
- Cameras are selected by configured keys and stable ordering.
- RGB is required for the initial path.
- Depth has two legal modes: `disabled` and `cached`. Online simulator rendering in a data-loader worker is forbidden.
- Normalization statistics are generated for an immutable dataset/task manifest and stored with provenance.

### M1 native loader boundary

- `data/robocasa365_index.py` owns dependency-free LeRobot v2.1 episode indexing, prompt parsing, path templates, and temporal windows.
- `data/robocasa365_dataset.py` owns runtime Parquet/MP4 decoding and emits the existing X-WAM batch keys without converting the dataset to legacy JSON.
- The current M1 tensor contract is RGB `[V,T,C,H,W]`, raw state `[T,16]`, raw action `[Ta,12]`, and explicit validity/camera masks. Camera order is configured and never shuffled.
- Raw state/action are intentionally restricted to the batch-audit path. `configs/data/robocasa365.yaml` remains `training_ready: false` until M2 freezes named PandaOmron slices and normalization.
- Only LeRobot v2.x episode-per-file data is accepted by the native loader. A v3 dataset must use a separate indexed shard adapter rather than silently assuming v2 paths.

## Action contract

- Never preserve the legacy evaluator behavior that copies only the first seven predicted values.
- Represent every PandaOmron action component with a name, slice, dtype, range, and environment mapping.
- Read the initial schema from official RoboCasa365 metadata and freeze the validated result in a versioned manifest.
- Keep base motion and control mode available even for manipulation-heavy atomic tasks.
- Validate round-trips using recorded dataset actions before closed-loop policy evaluation.

## Model initialization

Two modes remain supported:

- `xwam_pretrained`: load Wan2.2 base components, construct X-WAM modules, then load the public 40k cross-embodiment X-WAM checkpoint through a shape-aware adapter.
- `wan_base`: load Wan2.2 base components and initialize/copy new X-WAM modules using code, without loading the cross-embodiment checkpoint.

Checkpoint loading must emit a machine-readable report of loaded, remapped, newly initialized, missing, and unexpected parameters. Shape mismatches must never be silently ignored.

## Process boundary

The policy server owns Torch, CUDA, X-WAM, and model weights. The benchmark client owns RoboCasa, robosuite, MuJoCo, task creation, and rendering. ZeroMQ remains the boundary so each side can use a compatible environment and can be scaled independently.

## Configuration

Configuration will be layered rather than encoded in source:

```text
model defaults
  + dataset/schema profile
  + hardware profile
  + experiment profile
  + command-line overrides
```

Absolute cluster paths are allowed in cluster-local overrides but not as Python defaults. Every saved experiment must contain the resolved configuration, Git commit, dataset manifest ID, checkpoint source, and environment summary.

## Current external paths

These paths are cluster deployment facts, not portable defaults:

```text
Wan2.2 base:
/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/Wan-AI/Wan2.2-TI2V-5B

X-WAM pretrained root:
/HOME/sysu_xdliang/sysu_xdliang_5/HDD_POOL/nieyunshuang/models/x-wam/xwam_checkpoints/pretrained/checkpoints/last.ckpt
```
