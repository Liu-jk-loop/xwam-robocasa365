"""Runtime X-WAM initialization with an auditable PandaOmron checkpoint boundary."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data.robocasa365_schema import load_panda_omron_schema
from project_tools.xwam_checkpoint_contract import (
    plan_xwam_checkpoint_adaptation,
    remap_action_boundary,
    resolve_xwam_checkpoint,
)


SUPPORTED_INITIALIZATION_MODES = {"legacy_strict", "xwam_pretrained", "wan_base"}


def _write_report(path: str | Path | None, report: dict[str, Any]) -> None:
    if path is None:
        return
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


def initialize_xwam_runner(
    runner: Any,
    *,
    mode: str,
    checkpoint: str | Path | None,
    schema_path: str | Path | None,
    report_path: str | Path | None,
) -> dict[str, Any]:
    """Initialize a runner and persist the exact loading decision as JSON."""

    if mode not in SUPPORTED_INITIALIZATION_MODES:
        raise ValueError(f"initialization_mode 只允许 {sorted(SUPPORTED_INITIALIZATION_MODES)}，实际为 {mode!r}")
    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "checkpoint": None,
        "schema_path": str(Path(schema_path).expanduser().resolve()) if schema_path else None,
        "result": "fail",
        "ok": False,
    }

    try:
        if mode == "wan_base":
            if checkpoint is not None:
                raise ValueError("wan_base 模式不允许 pretrained_checkpoint，避免误以为加载了 X-WAM 权重")
            report.update(
                {
                    "policy": "保留 Wan2.2 base，并使用 XWAMModel.init_new_weights 初始化 action/proprio 边界",
                    "target_tensor_count": len(runner.state_dict()),
                    "loaded": ["Wan2.2 backbone（由 XWAMModel.from_pretrained 加载）"],
                    "remapped": [],
                    "initialized": [
                        "model.view_embedding",
                        "model.action_encoder",
                        "model.action_decoder",
                        "model.proprio_encoder",
                        "model.proprio_decoder",
                    ],
                    "missing": [],
                    "unexpected": [],
                    "result": "pass",
                    "ok": True,
                }
            )
            _write_report(report_path, report)
            return report

        if checkpoint is None:
            raise ValueError(f"{mode} 模式必须提供 pretrained_checkpoint")
        checkpoint_path = resolve_xwam_checkpoint(checkpoint)
        report["checkpoint"] = str(checkpoint_path)

        import torch

        payload = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
        source_state = payload.get("module", payload) if isinstance(payload, dict) else payload
        if not isinstance(source_state, dict):
            raise ValueError("checkpoint 顶层或 module 不是 state dict")

        if mode == "legacy_strict":
            runner.load_state_dict(source_state, strict=True)
            report.update(
                {
                    "policy": "upstream legacy strict load",
                    "source_tensor_count": len(source_state),
                    "result": "pass",
                    "ok": True,
                }
            )
            _write_report(report_path, report)
            return report

        if schema_path is None:
            raise ValueError("xwam_pretrained 模式必须提供 schema_path")
        schema = load_panda_omron_schema(schema_path)
        mapping = schema.checkpoint_mapping
        source_slice = mapping.get("arm_source_slice")
        target_slice = mapping.get("arm_target_slice")
        if mapping.get("source_action_schema") != "legacy_dual_arm_14d":
            raise ValueError("schema source_action_schema 不是已审计的 legacy_dual_arm_14d")
        if mapping.get("target_action_schema") != "panda_omron_12d":
            raise ValueError("schema target_action_schema 不是已审计的 panda_omron_12d")
        if source_slice != [0, 7] or target_slice != [5, 12]:
            raise ValueError("schema arm 映射必须为 legacy[0:7] -> PandaOmron[5:12]")
        if mapping.get("new_action_slice") != [0, 5]:
            raise ValueError("schema new_action_slice 必须为 PandaOmron base/control [0,5]")
        if mapping.get("proprio_boundary_policy") != "reinitialize_due_to_semantic_change":
            raise ValueError("schema 未声明 proprio boundary 语义重初始化策略")

        target_state = runner.state_dict()
        source_shapes = {
            str(key): tuple(int(dim) for dim in value.shape)
            for key, value in source_state.items()
            if hasattr(value, "shape")
        }
        target_shapes = {
            str(key): tuple(int(dim) for dim in value.shape)
            for key, value in target_state.items()
        }
        plan = plan_xwam_checkpoint_adaptation(
            source_shapes,
            target_shapes,
            source_arm_slice=source_slice,
            target_arm_slice=target_slice,
        )
        report["plan"] = plan
        report["source_tensor_count"] = len(source_shapes)
        report["target_tensor_count"] = len(target_shapes)
        if not plan["ok"]:
            raise ValueError("checkpoint 适配合同失败：\n- " + "\n- ".join(plan["errors"][:20]))

        with torch.no_grad():
            for key in plan["exact_load"]:
                target_state[key].copy_(source_state[key])
            for key in plan["action_remap"]:
                remapped = remap_action_boundary(
                    key,
                    source_state[key],
                    target_state[key],
                    source_arm_slice=source_slice,
                    target_arm_slice=target_slice,
                )
                target_state[key].copy_(remapped)

        report.update(
            {
                "policy": "严格同名同 shape 加载；legacy arm[0:7] 映射到 PandaOmron action[5:12]；proprio 边界重初始化",
                "result": "pass",
                "ok": True,
            }
        )
        _write_report(report_path, report)
        return report
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        _write_report(report_path, report)
        raise
