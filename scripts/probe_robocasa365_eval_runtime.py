#!/usr/bin/env python3
"""Verify that formal evaluation uses the Store RoboCasa tree and assets."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _module_path(module: Any) -> Path:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError(f"module has no __file__: {module!r}")
    return Path(module_file).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robocasa-root", required=True)
    parser.add_argument("--robosuite-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    robocasa_root = Path(args.robocasa_root).expanduser().resolve()
    robosuite_root = Path(args.robosuite_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    checks: dict[str, bool] = {}
    errors: list[str] = []
    report: dict[str, Any] = {
        "schema_version": 1,
        "result": "fail",
        "ok": False,
        "robocasa_root": str(robocasa_root),
        "robosuite_root": str(robosuite_root),
        "checks": checks,
        "errors": errors,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import robocasa
        import robosuite

        robocasa_module = _module_path(robocasa)
        robosuite_module = _module_path(robosuite)
        assets_root = robocasa_root / "robocasa" / "models" / "assets"
        sink_model = assets_root / "fixtures" / "sinks" / "Sink025" / "model.xml"
        checks.update(
            robocasa_import_from_store=robocasa_module.is_relative_to(robocasa_root),
            robosuite_import_from_store=robosuite_module.is_relative_to(robosuite_root),
            assets_root_exists=assets_root.is_dir(),
            sink025_model_exists=sink_model.is_file() and sink_model.stat().st_size > 0,
        )
        report.update(
            robocasa_module=str(robocasa_module),
            robosuite_module=str(robosuite_module),
            robocasa_version=importlib.metadata.version("robocasa"),
            robosuite_version=importlib.metadata.version("robosuite"),
            assets_root=str(assets_root),
            sink025_model=str(sink_model),
            sink025_bytes=sink_model.stat().st_size if sink_model.is_file() else None,
        )
        errors.extend(name for name, passed in checks.items() if not passed)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")

    report["ok"] = not errors and all(checks.values())
    report["result"] = "pass" if report["ok"] else "fail"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
