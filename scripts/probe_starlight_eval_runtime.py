#!/usr/bin/env python3
"""在星光双 Conda 环境中执行 Atomic9 评测运行时预检。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _module_path(module: Any) -> str:
    value = getattr(module, "__file__", None)
    return str(Path(value).resolve()) if value else "unavailable"


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON顶层必须为对象：{path}")
    return payload


def _manifest_digest(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("manifest_digest", None)
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _probe_policy(
    expected_gpus: int,
    manifest_path: Path,
    statistics_path: Path,
) -> tuple[dict[str, bool], dict[str, Any]]:
    import deepspeed
    import flash_attn
    import lightning
    import numpy
    import omegaconf
    import torch
    import transformers
    import zmq

    gpu_count = int(torch.cuda.device_count())
    manifest = _read_json(manifest_path)
    statistics = _read_json(statistics_path)
    tasks = manifest.get("tasks", [])
    dataset_paths = [
        Path(str(task.get("dataset_path", ""))).expanduser().resolve()
        for task in tasks
        if isinstance(task, dict)
    ]
    missing_dataset_paths = [str(path) for path in dataset_paths if not path.is_dir()]
    declared_manifest_digest = str(manifest.get("manifest_digest", ""))
    actual_manifest_digest = _manifest_digest(manifest)
    checks = {
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "expected_gpu_count": gpu_count == expected_gpus,
        "cuda_bfloat16_supported": bool(
            torch.cuda.is_available()
            and all(torch.cuda.is_bf16_supported(index) for index in range(gpu_count))
        ),
        "pyzmq_import": hasattr(zmq, "Context"),
        "omegaconf_import": hasattr(omegaconf, "OmegaConf"),
        "manifest_pass": manifest.get("ok") is True
        and manifest.get("result") == "pass",
        "manifest_content_digest": bool(declared_manifest_digest)
        and declared_manifest_digest == actual_manifest_digest,
        "atomic9_task_count": len(tasks) == 9 and len(dataset_paths) == 9,
        "dataset_paths_exist": not missing_dataset_paths,
        "statistics_pass": statistics.get("ok") is True
        and statistics.get("result") == "pass",
        "statistics_manifest_digest": bool(declared_manifest_digest)
        and statistics.get("manifest_digest") == declared_manifest_digest,
        "statistics_dimensions": len(
            statistics.get("observation.state", {}).get("q01", [])
        )
        == 16
        and len(statistics.get("action", {}).get("q01", [])) == 12,
    }
    details = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu_count": gpu_count,
        "gpus": [torch.cuda.get_device_name(index) for index in range(gpu_count)],
        "numpy": numpy.__version__,
        "transformers": transformers.__version__,
        "flash_attn": flash_attn.__version__,
        "lightning": lightning.__version__,
        "deepspeed": deepspeed.__version__,
        "omegaconf": _package_version("omegaconf"),
        "pyzmq": _package_version("pyzmq"),
        "manifest": str(manifest_path),
        "statistics": str(statistics_path),
        "declared_manifest_digest": declared_manifest_digest,
        "actual_manifest_digest": actual_manifest_digest,
        "task_count": len(tasks),
        "missing_dataset_paths": missing_dataset_paths,
    }
    return checks, details


def _probe_simulator() -> tuple[dict[str, bool], dict[str, Any]]:
    import gymnasium as gym
    import imageio
    import imageio_ffmpeg
    import numpy as np
    import robocasa
    import robosuite
    import zmq

    from data.robocasa365_schema import load_panda_omron_schema
    from evaluation.robocasa365_benchmark import pack_online_cameras, pack_online_state

    robocasa_root = Path(robocasa.__file__).resolve().parent
    assets_root = robocasa_root / "models" / "assets"
    sink_model = assets_root / "fixtures" / "sinks" / "Sink025" / "model.xml"
    environment = None
    observation: dict[str, Any] | None = None
    try:
        environment = gym.make("robocasa/CloseFridge", split="target")
        observation, _ = environment.reset(seed=42)
    finally:
        if environment is not None:
            environment.close()

    assert observation is not None
    camera_keys = (
        "video.robot0_agentview_left",
        "video.robot0_agentview_right",
        "video.robot0_eye_in_hand",
    )
    schema = load_panda_omron_schema(
        REPO_ROOT / "configs/schemas/robocasa365_panda_omron_v1.json"
    )
    state = pack_online_state(observation, schema)
    cameras = pack_online_cameras(observation, list(camera_keys), [256, 256, 3])
    ffmpeg_executable = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    checks = {
        "robocasa_assets": assets_root.is_dir(),
        "sink025_model": sink_model.is_file() and sink_model.stat().st_size > 0,
        "close_fridge_registered": "robocasa/CloseFridge" in gym.envs.registry,
        "close_fridge_reset": isinstance(observation, dict),
        "state_16d": state.shape == (16,),
        "three_rgb_cameras": cameras.shape == (3, 256, 256, 3)
        and cameras.dtype == np.uint8,
        "task_description": bool(
            str(observation.get("annotation.human.task_description", "")).strip()
        ),
        "pyzmq_import": hasattr(zmq, "Context"),
        "ffmpeg_available": ffmpeg_executable.is_file(),
    }
    details = {
        "robocasa": _package_version("robocasa"),
        "robosuite": _package_version("robosuite"),
        "gymnasium": _package_version("gymnasium"),
        "imageio": imageio.__version__,
        "pyzmq": _package_version("pyzmq"),
        "robocasa_module": _module_path(robocasa),
        "robosuite_module": _module_path(robosuite),
        "assets_root": str(assets_root),
        "sink025_model": str(sink_model),
        "state_shape": list(state.shape),
        "camera_shape": list(cameras.shape),
        "camera_dtype": str(cameras.dtype),
        "ffmpeg_executable": str(ffmpeg_executable),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
    }
    return checks, details


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("policy", "simulator"), required=True)
    parser.add_argument("--expected-gpus", type=int, default=4)
    parser.add_argument("--manifest")
    parser.add_argument("--statistics-path")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.expected_gpus <= 0:
        parser.error("--expected-gpus 必须为正整数")
    if args.role == "policy" and (not args.manifest or not args.statistics_path):
        parser.error("policy预检必须提供--manifest和--statistics-path")

    output = Path(args.output).expanduser().resolve()
    report: dict[str, Any] = {
        "schema_version": 1,
        "role": args.role,
        "result": "fail",
        "ok": False,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "checks": {},
        "details": {},
        "errors": [],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        if args.role == "policy":
            checks, details = _probe_policy(
                args.expected_gpus,
                Path(args.manifest).expanduser().resolve(),
                Path(args.statistics_path).expanduser().resolve(),
            )
        else:
            checks, details = _probe_simulator()
        report["checks"] = checks
        report["details"] = details
        report["errors"] = [name for name, passed in checks.items() if not passed]
    except Exception as exc:
        report["errors"] = [f"{type(exc).__name__}: {exc}"]
        report["traceback"] = traceback.format_exc()

    report["ok"] = not report["errors"] and all(report["checks"].values())
    report["result"] = "pass" if report["ok"] else "fail"
    _write_report(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
