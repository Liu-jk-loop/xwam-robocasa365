"""RoboCasa365 simulator 候选环境的无侵入审计逻辑。"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


VERSION_NUMBER = re.compile(r"\d+")
PROBE_PREFIX = "ROBOCASA365_AUDIT_JSON="


def version_key(version: str, width: int = 4) -> tuple[int, ...]:
    numbers = [int(value) for value in VERSION_NUMBER.findall(version.split("+", 1)[0])]
    return tuple((numbers + [0] * width)[:width])


def version_satisfies(version: str, requirement: dict[str, Any]) -> tuple[bool, list[str]]:
    actual = version_key(version)
    problems: list[str] = []
    if requirement.get("exact") and actual != version_key(str(requirement["exact"])):
        problems.append(f"{version} != {requirement['exact']}")
    if requirement.get("min") and actual < version_key(str(requirement["min"])):
        problems.append(f"{version} < {requirement['min']}")
    if requirement.get("max_exclusive") and actual >= version_key(str(requirement["max_exclusive"])):
        problems.append(f"{version} >= {requirement['max_exclusive']}")
    return not problems, problems


def load_manifest(path: str | Path, repo_root: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"不支持的 simulator manifest schema：{payload.get('schema_version')!r}")
    if payload.get("scope") != "atomic_only":
        raise ValueError("simulator manifest 必须为 scope=atomic_only")
    task_manifest = Path(repo_root).resolve() / str(payload["task_manifest"])
    task_payload = json.loads(task_manifest.read_text(encoding="utf-8"))
    if task_payload.get("scope") != "atomic_only":
        raise ValueError("任务清单必须为 scope=atomic_only")
    tasks = task_payload.get("tasks")
    if not isinstance(tasks, list) or not tasks or not all(isinstance(task, str) for task in tasks):
        raise ValueError("任务清单缺少非空 tasks 字符串列表")
    payload["resolved_task_manifest"] = str(task_manifest)
    payload["atomic_seen_tasks"] = tasks
    return payload


def installed_distribution_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_command(
    command: list[str],
    *,
    timeout: int = 90,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    executable = shutil.which(command[0]) if command else None
    if executable is None:
        return {
            "ok": False,
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": f"找不到命令：{command[0] if command else '<empty>'}",
        }
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "command": command,
            "returncode": None,
            "stdout": (exc.stdout or "")[-20000:] if isinstance(exc.stdout, str) else "",
            "stderr": f"命令在 {timeout} 秒后超时。",
        }
    return {
        "ok": result.returncode == 0,
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout[-20000:],
        "stderr": result.stderr[-20000:],
    }


def parse_probe_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    for line in reversed(str(result.get("stdout", "")).splitlines()):
        if line.startswith(PROBE_PREFIX):
            try:
                payload = json.loads(line[len(PROBE_PREFIX) :])
            except json.JSONDecodeError:
                return None
            return payload if isinstance(payload, dict) else None
    return None


def evaluate_packages(
    specs: list[dict[str, Any]],
    version_lookup: Callable[[str], str | None] = installed_distribution_version,
) -> tuple[list[dict[str, Any]], list[str]]:
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for spec in specs:
        distribution = str(spec["distribution"])
        installed = version_lookup(distribution)
        item: dict[str, Any] = {
            "distribution": distribution,
            "module": spec.get("module"),
            "installed": installed,
        }
        if installed is None:
            item["constraint_ok"] = False
            errors.append(f"缺少必需包：{distribution}")
        else:
            constraint_ok, problems = version_satisfies(installed, spec)
            item["constraint_ok"] = constraint_ok
            item["constraint_problems"] = problems
            if not constraint_ok:
                errors.append(f"{distribution} 版本不满足 RoboCasa365 契约：{'; '.join(problems)}")
        results.append(item)
    return results, errors


def _registry_probe_code(tasks: list[str]) -> str:
    return f"""
import json
import gymnasium as gym
import robocasa
import robosuite
import mujoco
import numpy

tasks = {tasks!r}
registered = set(gym.registry.keys())
missing = [task for task in tasks if f"robocasa/{{task}}" not in registered]
payload = {{
    "versions": {{
        "robocasa": getattr(robocasa, "__version__", None),
        "robosuite": getattr(robosuite, "__version__", None),
        "mujoco": getattr(mujoco, "__version__", None),
        "numpy": getattr(numpy, "__version__", None),
    }},
    "module_paths": {{
        "robocasa": getattr(robocasa, "__file__", None),
        "robosuite": getattr(robosuite, "__file__", None),
    }},
    "atomic_seen_total": len(tasks),
    "atomic_seen_registered": len(tasks) - len(missing),
    "missing_atomic_seen": missing,
}}
print({PROBE_PREFIX!r} + json.dumps(payload, sort_keys=True))
"""


def probe_registry(tasks: list[str]) -> dict[str, Any]:
    result = run_command([sys.executable, "-c", _registry_probe_code(tasks)], timeout=180)
    payload = parse_probe_payload(result)
    result["details"] = payload
    if payload is None:
        result["ok"] = False
        result["stderr"] = str(result.get("stderr", "")) + "\n无法解析 registry probe JSON。"
    elif payload.get("missing_atomic_seen"):
        result["ok"] = False
    return result


def _runtime_probe_code(runtime: dict[str, Any]) -> str:
    return f"""
import json
import numpy as np
import gymnasium as gym
import robocasa

runtime = {runtime!r}
expected_obs = runtime["observation_components"]
expected_cameras = runtime["camera_keys"]
expected_actions = runtime["action_components"]
env = None
try:
    env = gym.make(
        f"robocasa/{{runtime['task']}}",
        split=runtime["split"],
        seed=int(runtime["seed"]),
        enable_render=True,
    )
    obs, reset_info = env.reset(seed=int(runtime["seed"]))
    action_keys = sorted(env.action_space.spaces.keys())
    action_dims = {{key: int(np.prod(env.action_space[key].shape)) for key in action_keys}}
    action = {{
        key: np.zeros(env.action_space[key].shape, dtype=np.float32)
        for key in action_keys
    }}
    action["action.control_mode"][...] = -1.0
    action["action.gripper_close"][...] = -1.0
    next_obs, reward, terminated, truncated, step_info = env.step(action)

    state_dims = {{key: int(np.prod(np.asarray(obs[key]).shape)) for key in expected_obs if key in obs}}
    camera_shapes = {{key: list(np.asarray(obs[key]).shape) for key in expected_cameras if key in obs}}
    camera_dtypes = {{key: str(np.asarray(obs[key]).dtype) for key in expected_cameras if key in obs}}
    render = np.asarray(env.render())
    errors = []
    if state_dims != expected_obs:
        errors.append(f"state contract mismatch: {{state_dims}}")
    if action_dims != expected_actions:
        errors.append(f"action contract mismatch: {{action_dims}}")
    expected_camera_shape = runtime["camera_shape"]
    if camera_shapes != {{key: expected_camera_shape for key in expected_cameras}}:
        errors.append(f"camera contract mismatch: {{camera_shapes}}")
    if any(dtype != "uint8" for dtype in camera_dtypes.values()):
        errors.append(f"camera dtype mismatch: {{camera_dtypes}}")
    if list(render.shape) != expected_camera_shape:
        errors.append(f"render shape mismatch: {{list(render.shape)}}")
    payload = {{
        "task": runtime["task"],
        "split": runtime["split"],
        "seed": runtime["seed"],
        "state_dims": state_dims,
        "state_dim_total": sum(state_dims.values()),
        "action_dims": action_dims,
        "action_dim_total": sum(action_dims.values()),
        "camera_shapes": camera_shapes,
        "camera_dtypes": camera_dtypes,
        "render_shape": list(render.shape),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "reset_info_keys": sorted(reset_info.keys()),
        "step_info_keys": sorted(step_info.keys()),
        "errors": errors,
        "ok": not errors,
    }}
    print({PROBE_PREFIX!r} + json.dumps(payload, sort_keys=True))
    if errors:
        raise SystemExit(3)
finally:
    if env is not None:
        env.close()
"""


def probe_runtime(runtime: dict[str, Any], *, timeout: int) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.setdefault("MUJOCO_GL", str(runtime.get("mujoco_gl", "egl")))
    result = run_command(
        [sys.executable, "-c", _runtime_probe_code(runtime)],
        timeout=timeout,
        env=environment,
    )
    payload = parse_probe_payload(result)
    result["details"] = payload
    result["mujoco_gl"] = environment.get("MUJOCO_GL")
    if payload is None:
        result["ok"] = False
        result["stderr"] = str(result.get("stderr", "")) + "\n无法解析 runtime probe JSON。"
    elif not payload.get("ok"):
        result["ok"] = False
    return result


def _find_git_root(module_path: str | None) -> Path | None:
    if not module_path:
        return None
    current = Path(module_path).expanduser().resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def collect_source_provenance(module_paths: dict[str, Any]) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    for name, raw_path in module_paths.items():
        module_path = str(raw_path) if raw_path else None
        root = _find_git_root(module_path)
        item: dict[str, Any] = {"module_path": module_path, "git_root": str(root) if root else None}
        if root is not None:
            head = run_command(["git", "-C", str(root), "rev-parse", "HEAD"])
            status = run_command(["git", "-C", str(root), "status", "--short", "--branch"])
            item["git_head"] = head["stdout"].strip() if head["ok"] else None
            item["git_status"] = status["stdout"].strip() if status["ok"] else None
        sources[name] = item
    return sources


def classify_reuse(
    package_errors: list[str],
    registry_probe: dict[str, Any],
    runtime_probe: dict[str, Any] | None,
) -> tuple[bool, str, list[str]]:
    blockers = list(package_errors)
    robocasa_version = None
    details = registry_probe.get("details") or {}
    if isinstance(details, dict):
        robocasa_version = (details.get("versions") or {}).get("robocasa")

    if robocasa_version and version_key(str(robocasa_version))[0] == 0:
        blockers.append("检测到 RoboCasa 0.x：只覆盖旧版任务，不是 RoboCasa365 simulator")
        return False, "legacy_robocasa_only", blockers
    if not registry_probe.get("ok"):
        blockers.append("Atomic-Seen 18 注册或 RoboCasa import 检查失败")
        return False, "registry_blocked", blockers
    if runtime_probe is None:
        return not blockers, "runtime_smoke_required" if not blockers else "dependency_blocked", blockers
    if not runtime_probe.get("ok"):
        blockers.append("CloseFridge target reset/12D 单步/EGL 渲染检查失败")
        return False, "runtime_blocked", blockers
    if blockers:
        return False, "dependency_blocked", blockers
    return True, "reuse_ready", []


def audit_simulator_environment(
    manifest_path: str | Path,
    repo_root: str | Path,
    *,
    runtime_smoke: bool,
    runtime_timeout: int = 600,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path, repo_root)
    packages, package_errors = evaluate_packages(manifest["packages"])
    python_spec = manifest["python"]
    python_version = platform.python_version()
    python_ok, python_problems = version_satisfies(python_version, python_spec)
    if not python_ok:
        package_errors.append(f"Python 版本不满足 simulator 契约：{'; '.join(python_problems)}")

    registry = probe_registry(list(manifest["atomic_seen_tasks"]))
    runtime = probe_runtime(manifest["runtime_smoke"], timeout=runtime_timeout) if runtime_smoke else None
    ok, recommendation, blockers = classify_reuse(package_errors, registry, runtime)
    registry_details = registry.get("details") or {}
    module_paths = registry_details.get("module_paths", {}) if isinstance(registry_details, dict) else {}
    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "python_executable": sys.executable,
            "python_version": python_version,
            "python_ok": python_ok,
            "platform": platform.platform(),
        },
        "scope": manifest["scope"],
        "task_manifest": manifest["resolved_task_manifest"],
        "atomic_seen_tasks": manifest["atomic_seen_tasks"],
        "packages": packages,
        "registry_probe": registry,
        "runtime_smoke_requested": runtime_smoke,
        "runtime_probe": runtime,
        "source_provenance": collect_source_provenance(module_paths),
        "ok": ok,
        "reuse_recommendation": recommendation,
        "blockers": blockers,
        "warnings": [] if runtime_smoke else ["尚未创建 simulator；需要加 --runtime-smoke 完成最终复用门禁。"],
    }
