"""星光 X-WAM policy 环境的无侵入审计逻辑。"""

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


def version_key(version: str, width: int = 4) -> tuple[int, ...]:
    """提取足以比较本项目依赖的数字版本分量。"""
    numbers = [int(value) for value in VERSION_NUMBER.findall(version.split("+", 1)[0])]
    return tuple((numbers + [0] * width)[:width])


def version_satisfies(version: str, requirement: dict[str, Any]) -> tuple[bool, list[str]]:
    """检查一个已安装版本是否满足 manifest 中的边界。"""
    actual = version_key(version)
    problems: list[str] = []
    if requirement.get("min") and actual < version_key(str(requirement["min"])):
        problems.append(f"{version} < {requirement['min']}")
    if requirement.get("max_exclusive") and actual >= version_key(str(requirement["max_exclusive"])):
        problems.append(f"{version} >= {requirement['max_exclusive']}")
    if requirement.get("max_inclusive") and actual > version_key(str(requirement["max_inclusive"])):
        problems.append(f"{version} > {requirement['max_inclusive']}")
    return not problems, problems


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"不支持的环境 manifest schema：{manifest.get('schema_version')!r}")
    if not isinstance(manifest.get("packages"), list):
        raise ValueError("环境 manifest 缺少 packages 列表。")
    return manifest


def installed_distribution_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_command(command: list[str], timeout: int = 60) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {
            "ok": False,
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": f"找不到命令：{command[0]}",
        }
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "command": command,
            "returncode": None,
            "stdout": (exc.stdout or "")[-12000:] if isinstance(exc.stdout, str) else "",
            "stderr": f"命令在 {timeout} 秒后超时。",
        }
    return {
        "ok": result.returncode == 0,
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout[-12000:],
        "stderr": result.stderr[-12000:],
    }


def probe_import(module: str, timeout: int = 60) -> dict[str, Any]:
    """在子进程导入模块，避免坏 ABI 直接终止主审计进程。"""
    code = (
        "import importlib, json; "
        f"m=importlib.import_module({module!r}); "
        "print(json.dumps({'module': m.__name__, 'version': getattr(m, '__version__', None)}))"
    )
    return run_command([sys.executable, "-c", code], timeout=timeout)


def evaluate_packages(
    package_specs: list[dict[str, Any]],
    version_lookup: Callable[[str], str | None] = installed_distribution_version,
    run_imports: bool = True,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    for spec in package_specs:
        distribution = str(spec["distribution"])
        required = bool(spec.get("required", True))
        installed = version_lookup(distribution)
        item: dict[str, Any] = {
            "distribution": distribution,
            "module": spec.get("module"),
            "installed": installed,
            "required": required,
        }
        if installed is None:
            item["constraint_ok"] = not required
            message = f"缺少 {'必需' if required else '可选'}包：{distribution}"
            (errors if required else warnings).append(message)
            results.append(item)
            continue

        constraint_ok, problems = version_satisfies(installed, spec)
        item["constraint_ok"] = constraint_ok
        item["constraint_problems"] = problems
        if not constraint_ok:
            errors.append(f"{distribution} 版本不兼容：{'; '.join(problems)}")
        tested = spec.get("tested")
        if tested and version_key(installed) != version_key(str(tested)):
            warnings.append(f"{distribution}={installed}，X-WAM upstream 测试版本为 {tested}")

        if run_imports and spec.get("runtime_import"):
            import_result = probe_import(str(spec["module"]))
            item["runtime_import"] = import_result
            if not import_result["ok"]:
                errors.append(f"{distribution} runtime import 失败")
        results.append(item)
    return results, errors, warnings


def classify_clone_base(
    python_ok: bool,
    packages: list[dict[str, Any]],
    torch_probe: dict[str, Any],
    *,
    require_gpu: bool,
) -> tuple[bool, list[str]]:
    """区分“可 clone 后修补”与 Python/Torch/CUDA 核心底座不可复用。"""
    hard_blockers: list[str] = []
    if not python_ok:
        hard_blockers.append("Python 主次版本不满足 policy 环境契约")

    by_distribution = {item["distribution"]: item for item in packages}
    for distribution in ("torch", "torchvision", "torchaudio"):
        item = by_distribution.get(distribution)
        if item is None or item.get("installed") is None:
            hard_blockers.append(f"缺少 CUDA 核心包：{distribution}")
            continue
        if not item.get("constraint_ok"):
            hard_blockers.append(f"CUDA 核心包版本不兼容：{distribution}")
        runtime_import = item.get("runtime_import")
        if runtime_import is not None and not runtime_import.get("ok"):
            hard_blockers.append(f"CUDA 核心包 import 失败：{distribution}")

    if not torch_probe.get("ok"):
        hard_blockers.append("Torch runtime probe 失败")
    elif require_gpu and not torch_probe.get("details", {}).get("cuda_available"):
        hard_blockers.append("Torch 无法识别当前 CUDA GPU")
    return not hard_blockers, hard_blockers


def probe_torch() -> dict[str, Any]:
    code = """
import json
import torch
devices = []
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    devices.append({
        "index": index,
        "name": props.name,
        "total_memory_bytes": props.total_memory,
        "compute_capability": list(torch.cuda.get_device_capability(index)),
    })
print(json.dumps({
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "devices": devices,
}))
"""
    result = run_command([sys.executable, "-c", code], timeout=90)
    if result["ok"]:
        try:
            result["details"] = json.loads(result["stdout"].strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            result["ok"] = False
            result["stderr"] += "\n无法解析 Torch probe JSON。"
    return result


def collect_repository(repo_root: Path) -> dict[str, Any]:
    head = run_command(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    status = run_command(["git", "-C", str(repo_root), "status", "--short", "--branch"])
    submodules = run_command(
        [
            "git",
            "-C",
            str(repo_root),
            "ls-tree",
            "HEAD",
            "third_party/robocasa",
            "third_party/robosuite",
            "third_party/RoboTwin",
        ]
    )
    return {
        "root": str(repo_root),
        "head": head["stdout"].strip() if head["ok"] else None,
        "status": status["stdout"].strip(),
        "submodule_gitlinks": submodules["stdout"].strip().splitlines() if submodules["ok"] else [],
    }


def audit_environment(
    manifest_path: str | Path,
    repo_root: str | Path,
    *,
    require_gpu: bool = False,
    run_imports: bool = True,
    include_deepspeed_report: bool = False,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    errors: list[str] = []
    warnings: list[str] = []

    python_version = platform.python_version()
    python_ok, python_problems = version_satisfies(python_version, manifest["python"])
    if not python_ok:
        errors.append(f"Python 版本不兼容：{'; '.join(python_problems)}")

    packages, package_errors, package_warnings = evaluate_packages(
        manifest["packages"], run_imports=run_imports
    )
    errors.extend(package_errors)
    warnings.extend(package_warnings)

    torch_probe = probe_torch() if installed_distribution_version("torch") else {"ok": False, "skipped": True}
    if installed_distribution_version("torch") and not torch_probe.get("ok"):
        errors.append("Torch runtime probe 失败")
    elif require_gpu and not torch_probe.get("details", {}).get("cuda_available"):
        errors.append("当前节点没有可用 CUDA GPU，但本次使用了 --require-gpu。")

    deepspeed_report = None
    if include_deepspeed_report:
        deepspeed_report = run_command([sys.executable, "-m", "deepspeed.env_report"], timeout=120)
        if not deepspeed_report["ok"]:
            errors.append("DeepSpeed ds_report 执行失败")

    pip_check = run_command([sys.executable, "-m", "pip", "check"], timeout=120)
    if not pip_check["ok"]:
        errors.append("pip check 发现依赖冲突")

    nvidia_smi = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    nvcc = run_command(["nvcc", "--version"])
    if not nvcc["ok"]:
        warnings.append("未找到可用 nvcc；若需要编译 FlashAttention/DeepSpeed op，将无法完成。")

    clone_base_ok, hard_blockers = classify_clone_base(
        python_ok,
        packages,
        torch_probe,
        require_gpu=require_gpu,
    )
    ok = not errors
    if ok:
        reuse_recommendation = "clone_ready"
    elif clone_base_ok:
        reuse_recommendation = "clone_then_patch"
    else:
        reuse_recommendation = "select_another_base_or_rebuild"
    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": ok,
        "clone_base_ok": clone_base_ok,
        "reuse_recommendation": reuse_recommendation,
        "hard_blockers": hard_blockers,
        "environment": {
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "python_executable": sys.executable,
            "python_version": python_version,
            "python_constraint_ok": python_ok,
            "platform": platform.platform(),
        },
        "packages": packages,
        "torch_probe": torch_probe,
        "nvidia_smi": nvidia_smi,
        "nvcc": nvcc,
        "deepspeed_report": deepspeed_report,
        "pip_check": pip_check,
        "repository": collect_repository(Path(repo_root).expanduser().resolve()),
        "simulator_contract": manifest["simulator_contract"],
        "fallback": manifest["fallback"],
        "warnings": warnings,
        "errors": errors,
    }
