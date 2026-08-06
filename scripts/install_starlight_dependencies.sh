#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "用法: bash scripts/install_starlight_dependencies.sh dry-run|apply" >&2
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

mode="$1"
if [[ "$mode" != "dry-run" && "$mode" != "apply" ]]; then
  usage
  exit 2
fi

active_env="${CONDA_DEFAULT_ENV:-}"
if [[ "$active_env" == "abot_m05" ]]; then
  echo "拒绝执行：abot_m05 是只读母环境，请先 clone 并激活 xwam-robocasa365。" >&2
  exit 2
fi
if [[ "$active_env" != "xwam-robocasa365" ]]; then
  echo "拒绝执行：当前 Conda 环境为 '${active_env:-未激活}'，预期 xwam-robocasa365。" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
constraints="$repo_root/configs/environment/xwam_starlight_constraints.txt"
requirements="$repo_root/requirements.txt"
log_dir="$repo_root/logs/cluster"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="$log_dir/xwam_dependency_${mode}_${timestamp}.log"
freeze_file="$log_dir/xwam_dependency_${mode}_${timestamp}_before.txt"
after_file="$log_dir/xwam_dependency_${mode}_${timestamp}_after.txt"

mkdir -p "$log_dir"
python -m pip freeze > "$freeze_file"

if python -m pip show wam >/dev/null 2>&1; then
  echo "拒绝执行：clone 中仍有 ABot 的 wam 包，它与 X-WAM 的 NumPy/Transformers 约束冲突。" >&2
  echo "请仅在当前 clone 环境执行：python -m pip uninstall -y wam" >&2
  exit 2
fi

pip_args=(
  -m pip install
  --constraint "$constraints"
  --requirement "$requirements"
)
if [[ "$mode" == "dry-run" ]]; then
  pip_args+=(--dry-run)
fi

echo "环境：$active_env"
echo "模式：$mode"
echo "安装日志：$log_file"
echo "安装前快照：$freeze_file"
echo "不会修改代理变量；本次 pip 子进程固定 DS_BUILD_OPS=0。"

set +e
DS_BUILD_OPS=0 python "${pip_args[@]}" 2>&1 | tee "$log_file"
pip_status=${PIPESTATUS[0]}
set -e
if [[ $pip_status -ne 0 ]]; then
  echo "依赖解析/安装失败，完整输出位于：$log_file" >&2
  exit "$pip_status"
fi

if [[ "$mode" == "apply" ]]; then
  python -m pip freeze > "$after_file"

  set +e
  pip_check_output="$(python -m pip check 2>&1)"
  pip_check_status=$?
  set -e
  echo "$pip_check_output" | tee -a "$log_file"

  if [[ $pip_check_status -ne 0 ]]; then
    if [[ "$pip_check_output" == "decord 0.6.0 is not supported on this platform" ]] \
      && python -c "import decord" >/dev/null 2>&1; then
      echo "已知警告：Decord 0.6.0 runtime import 已通过，仅 wheel tag 被 pip 判为不支持。" | tee -a "$log_file"
    else
      echo "pip check 发现未允许的依赖问题；安装后快照已保存：$after_file" >&2
      exit "$pip_check_status"
    fi
  fi
fi

echo "完成：$mode；完整日志位于 $log_file"
