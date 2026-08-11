#!/usr/bin/env bash

# Shared error reporting for Clariden batch scripts and their EDF child shells.

xwam_write_failure_report() {
  local exit_code="$1"
  local line="$2"
  local command="$3"
  local report_tmp

  [[ -n "${XWAM_FAILURE_REPORT:-}" ]] || return 0
  [[ ! -s "$XWAM_FAILURE_REPORT" ]] || return 0
  command="${command//$'\n'/ }"
  command="${command:0:1000}"
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    command="${command//"$WANDB_API_KEY"/<redacted>}"
  fi
  mkdir -p "$(dirname "$XWAM_FAILURE_REPORT")"
  report_tmp="${XWAM_FAILURE_REPORT}.tmp.${BASHPID:-$$}"
  {
    printf 'timestamp=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'job_id=%s\n' "${SLURM_JOB_ID:-unknown}"
    printf 'phase=%s\n' "${XWAM_PHASE:-unknown}"
    printf 'exit_code=%s\n' "$exit_code"
    printf 'line=%s\n' "$line"
    printf 'command=%q\n' "$command"
    printf 'main_log=%s\n' "${XWAM_MAIN_LOG:-unknown}"
    printf 'train_log=%s\n' "${XWAM_TRAIN_LOG:-unknown}"
  } > "$report_tmp"
  mv "$report_tmp" "$XWAM_FAILURE_REPORT"
}

xwam_on_err() {
  local exit_code="$1"
  local line="$2"
  local command="$3"

  trap - ERR
  set +e
  command="${command//$'\n'/ }"
  command="${command:0:1000}"
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    command="${command//"$WANDB_API_KEY"/<redacted>}"
  fi
  xwam_write_failure_report "$exit_code" "$line" "$command"
  printf '[FAIL] X-WAM Clariden job exited: phase=%s exit_code=%s line=%s\n' \
    "${XWAM_PHASE:-unknown}" "$exit_code" "$line" >&2
  printf '[FAIL] command=%q\n' "$command" >&2
  printf '[FAIL] failure_report=%s\n' \
    "${XWAM_FAILURE_REPORT:-unavailable}" >&2
  exit "$exit_code"
}

xwam_install_err_trap() {
  trap 'xwam_on_err "$?" "$LINENO" "$BASH_COMMAND"' ERR
}

xwam_fail() {
  local message="$1"
  xwam_on_err 2 "${BASH_LINENO[0]:-unknown}" "$message"
}
