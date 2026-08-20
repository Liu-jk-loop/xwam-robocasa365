#!/usr/bin/env python3
"""将PASS preflight中的epoch调度导出为Slurm可安全source的整数环境文件。"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def resolve_epoch_schedule(
    preflight_path: str | Path,
    *,
    expected_epochs: int,
    milestone_epoch: int,
) -> dict[str, int]:
    path = Path(preflight_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    schedule = payload.get("schedule") or {}
    epochs = int(schedule.get("num_train_epochs", -1))
    steps_per_epoch = int(schedule.get("steps_per_epoch", -1))
    total_steps = int(schedule.get("num_training_steps", -1))
    if payload.get("ok") is not True or payload.get("result") != "pass":
        raise ValueError(f"preflight不是PASS：{path}")
    if schedule.get("schedule_mode") != "epochs":
        raise ValueError("8-epoch训练必须使用epochs调度，不能使用fixed_steps")
    if epochs != int(expected_epochs) or epochs <= 0:
        raise ValueError(f"epoch数不一致：{epochs} != {expected_epochs}")
    if not 0 < int(milestone_epoch) < epochs:
        raise ValueError("里程碑epoch必须位于完整训练区间内")
    if steps_per_epoch <= 0 or total_steps != steps_per_epoch * epochs:
        raise ValueError(
            f"epoch调度不自洽：steps_per_epoch={steps_per_epoch}, "
            f"epochs={epochs}, total_steps={total_steps}"
        )
    return {
        "XWAM_SCHEDULE_NUM_EPOCHS": epochs,
        "XWAM_SCHEDULE_STEPS_PER_EPOCH": steps_per_epoch,
        "XWAM_SCHEDULE_TOTAL_STEPS": total_steps,
        "XWAM_SCHEDULE_MILESTONE_EPOCH": int(milestone_epoch),
        "XWAM_SCHEDULE_MILESTONE_STEP": steps_per_epoch * int(milestone_epoch),
    }


def write_schedule_env(output_path: str | Path, schedule: dict[str, int]) -> Path:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for key, value in schedule.items():
            handle.write(f"{key}={int(value)}\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--expected-epochs", type=int, required=True)
    parser.add_argument("--milestone-epoch", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    schedule = resolve_epoch_schedule(
        args.preflight,
        expected_epochs=args.expected_epochs,
        milestone_epoch=args.milestone_epoch,
    )
    output = write_schedule_env(args.output, schedule)
    print(json.dumps(schedule, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Epoch schedule env: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
