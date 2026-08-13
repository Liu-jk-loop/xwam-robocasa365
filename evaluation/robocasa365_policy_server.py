#!/usr/bin/env python3
"""加载 M3/M6 X-WAM checkpoint，并通过 broker 提供 RoboCasa365 12D 动作。"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import logging
import os
import platform
import random
import sys
import time
import traceback
from collections.abc import Collection
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.robocasa365_contract import resolve_lerobot_root  # noqa: E402
from data.robocasa365_multitask import (  # noqa: E402
    load_multitask_dataset_manifest,
)
from data.robocasa365_schema import PandaOmronTensorCodec  # noqa: E402
from evaluation.robocasa365_protocol import (  # noqa: E402
    PROTOCOL_NAME,
    decode_message,
    deterministic_inference_seed,
    encode_message,
    make_error_response,
    make_success_response,
)
from project_tools.training_run import (  # noqa: E402
    append_jsonl_fsync,
    collect_git_state,
    write_json_atomic,
)


DEFAULT_REPORT = REPO_ROOT / "logs" / "cluster" / "robocasa365_m4_policy_server.json"
DEFAULT_REQUEST_JOURNAL = (
    REPO_ROOT / "logs" / "cluster" / "robocasa365_m4_policy_server_requests.jsonl"
)


def validate_checkpoint_task(
    request_task: str, checkpoint_tasks: str | Collection[str]
) -> None:
    allowed = (
        {checkpoint_tasks}
        if isinstance(checkpoint_tasks, str)
        else {str(task) for task in checkpoint_tasks}
    )
    if request_task not in allowed:
        raise ValueError(
            "请求任务不属于 checkpoint 的训练任务："
            f"checkpoint_tasks={sorted(allowed)!r}, request={request_task!r}"
        )


def resolve_model_state_checkpoint(
    experiment_dir: str | Path, checkpoint: str | Path | None
) -> Path:
    experiment = Path(experiment_dir).expanduser().resolve()
    candidate = (
        Path(checkpoint).expanduser().resolve()
        if checkpoint is not None
        else experiment / "checkpoints" / "last.ckpt"
    )
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        for relative in (
            Path("checkpoint") / "mp_rank_00_model_states.pt",
            Path("mp_rank_00_model_states.pt"),
        ):
            resolved = candidate / relative
            if resolved.is_file():
                return resolved.resolve()
    raise FileNotFoundError(
        "找不到 DeepSpeed model state；checkpoint 应为 model_states.pt 或包含 "
        f"checkpoint/mp_rank_00_model_states.pt 的 .ckpt 目录：{candidate}"
    )


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True, help="包含 config.yaml/checkpoints 的 X-WAM 实验目录。")
    parser.add_argument("--checkpoint", help="DeepSpeed .ckpt 目录或 mp_rank_00_model_states.pt；默认 last.ckpt。")
    parser.add_argument("--wan-checkpoint-dir", help="覆盖 config 中的 Wan2.2-TI2V-5B 路径。")
    parser.add_argument("--dataset-path", help="覆盖训练配置中的 RoboCasa365 单任务数据路径，用于读取 stats。")
    parser.add_argument(
        "--multitask-manifest",
        help="M6 多任务训练 manifest；覆盖训练 config 中的 dataset.multitask_manifest。",
    )
    parser.add_argument(
        "--statistics-path",
        help="M6 跨任务 normalization statistics；覆盖 dataset.statistics_path。",
    )
    parser.add_argument(
        "--allowed-task",
        action="append",
        default=[],
        help="限制本 server 可服务的任务；可重复。默认允许 checkpoint 中全部训练任务。",
    )
    parser.add_argument("--schema-path", help="覆盖训练配置中的 PandaOmron schema。")
    parser.add_argument("--broker-address", default="127.0.0.1")
    parser.add_argument("--broker-port", type=int, default=10087)
    parser.add_argument("--denoise-steps", type=int, default=50)
    parser.add_argument("--action-denoise-steps", type=int, default=10)
    parser.add_argument(
        "--model-seed",
        type=int,
        help="固定每次request的模型采样seed；省略时保留M4的request-specific确定性seed。",
    )
    parser.add_argument("--max-requests", type=int, default=0, help="0 表示持续服务；smoke 建议 1。")
    parser.add_argument("--minimum-requests", type=int, default=0)
    parser.add_argument(
        "--graceful-stop-is-pass",
        action="store_true",
        help="长服务在满足 minimum requests 后由 Ctrl-C 正常停止时记为 pass。",
    )
    parser.add_argument("--compile-model", action="store_true", help="启用 torch.compile；首轮 smoke 默认关闭。")
    parser.add_argument("--startup-report", default=str(DEFAULT_REPORT))
    parser.add_argument("--request-journal", default=str(DEFAULT_REQUEST_JOURNAL))
    parser.add_argument(
        "--disable-request-journal",
        action="store_true",
        help="不写逐request JSONL；用于正式长评估以避免无用小记录。",
    )
    parser.add_argument(
        "--compact-report",
        action="store_true",
        help="最终server报告只保留失败request，不嵌入全部成功request。",
    )
    args = parser.parse_args()
    if not 1 <= args.broker_port <= 65535:
        parser.error("broker port 必须位于 1..65535")
    if args.denoise_steps <= 0 or args.action_denoise_steps <= 0:
        parser.error("denoise steps 必须为正")
    if args.action_denoise_steps > args.denoise_steps:
        parser.error("action denoise steps 不能超过 video denoise steps")
    if args.max_requests < 0 or args.minimum_requests < 0:
        parser.error("max/minimum requests 不能为负")
    if args.max_requests and args.minimum_requests > args.max_requests:
        parser.error("minimum requests 不能超过 max requests")
    if args.model_seed is not None and (
        args.model_seed < 0 or args.model_seed >= 2**32
    ):
        parser.error("model seed 必须位于 0..2^32-1")
    if len(args.allowed_task) != len(set(args.allowed_task)):
        parser.error("--allowed-task 不能重复")
    return args


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _load_checkpoint_data_contract(
    args: argparse.Namespace,
    config: Any,
    schema_path: str,
) -> tuple[PandaOmronTensorCodec, list[str], dict[str, Any]]:
    dataset_config = config.dataset
    raw_manifest = args.multitask_manifest or dataset_config.get("multitask_manifest")
    if raw_manifest:
        manifest_path = _resolve_repo_path(str(raw_manifest))
        atomic_manifest = _resolve_repo_path(str(dataset_config.get("task_manifest")))
        manifest = load_multitask_dataset_manifest(
            manifest_path,
            atomic_task_manifest=atomic_manifest,
            expected_task_count=int(dataset_config.get("expected_task_count", 18)),
        )
        statistics_value = args.statistics_path or dataset_config.get("statistics_path")
        if not statistics_value:
            raise ValueError("M6 多任务 checkpoint 必须提供跨任务 statistics_path")
        statistics_path = _resolve_repo_path(str(statistics_value))
        try:
            statistics_payload = json.loads(
                statistics_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"无法读取M6 global statistics：{statistics_path}: {exc}") from exc
        if not isinstance(statistics_payload, dict):
            raise ValueError("M6 global statistics 顶层必须为对象")
        if statistics_payload.get("ok") is not True or statistics_payload.get("result") != "pass":
            raise ValueError("M6 global statistics 尚未通过数据门禁")
        if statistics_payload.get("manifest_digest") != manifest.manifest_digest:
            raise ValueError(
                "M6 global statistics与训练manifest digest不一致："
                f"stats={statistics_payload.get('manifest_digest')}, "
                f"manifest={manifest.manifest_digest}"
            )
        checkpoint_tasks = [entry.task_name for entry in manifest.tasks]
        allowed_tasks = list(args.allowed_task) or checkpoint_tasks
        unknown = sorted(set(allowed_tasks) - set(checkpoint_tasks))
        if unknown:
            raise ValueError(f"--allowed-task 不属于 checkpoint manifest：{unknown}")
        # All entries have already passed the immutable M6 manifest contract.  One
        # real dataset is sufficient to revalidate the shared PandaOmron modality;
        # normalization itself always comes from the manifest-bound global stats.
        codec = PandaOmronTensorCodec.from_dataset(
            manifest.tasks[0].dataset_path,
            schema_path,
            statistics_path=statistics_path,
        )
        return codec, allowed_tasks, {
            "mode": "multitask",
            "multitask_manifest": str(manifest_path),
            "manifest_digest": manifest.manifest_digest,
            "statistics_path": str(statistics_path),
            "checkpoint_tasks": checkpoint_tasks,
        }

    if args.statistics_path:
        raise ValueError("单任务 checkpoint 不能单独覆盖 --statistics-path")
    if args.allowed_task:
        raise ValueError("单任务 checkpoint 不需要 --allowed-task")
    if args.dataset_path is not None:
        dataset_config.dataset_path = str(Path(args.dataset_path).expanduser().resolve())
    dataset_path = dataset_config.get("dataset_path")
    if not dataset_path:
        raise ValueError("必须通过训练 config 或 --dataset-path 提供 RoboCasa365 数据路径")
    task_name = dataset_config.get("task_name")
    if not isinstance(task_name, str) or not task_name:
        raise ValueError("单任务 checkpoint config 必须包含 dataset.task_name")
    codec = PandaOmronTensorCodec.from_dataset(dataset_path, schema_path)
    return codec, [task_name], {
        "mode": "single_task",
        "dataset_path": str(Path(dataset_path).expanduser().resolve()),
        "stats_path": str((resolve_lerobot_root(dataset_path) / "meta" / "stats.json").resolve()),
        "checkpoint_tasks": [task_name],
    }


def _load_runtime(
    args: argparse.Namespace,
) -> tuple[Any, PandaOmronTensorCodec, Path, dict[str, Any]]:
    import lightning as lightning
    import torch
    from omegaconf import OmegaConf

    from runners.xwam_runner import XWAMRunner

    if args.model_seed is not None:
        random.seed(args.model_seed)
        np.random.seed(args.model_seed)
        torch.manual_seed(args.model_seed)
        torch.cuda.manual_seed_all(args.model_seed)

    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    config_path = experiment_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"X-WAM 实验缺少 config.yaml：{config_path}")
    config = OmegaConf.load(config_path)
    if args.wan_checkpoint_dir is not None:
        config.wan_checkpoint_dir = str(Path(args.wan_checkpoint_dir).expanduser().resolve())
    if not config.get("wan_checkpoint_dir"):
        raise ValueError("必须通过 config 或 --wan-checkpoint-dir 提供 Wan2.2 路径")
    schema_path = args.schema_path or config.get("checkpoint_schema_path") or config.dataset.get("schema_path")
    if not schema_path:
        raise ValueError("必须提供 PandaOmron schema path")
    schema_path = str(_resolve_repo_path(schema_path))

    if str(config.dataset.get("format")) != "robocasa365_lerobot_v21":
        raise ValueError(f"M4.2 只允许 RoboCasa365 v2.1 dataset config：{config.dataset.get('format')!r}")
    if bool(config.get("use_depth")):
        raise ValueError("M4.2 首轮只允许 RGB-only / use_depth=false")
    if int(config.get("action_dim")) != 12 or int(config.get("proprio_dim")) != 16:
        raise ValueError("M4.2 要求 model action_dim=12、proprio_dim=16")
    frame_skip = int(config.dataset.frame_skip)
    action_skip = int(config.dataset.action_skip)
    if frame_skip <= 0 or action_skip <= 0 or frame_skip % action_skip != 0:
        raise ValueError("dataset frame_skip/action_skip 不能形成整数 action_num")
    config.action_num = frame_skip // action_skip
    config.sample_steps = int(args.denoise_steps)
    config.action_denoise_steps = int(args.action_denoise_steps)
    config.use_decoupled_inference = True
    config.use_gradient_checkpointing = False

    codec, allowed_tasks, data_contract = _load_checkpoint_data_contract(
        args, config, schema_path
    )
    checkpoint_path = resolve_model_state_checkpoint(experiment_dir, args.checkpoint)

    runner = XWAMRunner(config=config, run_depth=False).cuda().bfloat16()
    allow_missing_frozen = bool(config.get("deepspeed_exclude_frozen_parameters", False))
    if allow_missing_frozen:
        runner.enable_excluded_frozen_resume_loading()
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("module"), dict):
        raise ValueError("DeepSpeed model_states.pt 缺少 module state dict")
    runner.load_state_dict(payload["module"], strict=True)
    del payload
    gc.collect()
    runner.eval()
    runner.model.gradient_checkpointing = False
    if args.compile_model:
        runner.model = torch.compile(runner.model)

    torch.cuda.empty_cache()
    runtime = {
        "protocol": PROTOCOL_NAME,
        "config_path": str(config_path),
        # 保留 M4 单任务审计依赖的顶层字段；M6 完整记录在 data_contract 中。
        "task": allowed_tasks[0] if len(allowed_tasks) == 1 else None,
        "dataset_path": data_contract.get("dataset_path"),
        "stats_path": data_contract.get("stats_path") or data_contract.get("statistics_path"),
        "checkpoint": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "data_contract": data_contract,
        "tasks": allowed_tasks,
        "model_seed": int(args.model_seed) if args.model_seed is not None else None,
        "schema_path": schema_path,
        "schema_sha256": codec.schema.canonical_sha256,
        "action_num": int(config.action_num),
        "frame_num": int(config.frame_num),
        "predicted_action_steps": int((config.frame_num - 1) * config.action_num),
        "action_dim": int(config.action_dim),
        "proprio_dim": int(config.proprio_dim),
        "denoise_steps": int(config.sample_steps),
        "action_denoise_steps": int(config.action_denoise_steps),
        "compile_model": bool(args.compile_model),
        "excluded_frozen_load": getattr(runner, "_excluded_frozen_resume_report", None),
        "parameters": {
            "total": sum(parameter.numel() for parameter in runner.parameters()),
            "trainable": sum(parameter.numel() for parameter in runner.parameters() if parameter.requires_grad),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "lightning": getattr(lightning, "__version__", None),
            "numpy": np.__version__,
            "pyzmq": _package_version("pyzmq"),
            "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
            "cuda_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_reserved_bytes": torch.cuda.memory_reserved(),
        },
    }
    return runner, codec, checkpoint_path, runtime


def _prepare_inputs(request: dict[str, Any], codec: PandaOmronTensorCodec, config: Any) -> tuple[Any, Any, list[str]]:
    import torch
    import torch.nn.functional as functional

    video = torch.from_numpy(request["video"]).float().permute(0, 3, 1, 2).unsqueeze(0)
    video = video.div_(127.5).sub_(1.0)
    target_size = tuple(int(value) for value in config.dataset.video_size)
    if tuple(video.shape[-2:]) != target_size:
        flattened = video.flatten(0, 1)
        flattened = functional.interpolate(
            flattened,
            size=target_size,
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )
        video = flattened.unflatten(0, (1, 3))
    state_normalized = codec.encode_state(request["state"], clip=True)
    proprio = torch.from_numpy(state_normalized).unsqueeze(0)
    return video.bfloat16().cuda(), proprio.bfloat16().cuda(), [request["prompt"]]


def _serve(args: argparse.Namespace, report: dict[str, Any]) -> int:
    import torch
    import zmq

    runner, codec, checkpoint_path, runtime = _load_runtime(args)
    report.update(runtime)
    report.update(result="ready", ok=True, ready_at_utc=datetime.now(timezone.utc).isoformat())
    write_json_atomic(args.startup_report, report)
    logging.info("Policy runtime ready; report=%s", Path(args.startup_report).resolve())

    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.IDENTITY, f"robocasa365-policy-{os.getpid()}".encode("utf-8"))
    socket.connect(f"tcp://{args.broker_address}:{args.broker_port}")
    socket.send_multipart([b"READY"])
    processed = 0
    failed_requests = 0
    request_records: list[dict[str, Any]] = []
    interrupted = False
    try:
        while args.max_requests == 0 or processed < args.max_requests:
            frames = socket.recv_multipart()
            if len(frames) != 3 or frames[0] != b"WORK":
                raise RuntimeError(f"broker WORK frame 非法：count={len(frames)}")
            client_id, wire = frames[1], frames[2]
            request: dict[str, Any] | None = None
            try:
                request = decode_message(wire)
                if request["kind"] != "policy_request":
                    raise ValueError(f"server 收到非 request kind：{request['kind']}")
                validate_checkpoint_task(request["task"], runtime["tasks"])
                # The formal comparison fixes model-side sampling to seed 42 for
                # every replan.  Environment diversity remains controlled by the
                # per-episode RoboCasa seed carried in the request.
                inference_seed = (
                    int(args.model_seed)
                    if args.model_seed is not None
                    else deterministic_inference_seed(
                        request["task"], request["episode_id"], request["step_id"]
                    )
                )
                rgb, proprio, prompt = _prepare_inputs(request, codec, runner.config)
                torch.cuda.synchronize()
                started = time.monotonic()
                with torch.inference_mode():
                    _, actions, _, _ = runner.generate(
                        rgb,
                        proprio,
                        prompt,
                        seeds=[inference_seed],
                        early_stop=True,
                        cfg=float(request["cfg"]),
                        run_depth=False,
                    )
                torch.cuda.synchronize()
                inference_seconds = time.monotonic() - started
                normalized_actions = actions[0].detach().float().cpu().numpy()
                normalized_actions = np.clip(normalized_actions, -1.0, 1.0)
                environment_actions = codec.decode_action(
                    normalized_actions,
                    discretize_control_mode=True,
                )
                response = make_success_response(
                    request,
                    actions=environment_actions,
                    inference_seed=inference_seed,
                    inference_seconds=inference_seconds,
                    checkpoint=str(checkpoint_path),
                )
                logging.info(
                    "request complete id=%s step=%d actions=%s inference=%.3fs",
                    request["request_id"],
                    request["step_id"],
                    tuple(environment_actions.shape),
                    inference_seconds,
                )
                if not args.compact_report:
                    request_records.append(
                        {
                            "request_id": request["request_id"],
                            "task": request["task"],
                            "step_id": request["step_id"],
                            "result": "pass",
                            "inference_seed": inference_seed,
                            "inference_seconds": inference_seconds,
                            "action_shape": list(environment_actions.shape),
                        }
                    )
                if not args.disable_request_journal:
                    append_jsonl_fsync(
                        args.request_journal,
                        {
                        "schema_version": 1,
                        "event": "policy_request_complete",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "server_started_at_utc": report["created_at_utc"],
                        "git_commit": report.get("git", {}).get("commit"),
                        "request_id": request["request_id"],
                        "task": request["task"],
                        "episode_id": request["episode_id"],
                        "step_id": request["step_id"],
                        "result": "pass",
                        "inference_seed": inference_seed,
                        "inference_seconds": inference_seconds,
                        "action_shape": list(environment_actions.shape),
                        "checkpoint": str(checkpoint_path),
                        },
                    )
            except Exception as exc:
                logging.exception("policy request failed")
                failed_requests += 1
                request_records.append(
                    {
                        "request_id": request.get("request_id", "unknown")
                        if request
                        else "unknown",
                        "task": request.get("task", "unknown") if request else "unknown",
                        "step_id": request.get("step_id", -1) if request else -1,
                        "result": "fail",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if not args.disable_request_journal:
                    append_jsonl_fsync(
                        args.request_journal,
                        {
                        "schema_version": 1,
                        "event": "policy_request_complete",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "server_started_at_utc": report["created_at_utc"],
                        "git_commit": report.get("git", {}).get("commit"),
                        "request_id": request.get("request_id", "unknown")
                        if request
                        else "unknown",
                        "task": request.get("task", "unknown") if request else "unknown",
                        "episode_id": request.get("episode_id", "unknown")
                        if request
                        else "unknown",
                        "step_id": request.get("step_id", -1) if request else -1,
                        "result": "fail",
                        "error": f"{type(exc).__name__}: {exc}",
                        "checkpoint": str(checkpoint_path),
                        },
                    )
                response = make_error_response(
                    request,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            socket.send_multipart([b"RESULT", client_id, encode_message(response)])
            processed += 1
            if args.max_requests == 0 or processed < args.max_requests:
                socket.send_multipart([b"READY"])
    except KeyboardInterrupt:
        interrupted = True
        logging.info("policy server stopped by user")
    finally:
        socket.close()
        context.term()
    report["environment"].update(
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    )
    minimum_met = processed >= args.minimum_requests
    graceful_interrupt = interrupted and args.graceful_stop_is_pass and minimum_met
    ok = failed_requests == 0 and minimum_met and (not interrupted or graceful_interrupt)
    report.update(
        result="pass" if ok else ("stopped" if interrupted else "fail"),
        ok=ok,
        interrupted=interrupted,
        graceful_interrupt=graceful_interrupt,
        minimum_requests=args.minimum_requests,
        processed_requests=processed,
        failed_requests=failed_requests,
        request_records=request_records,
        finished_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_json_atomic(args.startup_report, report)
    logging.info(
        "policy server completed processed_requests=%d failed_requests=%d",
        processed,
        failed_requests,
    )
    return 0 if ok else 1


def main() -> int:
    args = _parse_args()
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": "loading",
        "ok": False,
        "git": collect_git_state(REPO_ROOT),
        "command": sys.argv,
        "broker": f"tcp://{args.broker_address}:{args.broker_port}",
        "request_journal": (
            None
            if args.disable_request_journal
            else str(Path(args.request_journal).expanduser().resolve())
        ),
    }
    write_json_atomic(args.startup_report, report)
    try:
        return _serve(args, report)
    except Exception as exc:
        report.update(
            result="fail",
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        write_json_atomic(args.startup_report, report)
        print(f"Policy server 启动失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
