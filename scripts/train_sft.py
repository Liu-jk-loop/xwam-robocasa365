import os
import sys
import logging
import importlib.metadata
import platform
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from pprint import pprint

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Subset
from omegaconf import OmegaConf
import lightning as L
from lightning.pytorch.callbacks import (
    Callback,
    ModelCheckpoint,
    ModelSummary,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.strategies import DeepSpeedStrategy

from data.dataset_factory import build_dataset
from data.epoch_aligned_sampler import EpochAlignedDistributedSampler
from project_tools.training_topology import (
    resolve_cpu_adam_options,
    resolve_deepspeed_options,
    resolve_optimizer_backend,
    resolve_training_topology,
)
from project_tools.training_run import (
    append_jsonl_fsync,
    collect_memory_snapshot,
    collect_git_state,
    resolve_resume_checkpoint,
    resolve_checkpoint_monitor,
    resolve_save_last,
    resolve_subset_indices,
    resolve_training_schedule,
    write_json_atomic,
)
from project_tools.h100_training import (
    load_epoch_schedule_from_manifest,
    validate_global_stats_contract,
    validate_m6_formal_training_contract,
)
from runners.xwam_runner import XWAMRunner
from utils.console_logger import ConsoleLogger
from utils.xwam_checkpoint_loader import initialize_xwam_runner


class ResourceAwareModelCheckpoint(ModelCheckpoint):
    """Persist memory evidence immediately before and after DeepSpeed saves."""

    def __init__(
        self,
        *args,
        diagnostics_path: Path,
        checkpoint_tier: str = "primary",
        **kwargs,
    ):
        self.diagnostics_path = Path(diagnostics_path)
        self.checkpoint_tier = str(checkpoint_tier)
        super().__init__(*args, **kwargs)

    def _record_checkpoint_event(self, event, trainer, filepath, error=None):
        if not trainer.is_global_zero:
            return
        payload = {
            "event": event,
            "filepath": str(filepath),
            "global_step": int(trainer.global_step),
            "checkpoint_tier": self.checkpoint_tier,
            "error": error,
            "memory": collect_memory_snapshot(),
        }
        append_jsonl_fsync(self.diagnostics_path, payload)
        rss_bytes = payload["memory"]["process"].get("VmRSS")
        print(
            "Checkpoint resource event: "
            f"event={event}, tier={self.checkpoint_tier}, "
            f"global_step={trainer.global_step}, "
            f"rss_bytes={rss_bytes}, diagnostics={self.diagnostics_path}",
            flush=True,
        )

    def _save_checkpoint(self, trainer, filepath):
        self._record_checkpoint_event("checkpoint_save_start", trainer, filepath)
        try:
            super()._save_checkpoint(trainer, filepath)
        except BaseException as exc:
            self._record_checkpoint_event(
                "checkpoint_save_error",
                trainer,
                filepath,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        self._record_checkpoint_event("checkpoint_save_complete", trainer, filepath)


class OptimizerStateDtypeAudit(Callback):
    """Write the actual optimizer-state tensor dtypes after the first update."""

    def __init__(self, output_dir: Path, *, require_fp32: bool):
        self.output_dir = Path(output_dir)
        self.require_fp32 = bool(require_fp32)
        self.written = False

    @staticmethod
    def _optimizer_candidates(optimizer):
        queue = [("trainer_optimizer", optimizer)]
        seen = set()
        while queue:
            name, candidate = queue.pop(0)
            if id(candidate) in seen:
                continue
            seen.add(id(candidate))
            yield name, candidate
            for attribute in ("optimizer", "basic_optimizer"):
                nested = getattr(candidate, attribute, None)
                if nested is not None:
                    queue.append((f"{name}.{attribute}", nested))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.written or int(trainer.global_step) < 1 or not trainer.optimizers:
            return
        dtype_counts = {}
        candidate_reports = []
        for name, candidate in self._optimizer_candidates(trainer.optimizers[0]):
            state = getattr(candidate, "state", None)
            tensor_count = 0
            if isinstance(state, dict):
                for item in state.values():
                    if not isinstance(item, dict):
                        continue
                    for value in item.values():
                        if torch.is_tensor(value) and value.is_floating_point():
                            dtype_name = str(value.dtype)
                            dtype_counts[dtype_name] = (
                                dtype_counts.get(dtype_name, 0) + 1
                            )
                            tensor_count += 1
            candidate_reports.append(
                {
                    "name": name,
                    "class": type(candidate).__name__,
                    "floating_state_tensors": tensor_count,
                }
            )
        floating_count = sum(dtype_counts.values())
        fp32_only = floating_count > 0 and set(dtype_counts) == {"torch.float32"}
        payload = {
            "schema_version": 1,
            "global_rank": int(trainer.global_rank),
            "world_size": int(trainer.world_size),
            "global_step": int(trainer.global_step),
            "require_fp32": self.require_fp32,
            "dtype_counts": dtype_counts,
            "floating_state_tensors": floating_count,
            "fp32_only": fp32_only,
            "candidates": candidate_reports,
            "result": "pass" if (not self.require_fp32 or fp32_only) else "fail",
        }
        path = self.output_dir / f"optimizer_state_rank_{trainer.global_rank:03d}.json"
        write_json_atomic(path, payload)
        print(f"Optimizer state dtype audit: {path} ({payload['result']})", flush=True)
        self.written = True


def _formal_guard_enabled(config):
    return bool(
        config.get("m6_formal_guard", False) or config.get("h100_formal_guard", False)
    )


def _validate_formal_runtime(config, topology):
    if not _formal_guard_enabled(config):
        return None
    expected_accelerator = str(config.get("formal_accelerator", "H100")).upper()
    minimum_memory_gib = float(config.get("formal_minimum_memory_gib", 75.0))
    expected_world_size = int(config.get("formal_world_size", 4))
    expected_num_nodes = int(config.get("formal_num_nodes", 1))
    expected_devices_per_node = int(config.get("formal_devices_per_node", 4))
    names = [
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    ]
    memory_gib = [
        torch.cuda.get_device_properties(index).total_memory / (1024**3)
        for index in range(torch.cuda.device_count())
    ]
    checks = {
        "visible_gpu_count": len(names) == expected_devices_per_node,
        "world_size": int(topology["world_size"]) == expected_world_size,
        "num_nodes": int(topology["num_nodes"]) == expected_num_nodes,
        "accelerator_names": len(names) == expected_devices_per_node
        and all(expected_accelerator in name.upper() for name in names),
        "minimum_memory": len(memory_gib) == expected_devices_per_node
        and all(value >= minimum_memory_gib for value in memory_gib),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "M6 formal runtime门禁失败："
            f"{failed}; expected={expected_accelerator}, "
            f"gpu_names={names}, memory_gib={memory_gib}"
        )
    return {
        "checks": checks,
        "accelerator": expected_accelerator,
        "minimum_memory_gib": minimum_memory_gib,
        "expected_world_size": expected_world_size,
        "expected_num_nodes": expected_num_nodes,
        "expected_devices_per_node": expected_devices_per_node,
        "gpu_names": names,
        "memory_gib": memory_gib,
    }


def _load_config():
    overrides = OmegaConf.from_cli()
    model_config_path = overrides.pop("model_config", "configs/model/wan22_5b_sft.yaml")
    model_config = OmegaConf.load(model_config_path)
    config_sources = {"model": str(Path(model_config_path).resolve())}

    dataset_override = overrides.get("dataset")
    if isinstance(dataset_override, str):
        dataset_name = dataset_override
        del overrides["dataset"]
    else:
        dataset_name = str(model_config.dataset)
    data_config_path = overrides.pop("data_config", f"configs/data/{dataset_name}.yaml")
    model_config["dataset"] = OmegaConf.load(data_config_path)
    config_sources["data"] = str(Path(data_config_path).resolve())

    layers = [model_config]
    for key, source_name in (
        ("hardware_config", "hardware"),
        ("experiment_config", "experiment"),
    ):
        config_path = overrides.pop(key, None)
        if config_path is not None:
            if not isinstance(config_path, str):
                raise TypeError(f"{key} 必须是 YAML 路径字符串")
            layers.append(OmegaConf.load(config_path))
            config_sources[source_name] = str(Path(config_path).resolve())

    config = OmegaConf.merge(*layers, overrides)
    config["config_sources"] = config_sources
    return config


def _loader_worker_options(num_workers: int, prefetch_factor: int) -> dict:
    if num_workers <= 0:
        return {}
    return {
        "prefetch_factor": int(prefetch_factor),
        "multiprocessing_context": "forkserver",
    }


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _resolve_trainer_topology(config):
    return resolve_training_topology(
        visible_devices=torch.cuda.device_count(),
        requested_devices=config.get("devices", "auto"),
        world_size=os.environ.get("WORLD_SIZE"),
    )


def _global_rank_from_environment() -> int:
    for name in ("RANK", "GLOBAL_RANK", "LOCAL_RANK"):
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return 0


def main():
    config = _load_config()
    topology = _resolve_trainer_topology(config)
    formal_contract = None
    stats_contract = None
    if _formal_guard_enabled(config):
        formal_contract = validate_m6_formal_training_contract(
            config, world_size=int(topology["world_size"])
        )
        schedule = load_epoch_schedule_from_manifest(
            config.dataset.multitask_manifest,
            global_batch_size=int(config.global_batch_size),
            num_train_epochs=int(config.num_train_epochs),
            trainer_max_steps=config.get("trainer_max_steps"),
        )
        stats_contract = validate_global_stats_contract(
            config.dataset.statistics_path,
            manifest_digest=str(schedule["manifest_digest"]),
        )
        config.num_training_steps = int(schedule["num_training_steps"])
        config.steps_per_epoch = int(schedule["steps_per_epoch"])
    else:
        schedule = resolve_training_schedule(
            num_training_steps=config.num_training_steps,
            trainer_max_steps=config.get("trainer_max_steps"),
        )
    config.trainer_max_steps = int(schedule["trainer_max_steps"])
    resume_checkpoint = resolve_resume_checkpoint(config.get("resume_checkpoint"))
    config.resume_checkpoint = resume_checkpoint
    if (
        bool(config.get("persist_generator_state", False))
        and topology["world_size"] != 1
        and not bool(config.get("allow_distributed_generator_state", False))
    ):
        raise ValueError(
            "多卡 persist_generator_state 必须显式启用 allow_distributed_generator_state"
        )
    deepspeed_options = resolve_deepspeed_options(config)
    allow_missing_frozen_resume_parameters = bool(
        resume_checkpoint is not None and deepspeed_options["exclude_frozen_parameters"]
    )

    runtime_rank = _global_rank_from_environment()
    if runtime_rank == 0:
        pprint(OmegaConf.to_container(config))

    exp_dir = Path(config.exp_root) / config.exp_name
    run_id = os.environ.get("XWAM_RUN_ID")
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        os.environ["XWAM_RUN_ID"] = run_id
    run_dir = exp_dir / "runs"
    exp_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = run_dir / f"{run_id}_config.yaml"
    if runtime_rank == 0:
        OmegaConf.save(
            config,
            exp_dir / "config.yaml",
            resolve=True,
        )
        OmegaConf.save(
            config,
            resolved_config_path,
            resolve=True,
        )
    L.seed_everything(config.seed, workers=True)
    formal_runtime = _validate_formal_runtime(config, topology)

    callbacks = [
        ModelSummary(max_depth=2),
        LearningRateMonitor(logging_interval="step"),
    ]
    checkpoint_callback = None
    durable_checkpoint_callback = None
    checkpoint_events_path = run_dir / f"{run_id}_checkpoint_events.jsonl"
    optimizer_audit_dir = run_dir / f"{run_id}_optimizer_state"
    if bool(config.get("audit_optimizer_state_dtype", False)):
        callbacks.append(
            OptimizerStateDtypeAudit(
                optimizer_audit_dir,
                require_fp32=bool(config.get("deepspeed_fp32_optimizer_states", True)),
            )
        )
    if bool(config.get("enable_checkpointing", True)):
        checkpoint_dir = Path(
            config.get("checkpoint_dir") or (exp_dir / "checkpoints")
        ).expanduser()
        rolling_save_top_k = int(config.get("save_top_k", -1))
        rolling_monitor = resolve_checkpoint_monitor(rolling_save_top_k)
        checkpoint_callback = ResourceAwareModelCheckpoint(
            diagnostics_path=checkpoint_events_path,
            checkpoint_tier="rolling",
            dirpath=checkpoint_dir,
            save_top_k=rolling_save_top_k,
            save_last=resolve_save_last(config.get("save_last", True)),
            save_weights_only=False,
            save_on_exception=bool(config.get("save_on_exception", False)),
            every_n_train_steps=int(config.save_interval),
            enable_version_counter=False,
            **rolling_monitor,
        )
        callbacks.insert(
            1,
            checkpoint_callback,
        )
        durable_checkpoint_dir_value = config.get("durable_checkpoint_dir")
        if durable_checkpoint_dir_value:
            durable_checkpoint_dir = Path(
                str(durable_checkpoint_dir_value)
            ).expanduser()
            if durable_checkpoint_dir.resolve() == checkpoint_dir.resolve():
                raise ValueError("滚动与永久checkpoint目录必须不同")
            durable_save_top_k = int(config.get("durable_save_top_k", -1))
            durable_monitor = resolve_checkpoint_monitor(durable_save_top_k)
            durable_checkpoint_callback = ResourceAwareModelCheckpoint(
                diagnostics_path=checkpoint_events_path,
                checkpoint_tier="durable",
                dirpath=durable_checkpoint_dir,
                save_top_k=durable_save_top_k,
                save_last=resolve_save_last(config.get("durable_save_last", False)),
                save_weights_only=False,
                save_on_exception=False,
                every_n_train_steps=int(config.durable_save_interval),
                enable_version_counter=False,
                **durable_monitor,
            )
            callbacks.insert(2, durable_checkpoint_callback)
        final_checkpoint_dir = Path(
            config.get("final_checkpoint_dir")
            or durable_checkpoint_dir_value
            or checkpoint_dir
        ).expanduser()
        checkpoint_contract = {
            "rolling": {
                "directory": str(checkpoint_dir.resolve()),
                "interval_steps": int(config.save_interval),
                "save_top_k": rolling_save_top_k,
                "save_last": config.get("save_last", True),
                "monitor": rolling_monitor.get("monitor"),
                "mode": rolling_monitor.get("mode"),
            },
            "durable": (
                {
                    "directory": str(durable_checkpoint_dir.resolve()),
                    "interval_steps": int(config.durable_save_interval),
                    "save_top_k": durable_save_top_k,
                    "save_last": config.get("durable_save_last", False),
                    "monitor": durable_monitor.get("monitor"),
                    "mode": durable_monitor.get("mode"),
                }
                if durable_checkpoint_callback is not None
                else None
            ),
            "final_directory": str(final_checkpoint_dir.resolve()),
        }
    else:
        final_checkpoint_dir = exp_dir / "checkpoints"
        checkpoint_contract = None

    tb_path = os.getenv("TENSORBOARD_LOG_PATH", None)
    loggers = [ConsoleLogger(max_steps=schedule["trainer_max_steps"])]
    wandb_contract = None
    if bool(config.get("enable_wandb", False)):
        wandb_run_id = os.environ.get("WANDB_RUN_ID") or config.get("wandb_run_id")
        if not wandb_run_id:
            raise ValueError("enable_wandb=true 时必须提供持久化 WANDB_RUN_ID")
        wandb_mode = str(
            os.environ.get("WANDB_MODE") or config.get("wandb_mode", "online")
        ).lower()
        if wandb_mode not in {"online", "offline"}:
            raise ValueError(f"wandb_mode 只允许 online/offline，当前为 {wandb_mode}")
        wandb_require_api_key = bool(config.get("wandb_require_api_key", False))
        wandb_api_key_present = bool(os.environ.get("WANDB_API_KEY"))
        if wandb_require_api_key and not wandb_api_key_present:
            raise ValueError(
                "wandb_require_api_key=true 时必须通过环境变量提供 WANDB_API_KEY"
            )
        wandb_project = str(
            os.environ.get("WANDB_PROJECT")
            or config.get("wandb_project", "xwam-robocasa365")
        )
        wandb_name = str(config.get("wandb_name", config.exp_name))
        wandb_entity_value = os.environ.get("WANDB_ENTITY") or config.get(
            "wandb_entity"
        )
        wandb_entity = (
            str(wandb_entity_value).strip() if wandb_entity_value is not None else None
        )
        wandb_group = config.get("wandb_group")
        wandb_save_dir = os.environ.get("WANDB_DIR") or os.path.join(
            config.exp_root, config.exp_name, "wandb"
        )
        loggers.insert(
            0,
            WandbLogger(
                project=wandb_project,
                name=wandb_name,
                version=str(wandb_run_id),
                save_dir=wandb_save_dir,
                entity=wandb_entity,
                group=wandb_group,
                offline=wandb_mode == "offline",
                resume="allow",
                log_model=bool(config.get("wandb_log_model", False)),
            ),
        )
        wandb_contract = {
            "enabled": True,
            "project": wandb_project,
            "entity": wandb_entity,
            "name": wandb_name,
            "group": wandb_group,
            "run_id": str(wandb_run_id),
            "mode": wandb_mode,
            "resume": "allow",
            "auth": "api_key_env" if wandb_api_key_present else "wandb_default",
            "save_dir": str(Path(wandb_save_dir).expanduser().resolve()),
        }
    if bool(config.get("enable_tensorboard", True)):
        loggers.insert(
            0,
            TensorBoardLogger(
                save_dir=tb_path
                if tb_path
                else os.path.join(config.exp_root, config.exp_name, "tb_logs"),
                name=config.exp_name,
            ),
        )
    logging.getLogger("lightning.pytorch").setLevel(logging.INFO)

    base_train_dataset = build_dataset(config.dataset, use_depth=config.use_depth)
    dataset_provenance = (
        base_train_dataset.provenance()
        if callable(getattr(base_train_dataset, "provenance", None))
        else None
    )
    if dataset_provenance is not None:
        print(f"Training dataset provenance: {dataset_provenance}")
    config.action_num = base_train_dataset.action_num
    if _formal_guard_enabled(config):
        expected_samples = int(schedule["total_samples"])
        if len(base_train_dataset) != expected_samples:
            raise ValueError(
                "M6 manifest样本数与实际Dataset不一致："
                f"manifest={expected_samples}, dataset={len(base_train_dataset)}"
            )
    if int(config.action_dim) != int(base_train_dataset.action_dim):
        raise ValueError(
            f"模型 action_dim={config.action_dim} 与数据 action_dim={base_train_dataset.action_dim} 不一致"
        )
    if int(config.proprio_dim) != int(base_train_dataset.proprio_dim):
        raise ValueError(
            f"模型 proprio_dim={config.proprio_dim} 与数据 proprio_dim={base_train_dataset.proprio_dim} 不一致"
        )

    subset_indices = resolve_subset_indices(
        dataset_length=len(base_train_dataset),
        subset_size=config.get("train_subset_size"),
        subset_start=int(config.get("train_subset_start", 0)),
    )
    train_dataset = (
        base_train_dataset
        if subset_indices is None
        else Subset(base_train_dataset, subset_indices)
    )
    train_shuffle = bool(config.get("train_shuffle", True))
    train_sampler = None
    sampler_provenance = None
    if _formal_guard_enabled(config):
        samples_per_rank = (
            int(schedule["steps_per_epoch"])
            * int(config.batch_size_per_gpu)
            * int(config.accumulate_grad_batches)
        )
        train_sampler = EpochAlignedDistributedSampler(
            train_dataset,
            num_replicas=int(topology["world_size"]),
            rank=_global_rank_from_environment(),
            samples_per_rank=samples_per_rank,
            seed=int(config.seed),
        )
        sampler_provenance = train_sampler.provenance()
        train_shuffle = False
    print(
        "Training data selection: "
        f"base_samples={len(base_train_dataset)}, selected_samples={len(train_dataset)}, "
        f"subset_indices={subset_indices}, shuffle={train_shuffle}"
    )
    if runtime_rank == 0:
        OmegaConf.save(config, exp_dir / "config.yaml", resolve=True)
        OmegaConf.save(config, resolved_config_path, resolve=True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size_per_gpu,
        num_workers=config.num_workers_per_gpu,
        shuffle=train_shuffle,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True,
        **_loader_worker_options(config.num_workers_per_gpu, config.prefetch_factor),
    )

    val_dataloader = None
    if float(config.limit_val_batches) > 0:
        val_dataset = build_dataset(
            config.dataset, use_depth=config.use_depth, augment=False
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=config.num_workers_per_gpu,
            shuffle=True,
            pin_memory=True,
            **_loader_worker_options(
                config.num_workers_per_gpu, config.prefetch_factor
            ),
        )

    run_metadata_path = run_dir / f"{run_id}_metadata.json"
    run_result_path = run_dir / f"{run_id}_result.json"
    if runtime_rank == 0:
        write_json_atomic(
            run_metadata_path,
            {
                "schema_version": 1,
                "run_id": run_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "git": collect_git_state(Path(__file__).resolve().parents[1]),
                "command": sys.argv,
                "config_sources": OmegaConf.to_container(
                    config.config_sources, resolve=True
                ),
                "resolved_config": str(resolved_config_path.resolve()),
                "dataset": {
                    "path": config.dataset.get("dataset_path"),
                    "task_name": config.dataset.get("task_name"),
                    "base_samples": len(base_train_dataset),
                    "selected_samples": len(train_dataset),
                    "subset_indices": list(subset_indices)
                    if subset_indices is not None
                    else None,
                    "shuffle": train_shuffle,
                    "task_manifest": config.dataset.get("task_manifest"),
                    "schema_path": config.dataset.get("schema_path"),
                    "use_depth": bool(config.use_depth),
                    "adapter_provenance": dataset_provenance,
                    "sampler_provenance": sampler_provenance,
                },
                "checkpoint": {
                    "initialization_mode": config.get("initialization_mode"),
                    "pretrained_checkpoint": config.get("pretrained_checkpoint"),
                    "resume_checkpoint": resume_checkpoint,
                    "events": str(checkpoint_events_path.resolve()),
                    "allow_missing_frozen_resume_parameters": (
                        allow_missing_frozen_resume_parameters
                    ),
                    "storage": checkpoint_contract,
                },
                "environment": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "cuda_runtime": torch.version.cuda,
                    "lightning": getattr(L, "__version__", None),
                    "deepspeed": _package_version("deepspeed"),
                    "gpu_names": [
                        torch.cuda.get_device_name(index)
                        for index in range(torch.cuda.device_count())
                    ],
                },
                "training": schedule,
                "resume_checkpoint": resume_checkpoint,
                "deepspeed": deepspeed_options,
                "optimizer": {
                    "backend": resolve_optimizer_backend(config),
                    **resolve_cpu_adam_options(config),
                },
                "tracking": {"wandb": wandb_contract},
                "topology": topology,
                "formal_contract": formal_contract,
                "formal_runtime": formal_runtime,
                "global_stats_contract": stats_contract,
                "optimizer_state_audit_dir": str(optimizer_audit_dir.resolve()),
                "memory_at_metadata": collect_memory_snapshot(),
                "result": "pending",
            },
        )
        print(f"Run metadata: {run_metadata_path}")

    model = XWAMRunner(config, run_depth=bool(config.use_depth))
    if allow_missing_frozen_resume_parameters:
        model.enable_excluded_frozen_resume_loading()
    initialization_report = config.get("checkpoint_initialization_report")
    if initialization_report is None:
        initialization_report = str(
            run_dir / f"{run_id}_checkpoint_initialization_rank_{runtime_rank:03d}.json"
        )
    if resume_checkpoint is None:
        report = initialize_xwam_runner(
            model,
            mode=str(config.get("initialization_mode", "legacy_strict")),
            checkpoint=config.get("pretrained_checkpoint"),
            schema_path=config.get("checkpoint_schema_path"),
            report_path=initialization_report,
        )
    else:
        report = {
            "mode": "resume_checkpoint",
            "result": "deferred_to_trainer",
        }
    print(
        "Checkpoint initialization: "
        f"mode={report['mode']}, result={report['result']}, "
        f"report={initialization_report if resume_checkpoint is None else resume_checkpoint}"
    )

    print(
        "Runner v2, "
        f"num_nodes: {topology['num_nodes']}, world_size: {topology['world_size']}, "
        f"trainer_devices: {topology['trainer_devices']}, "
        f"visible_devices: {topology['visible_devices']}"
    )
    print(f"DeepSpeed options: {deepspeed_options}")

    trainer = L.Trainer(
        accelerator="auto",
        devices=topology["trainer_devices"],
        strategy=DeepSpeedStrategy(**deepspeed_options),
        precision=str(config.get("precision", "bf16-mixed")),
        num_nodes=topology["num_nodes"],
        max_steps=schedule["trainer_max_steps"],
        accumulate_grad_batches=config.accumulate_grad_batches,
        gradient_clip_val=config.gradient_clip_val,
        gradient_clip_algorithm=config.gradient_clip_algorithm,
        enable_progress_bar=False,
        enable_checkpointing=bool(config.get("enable_checkpointing", True)),
        callbacks=callbacks,
        logger=loggers,
        val_check_interval=config.val_interval,
        check_val_every_n_epoch=None,
        limit_val_batches=config.limit_val_batches,
        log_every_n_steps=config.log_interval,
        default_root_dir=os.path.join(config.exp_root, config.exp_name),
        use_distributed_sampler=train_sampler is None,
    )
    torch.cuda.reset_peak_memory_stats()
    fit_error = None
    final_checkpoint_path = None
    fit_started = time.monotonic()
    try:
        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=resume_checkpoint,
        )
        if bool(config.get("save_final_checkpoint", False)):
            final_checkpoint_path = str(
                (
                    final_checkpoint_dir / f"final-step={trainer.global_step}.ckpt"
                ).resolve()
            )
            if trainer.is_global_zero:
                append_jsonl_fsync(
                    checkpoint_events_path,
                    {
                        "event": "final_checkpoint_save_start",
                        "filepath": final_checkpoint_path,
                        "global_step": int(trainer.global_step),
                        "checkpoint_tier": "final_durable",
                        "memory": collect_memory_snapshot(),
                    },
                )
            trainer.strategy.barrier("before_final_checkpoint")
            trainer.save_checkpoint(final_checkpoint_path, weights_only=False)
            trainer.strategy.barrier("after_final_checkpoint")
            if trainer.is_global_zero:
                append_jsonl_fsync(
                    checkpoint_events_path,
                    {
                        "event": "final_checkpoint_save_complete",
                        "filepath": final_checkpoint_path,
                        "global_step": int(trainer.global_step),
                        "checkpoint_tier": "final_durable",
                        "memory": collect_memory_snapshot(),
                    },
                )
    except BaseException as exc:
        fit_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        gib = 1024**3
        allocated_gib = torch.cuda.max_memory_allocated() / gib
        reserved_gib = torch.cuda.max_memory_reserved() / gib
        print(
            "CUDA peak memory: "
            f"allocated={allocated_gib:.3f} GiB, "
            f"reserved={reserved_gib:.3f} GiB"
        )
        result_payload = {
            "schema_version": 1,
            "run_id": run_id,
            "result": "pass" if fit_error is None else "fail",
            "error": fit_error,
            "global_step": int(trainer.global_step),
            "trainer_max_steps": schedule["trainer_max_steps"],
            "elapsed_seconds": time.monotonic() - fit_started,
            "process_max_rss_raw": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ),
            "process_max_rss_platform": sys.platform,
            "cuda_peak_allocated_gib": allocated_gib,
            "cuda_peak_reserved_gib": reserved_gib,
            "resume_checkpoint": resume_checkpoint,
            "resume_module_load": getattr(
                model, "_excluded_frozen_resume_report", None
            ),
            "last_checkpoint": (
                final_checkpoint_path
                or (
                    checkpoint_callback.last_model_path
                    if checkpoint_callback is not None
                    else None
                )
            ),
            "best_checkpoint": (
                checkpoint_callback.best_model_path
                if checkpoint_callback is not None
                else None
            ),
            "checkpoint_events": str(checkpoint_events_path.resolve()),
            "memory_at_result": collect_memory_snapshot(),
            "optimizer_state_audit_dir": str(optimizer_audit_dir.resolve()),
        }
        if trainer.is_global_zero:
            write_json_atomic(run_result_path, result_payload)
            print(f"Run result: {run_result_path} ({result_payload['result']})")


if __name__ == "__main__":
    main()
