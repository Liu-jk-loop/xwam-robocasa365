import os
import gc
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
    ModelCheckpoint,
    ModelSummary,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DeepSpeedStrategy

from data.dataset_factory import build_dataset
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
    resolve_save_last,
    resolve_subset_indices,
    resolve_training_schedule,
    write_json_atomic,
)
from runners.xwam_runner import XWAMRunner
from utils.console_logger import ConsoleLogger
from utils.xwam_checkpoint_loader import initialize_xwam_runner


class ResourceAwareModelCheckpoint(ModelCheckpoint):
    """Persist memory evidence immediately before and after DeepSpeed saves."""

    def __init__(self, *args, diagnostics_path: Path, **kwargs):
        self.diagnostics_path = Path(diagnostics_path)
        super().__init__(*args, **kwargs)

    def _record_checkpoint_event(self, event, trainer, filepath, error=None):
        payload = {
            "event": event,
            "filepath": str(filepath),
            "global_step": int(trainer.global_step),
            "error": error,
            "memory": collect_memory_snapshot(),
        }
        append_jsonl_fsync(self.diagnostics_path, payload)
        rss_bytes = payload["memory"]["process"].get("VmRSS")
        print(
            "Checkpoint resource event: "
            f"event={event}, global_step={trainer.global_step}, "
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


def main():
    config = _load_config()
    schedule = resolve_training_schedule(
        num_training_steps=config.num_training_steps,
        trainer_max_steps=config.get("trainer_max_steps"),
    )
    config.trainer_max_steps = schedule["trainer_max_steps"]
    resume_checkpoint = resolve_resume_checkpoint(config.get("resume_checkpoint"))
    config.resume_checkpoint = resume_checkpoint
    topology = _resolve_trainer_topology(config)
    if bool(config.get("persist_generator_state", False)) and topology["world_size"] != 1:
        raise ValueError("persist_generator_state 当前只允许 M3 单 GPU 确定性恢复门禁")
    deepspeed_options = resolve_deepspeed_options(config)
    allow_missing_frozen_resume_parameters = bool(
        resume_checkpoint is not None
        and deepspeed_options["exclude_frozen_parameters"]
    )

    pprint(OmegaConf.to_container(config))

    exp_dir = Path(config.exp_root) / config.exp_name
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = exp_dir / "runs"
    exp_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(
        config,
        exp_dir / "config.yaml",
        resolve=True,
    )
    resolved_config_path = run_dir / f"{run_id}_config.yaml"
    OmegaConf.save(
        config,
        resolved_config_path,
        resolve=True,
    )
    L.seed_everything(config.seed, workers=True)

    callbacks = [ModelSummary(max_depth=2), LearningRateMonitor(logging_interval="step")]
    checkpoint_callback = None
    checkpoint_events_path = run_dir / f"{run_id}_checkpoint_events.jsonl"
    if bool(config.get("enable_checkpointing", True)):
        checkpoint_callback = ResourceAwareModelCheckpoint(
            diagnostics_path=checkpoint_events_path,
            dirpath=exp_dir / "checkpoints",
            save_top_k=int(config.get("save_top_k", -1)),
            save_last=resolve_save_last(config.get("save_last", True)),
            save_weights_only=False,
            save_on_exception=bool(config.get("save_on_exception", False)),
            every_n_train_steps=int(config.save_interval),
            enable_version_counter=False,
        )
        callbacks.insert(
            1,
            checkpoint_callback,
        )

    tb_path = os.getenv("TENSORBOARD_LOG_PATH", None)
    loggers = [ConsoleLogger(max_steps=schedule["trainer_max_steps"])]
    if bool(config.get("enable_tensorboard", True)):
        loggers.insert(
            0,
            TensorBoardLogger(
                save_dir=tb_path if tb_path else os.path.join(config.exp_root, config.exp_name, "tb_logs"),
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
    print(
        "Training data selection: "
        f"base_samples={len(base_train_dataset)}, selected_samples={len(train_dataset)}, "
        f"subset_indices={subset_indices}, shuffle={train_shuffle}"
    )
    OmegaConf.save(config, exp_dir / "config.yaml", resolve=True)
    OmegaConf.save(config, resolved_config_path, resolve=True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size_per_gpu,
        num_workers=config.num_workers_per_gpu,
        shuffle=train_shuffle,
        pin_memory=True,
        drop_last=True,
        **_loader_worker_options(config.num_workers_per_gpu, config.prefetch_factor),
    )

    val_dataloader = None
    if float(config.limit_val_batches) > 0:
        val_dataset = build_dataset(config.dataset, use_depth=config.use_depth, augment=False)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=config.num_workers_per_gpu,
            shuffle=True,
            pin_memory=True,
            **_loader_worker_options(config.num_workers_per_gpu, config.prefetch_factor),
        )

    run_metadata_path = run_dir / f"{run_id}_metadata.json"
    run_result_path = run_dir / f"{run_id}_result.json"
    write_json_atomic(
        run_metadata_path,
        {
            "schema_version": 1,
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git": collect_git_state(Path(__file__).resolve().parents[1]),
            "command": sys.argv,
            "config_sources": OmegaConf.to_container(config.config_sources, resolve=True),
            "resolved_config": str(resolved_config_path.resolve()),
            "dataset": {
                "path": config.dataset.get("dataset_path"),
                "task_name": config.dataset.get("task_name"),
                "base_samples": len(base_train_dataset),
                "selected_samples": len(train_dataset),
                "subset_indices": list(subset_indices) if subset_indices is not None else None,
                "shuffle": train_shuffle,
                "task_manifest": config.dataset.get("task_manifest"),
                "schema_path": config.dataset.get("schema_path"),
                "use_depth": bool(config.use_depth),
                "adapter_provenance": dataset_provenance,
            },
            "checkpoint": {
                "initialization_mode": config.get("initialization_mode"),
                "pretrained_checkpoint": config.get("pretrained_checkpoint"),
                "resume_checkpoint": resume_checkpoint,
                "events": str(checkpoint_events_path.resolve()),
                "allow_missing_frozen_resume_parameters": (
                    allow_missing_frozen_resume_parameters
                ),
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
            "topology": topology,
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
        initialization_report = os.path.join(
            config.exp_root, config.exp_name, "checkpoint_initialization.json"
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
    )
    torch.cuda.reset_peak_memory_stats()
    fit_error = None
    fit_started = time.monotonic()
    try:
        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=resume_checkpoint,
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
                checkpoint_callback.last_model_path if checkpoint_callback is not None else None
            ),
            "best_checkpoint": (
                checkpoint_callback.best_model_path if checkpoint_callback is not None else None
            ),
            "checkpoint_events": str(checkpoint_events_path.resolve()),
            "memory_at_result": collect_memory_snapshot(),
        }
        write_json_atomic(run_result_path, result_payload)
        print(f"Run result: {run_result_path} ({result_payload['result']})")


if __name__ == "__main__":
    main()
