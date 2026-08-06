import os
import gc
import sys
import logging
from pprint import pprint

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
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
from project_tools.training_topology import resolve_training_topology
from runners.xwam_runner import XWAMRunner
from utils.console_logger import ConsoleLogger
from utils.xwam_checkpoint_loader import initialize_xwam_runner


def _load_config():
    overrides = OmegaConf.from_cli()
    model_config_path = overrides.pop("model_config", "configs/model/wan22_5b_sft.yaml")
    model_config = OmegaConf.load(model_config_path)

    dataset_override = overrides.get("dataset")
    if isinstance(dataset_override, str):
        dataset_name = dataset_override
        del overrides["dataset"]
    else:
        dataset_name = str(model_config.dataset)
    data_config_path = overrides.pop("data_config", f"configs/data/{dataset_name}.yaml")
    model_config["dataset"] = OmegaConf.load(data_config_path)
    return OmegaConf.merge(model_config, overrides)


def _loader_worker_options(num_workers: int, prefetch_factor: int) -> dict:
    if num_workers <= 0:
        return {}
    return {
        "prefetch_factor": int(prefetch_factor),
        "multiprocessing_context": "forkserver",
    }


def _resolve_trainer_topology(config):
    return resolve_training_topology(
        visible_devices=torch.cuda.device_count(),
        requested_devices=config.get("devices", "auto"),
        world_size=os.environ.get("WORLD_SIZE"),
    )


def main():
    config = _load_config()
    topology = _resolve_trainer_topology(config)

    pprint(OmegaConf.to_container(config))

    os.makedirs(os.path.join(config.exp_root, config.exp_name), exist_ok=True)
    OmegaConf.save(
        config,
        os.path.join(config.exp_root, config.exp_name, "config.yaml"),
        resolve=True,
    )
    L.seed_everything(config.seed, workers=True)

    callbacks = [ModelSummary(max_depth=2), LearningRateMonitor(logging_interval="step")]
    if bool(config.get("enable_checkpointing", True)):
        callbacks.insert(
            1,
            ModelCheckpoint(
                dirpath=os.path.join(config.exp_root, config.exp_name, "checkpoints"),
                save_top_k=-1,
                save_last=True,
                every_n_train_steps=config.save_interval,
                enable_version_counter=False,
            ),
        )

    tb_path = os.getenv("TENSORBOARD_LOG_PATH", None)
    loggers = [
        TensorBoardLogger(
            save_dir=tb_path if tb_path else os.path.join(config.exp_root, config.exp_name, "tb_logs"),
            name=config.exp_name,
        ),
        ConsoleLogger(max_steps=config.num_training_steps),
    ]
    logging.getLogger("lightning.pytorch").setLevel(logging.INFO)

    train_dataset = build_dataset(config.dataset, use_depth=config.use_depth)
    config.action_num = train_dataset.action_num
    if int(config.action_dim) != int(train_dataset.action_dim):
        raise ValueError(
            f"模型 action_dim={config.action_dim} 与数据 action_dim={train_dataset.action_dim} 不一致"
        )
    if int(config.proprio_dim) != int(train_dataset.proprio_dim):
        raise ValueError(
            f"模型 proprio_dim={config.proprio_dim} 与数据 proprio_dim={train_dataset.proprio_dim} 不一致"
        )

    val_dataset = build_dataset(config.dataset, use_depth=config.use_depth, augment=False)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size_per_gpu,
        num_workers=config.num_workers_per_gpu,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        **_loader_worker_options(config.num_workers_per_gpu, config.prefetch_factor),
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=config.num_workers_per_gpu,
        shuffle=True,
        pin_memory=True,
        **_loader_worker_options(config.num_workers_per_gpu, config.prefetch_factor),
    )

    model = XWAMRunner(config, run_depth=bool(config.use_depth))
    initialization_report = config.get("checkpoint_initialization_report")
    if initialization_report is None:
        initialization_report = os.path.join(
            config.exp_root, config.exp_name, "checkpoint_initialization.json"
        )
    report = initialize_xwam_runner(
        model,
        mode=str(config.get("initialization_mode", "legacy_strict")),
        checkpoint=config.get("pretrained_checkpoint"),
        schema_path=config.get("checkpoint_schema_path"),
        report_path=initialization_report,
    )
    print(
        "Checkpoint initialization: "
        f"mode={report['mode']}, result={report['result']}, report={initialization_report}"
    )

    print(
        "Runner v2, "
        f"num_nodes: {topology['num_nodes']}, world_size: {topology['world_size']}, "
        f"trainer_devices: {topology['trainer_devices']}, "
        f"visible_devices: {topology['visible_devices']}"
    )

    trainer = L.Trainer(
        accelerator="auto",
        devices=topology["trainer_devices"],
        strategy=DeepSpeedStrategy(
            allgather_bucket_size=5e8,
            reduce_bucket_size=5e8,
        ),
        precision="bf16-mixed",
        num_nodes=topology["num_nodes"],
        max_steps=config.num_training_steps,
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
    try:
        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    finally:
        gib = 1024**3
        print(
            "CUDA peak memory: "
            f"allocated={torch.cuda.max_memory_allocated() / gib:.3f} GiB, "
            f"reserved={torch.cuda.max_memory_reserved() / gib:.3f} GiB"
        )


if __name__ == "__main__":
    main()
