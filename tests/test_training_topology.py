from __future__ import annotations

import unittest
from pathlib import Path

from project_tools.training_topology import (
    resolve_cpu_adam_options,
    resolve_deepspeed_options,
    resolve_optimizer_backend,
    resolve_training_topology,
)


class TrainingTopologyTest(unittest.TestCase):
    def test_optimizer_backend_is_cpu_adam_only_for_explicit_offload(self) -> None:
        self.assertEqual(resolve_optimizer_backend({}), "torch_adamw")
        self.assertEqual(
            resolve_optimizer_backend({"deepspeed_offload_optimizer": False}),
            "torch_adamw",
        )
        self.assertEqual(
            resolve_optimizer_backend({"deepspeed_offload_optimizer": True}),
            "deepspeed_cpu_adam",
        )

    def test_runner_wires_both_optimizer_backends(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        runner = (repo_root / "runners/xwam_runner.py").read_text()
        legacy_config = (repo_root / "configs/model/wan22_5b_sft.yaml").read_text()
        smoke_config = (
            repo_root / "configs/model/wan22_5b_robocasa365_atomic_m2.yaml"
        ).read_text()
        self.assertIn("DeepSpeedCPUAdam", runner)
        self.assertIn("torch.optim.AdamW", runner)
        self.assertIn("adamw_mode=True", runner)
        self.assertIn("resolve_cpu_adam_options(self.config)", runner)
        self.assertIn("deepspeed_offload_optimizer: false", legacy_config)
        self.assertIn("deepspeed_offload_optimizer: true", smoke_config)
        self.assertIn("deepspeed_fp32_optimizer_states: true", legacy_config)
        self.assertIn("deepspeed_fp32_optimizer_states: true", smoke_config)

    def test_cpu_adam_keeps_fp32_default_and_allows_explicit_debug_bf16(self) -> None:
        self.assertEqual(
            resolve_cpu_adam_options({}), {"fp32_optimizer_states": True}
        )
        self.assertEqual(
            resolve_cpu_adam_options(
                {"deepspeed_fp32_optimizer_states": False}
            ),
            {"fp32_optimizer_states": False},
        )
        with self.assertRaisesRegex(ValueError, "必须是布尔值"):
            resolve_cpu_adam_options(
                {"deepspeed_fp32_optimizer_states": "false"}
            )

    def test_legacy_deepspeed_defaults_are_unchanged(self) -> None:
        options = resolve_deepspeed_options({})
        self.assertEqual(options["stage"], 2)
        self.assertFalse(options["offload_optimizer"])
        self.assertTrue(options["overlap_comm"])
        self.assertEqual(options["allgather_bucket_size"], 500_000_000)
        self.assertEqual(options["reduce_bucket_size"], 500_000_000)
        self.assertFalse(options["exclude_frozen_parameters"])

    def test_m2_smoke_offloads_optimizer_and_reduces_buckets(self) -> None:
        options = resolve_deepspeed_options(
            {
                "deepspeed_stage": 2,
                "deepspeed_offload_optimizer": True,
                "deepspeed_offload_optimizer_device": "cpu",
                "deepspeed_overlap_comm": False,
                "deepspeed_bucket_size": 100_000_000,
                "deepspeed_exclude_frozen_parameters": True,
            }
        )
        self.assertTrue(options["offload_optimizer"])
        self.assertEqual(options["offload_optimizer_device"], "cpu")
        self.assertFalse(options["overlap_comm"])
        self.assertEqual(options["allgather_bucket_size"], 100_000_000)
        self.assertTrue(options["exclude_frozen_parameters"])

    def test_rejects_invalid_deepspeed_values(self) -> None:
        for config in (
            {"deepspeed_stage": 0},
            {"deepspeed_bucket_size": 0},
            {"deepspeed_offload_optimizer_device": "cuda"},
            {"deepspeed_exclude_frozen_parameters": "true"},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                resolve_deepspeed_options(config)

    def test_m2_config_and_entrypoint_wire_optimizer_offload(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        config = (repo_root / "configs/model/wan22_5b_robocasa365_atomic_m2.yaml").read_text()
        train_entrypoint = (repo_root / "scripts/train_sft.py").read_text()
        self.assertIn("deepspeed_offload_optimizer: true", config)
        self.assertIn("deepspeed_bucket_size: 100000000", config)
        self.assertIn("resolve_deepspeed_options(config)", train_entrypoint)
        self.assertIn("DeepSpeedStrategy(**deepspeed_options)", train_entrypoint)

    def test_m2_smoke_disables_optional_tensorboard_logger(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        config = (repo_root / "configs/model/wan22_5b_robocasa365_atomic_m2.yaml").read_text()
        train_entrypoint = (repo_root / "scripts/train_sft.py").read_text()
        self.assertIn("enable_tensorboard: false", config)
        self.assertIn('config.get("enable_tensorboard", True)', train_entrypoint)

    def test_single_gpu_smoke_does_not_use_all_visible_devices(self) -> None:
        topology = resolve_training_topology(visible_devices=4, requested_devices=1)
        self.assertEqual(topology["trainer_devices"], 1)
        self.assertEqual(topology["devices_per_node"], 1)
        self.assertEqual(topology["world_size"], 1)
        self.assertEqual(topology["num_nodes"], 1)

    def test_auto_uses_all_visible_devices_for_legacy_config(self) -> None:
        topology = resolve_training_topology(
            visible_devices=4, requested_devices="auto", world_size=4
        )
        self.assertEqual(topology["trainer_devices"], "auto")
        self.assertEqual(topology["devices_per_node"], 4)
        self.assertEqual(topology["num_nodes"], 1)

    def test_rejects_more_devices_than_visible(self) -> None:
        with self.assertRaisesRegex(ValueError, "GPU 配置无效"):
            resolve_training_topology(visible_devices=1, requested_devices=2)

    def test_rejects_non_divisible_world_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "WORLD_SIZE"):
            resolve_training_topology(
                visible_devices=4, requested_devices=2, world_size=3
            )


if __name__ == "__main__":
    unittest.main()
