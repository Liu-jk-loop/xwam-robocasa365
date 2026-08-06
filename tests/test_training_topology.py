from __future__ import annotations

import unittest

from project_tools.training_topology import resolve_training_topology


class TrainingTopologyTest(unittest.TestCase):
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
