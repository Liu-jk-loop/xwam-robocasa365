from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from project_tools.xwam_checkpoint_contract import (
    plan_xwam_checkpoint_adaptation,
    remap_action_boundary,
    resolve_xwam_checkpoint,
)


def boundary_shapes(action_dim: int, hidden_dim: int = 4) -> dict[str, tuple[int, ...]]:
    return {
        "model.action_encoder.0.weight": (hidden_dim, action_dim),
        "model.action_encoder.0.bias": (hidden_dim,),
        "model.action_decoder.2.weight": (action_dim, hidden_dim),
        "model.action_decoder.2.bias": (action_dim,),
        "model.proprio_encoder.0.weight": (hidden_dim, 16),
        "model.proprio_encoder.0.bias": (hidden_dim,),
        "model.proprio_decoder.2.weight": (16, hidden_dim),
        "model.proprio_decoder.2.bias": (16,),
        "model.blocks.0.weight": (hidden_dim, hidden_dim),
    }


class XWAMCheckpointContractTest(unittest.TestCase):
    def test_plan_classifies_every_declared_difference(self) -> None:
        source = boundary_shapes(14)
        source["model.extra_blocks.0.0.weight"] = (4, 4)
        source["model.extra_heads.0.head.weight"] = (4, 4)
        target = boundary_shapes(12)

        plan = plan_xwam_checkpoint_adaptation(source, target)

        self.assertTrue(plan["ok"], plan["errors"])
        self.assertEqual(len(plan["action_remap"]), 3)
        self.assertEqual(len(plan["reinitialize"]), 4)
        self.assertEqual(len(plan["initialized"]), 7)
        self.assertEqual(len(plan["discard_source"]), 2)
        self.assertIn("model.action_encoder.0.bias", plan["exact_load"])
        self.assertIn("model.blocks.0.weight", plan["exact_load"])

    def test_undeclared_shape_mismatch_is_blocking(self) -> None:
        source = boundary_shapes(14)
        target = boundary_shapes(12)
        target["model.blocks.0.weight"] = (5, 4)

        plan = plan_xwam_checkpoint_adaptation(source, target)

        self.assertFalse(plan["ok"])
        self.assertTrue(any("未声明的 shape mismatch" in item for item in plan["errors"]))

    def test_action_mapping_preserves_new_base_and_control_dimensions(self) -> None:
        source_encoder = [list(range(row * 14, (row + 1) * 14)) for row in range(4)]
        target_encoder = [[-9.0] * 12 for _ in range(4)]
        mapped_encoder = remap_action_boundary(
            "model.action_encoder.0.weight", source_encoder, target_encoder
        )
        self.assertEqual([row[:5] for row in mapped_encoder], [row[:5] for row in target_encoder])
        self.assertEqual([row[5:12] for row in mapped_encoder], [row[:7] for row in source_encoder])

        source_decoder = [list(range(row * 4, (row + 1) * 4)) for row in range(14)]
        target_decoder = [[-7.0] * 4 for _ in range(12)]
        mapped_decoder = remap_action_boundary(
            "model.action_decoder.2.weight", source_decoder, target_decoder
        )
        self.assertEqual(mapped_decoder[:5], target_decoder[:5])
        self.assertEqual(mapped_decoder[5:12], source_decoder[:7])

        source_bias = list(range(14))
        target_bias = [-3.0] * 12
        mapped_bias = remap_action_boundary(
            "model.action_decoder.2.bias", source_bias, target_bias
        )
        self.assertEqual(mapped_bias[:5], target_bias[:5])
        self.assertEqual(mapped_bias[5:12], source_bias[:7])

    def test_resolve_accepts_public_checkpoint_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = (
                root
                / "pretrained"
                / "checkpoints"
                / "last.ckpt"
                / "checkpoint"
                / "mp_rank_00_model_states.pt"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            self.assertEqual(resolve_xwam_checkpoint(root), checkpoint.resolve())


if __name__ == "__main__":
    unittest.main()
