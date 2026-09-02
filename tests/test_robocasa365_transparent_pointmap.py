from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from project_tools.robocasa365_transparent_pointmap import (
    alpha_change_mask,
    find_target_geom_ids,
    resolve_close_blender_lid_geom_ids,
    target_alpha_change_count,
    temporarily_force_visible_alpha_opaque,
    transparency_conclusion,
    transparent_depth_delta,
)


class _FakeModel:
    def __init__(self) -> None:
        self.ngeom = 3
        self.geom_rgba = np.asarray(
            [
                [1.0, 1.0, 1.0, 1.0],
                [0.8, 0.8, 0.8, 0.25],
                [0.3, 0.3, 0.3, 0.0],
            ],
            dtype=np.float64,
        )
        self.mat_rgba = np.asarray([[1.0, 1.0, 1.0, 0.5]], dtype=np.float64)
        self.geom_bodyid = np.asarray([0, 1, 2], dtype=np.int32)
        self.geom_matid = np.asarray([-1, 0, -1], dtype=np.int32)
        self._geom_names = ["counter", "blender_lid_visual", "pot_lid"]
        self._body_names = ["kitchen", "blender_main", "cookware"]
        self._material_names = ["transparent_lid"]

    def geom_id2name(self, object_id: int) -> str:
        return self._geom_names[object_id]

    def body_id2name(self, object_id: int) -> str:
        return self._body_names[object_id]

    def material_id2name(self, object_id: int) -> str:
        return self._material_names[object_id]

    def geom_name2id(self, name: str) -> int:
        return self._geom_names.index(name)


class RoboCasa365TransparentPointMapTests(unittest.TestCase):
    def test_alpha_mask_changes_only_visible_translucent_rows(self):
        rgba = np.asarray(
            [
                [1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 0.2],
                [1.0, 1.0, 1.0, 1.0],
            ]
        )
        np.testing.assert_array_equal(alpha_change_mask(rgba), [False, True, False])

    def test_forced_opaque_context_restores_geom_and_material_alpha(self):
        model = _FakeModel()
        original_geom = model.geom_rgba.copy()
        original_mat = model.mat_rgba.copy()
        with temporarily_force_visible_alpha_opaque(model) as report:
            self.assertEqual(report["changed_count"], 2)
            self.assertEqual(model.geom_rgba[1, 3], 1.0)
            self.assertEqual(model.mat_rgba[0, 3], 1.0)
            self.assertEqual(model.geom_rgba[2, 3], 0.0)
        np.testing.assert_array_equal(model.geom_rgba, original_geom)
        np.testing.assert_array_equal(model.mat_rgba, original_mat)

    def test_target_geom_matching_uses_geom_body_and_material_names(self):
        ids, rows = find_target_geom_ids(
            _FakeModel(), (r"blender.*lid", r"lid.*blender")
        )
        self.assertEqual(ids, [1])
        self.assertEqual(rows[0]["geom_name"], "blender_lid_visual")
        self.assertEqual(rows[0]["body_name"], "blender_main")

    def test_explicit_blender_lid_entity_is_preferred_over_name_fallback(self):
        model = _FakeModel()
        lid = SimpleNamespace(visual_geoms=["blender_lid_visual"])
        env = SimpleNamespace(
            sim=SimpleNamespace(model=model),
            blender=SimpleNamespace(blender_lid=lid),
        )
        ids, rows, source = resolve_close_blender_lid_geom_ids(env)
        self.assertEqual(ids, [1])
        self.assertEqual(rows[0]["geom_name"], "blender_lid_visual")
        self.assertEqual(source, "base_env.blender.blender_lid")

    def test_target_alpha_count_excludes_unrelated_translucent_objects(self):
        model = _FakeModel()
        with temporarily_force_visible_alpha_opaque(model) as opacity:
            count = target_alpha_change_count(model, [1], opacity)
        self.assertEqual(count, 2)

    def test_target_only_depth_delta_detects_forced_opaque_surface(self):
        original = np.asarray([[1.0, 1.0], [1.0, 3.0]], dtype=np.float32)
        opaque = np.asarray([[0.8, 1.0], [1.2, 0.7]], dtype=np.float32)
        segmentation = np.zeros((2, 2, 2), dtype=np.int32)
        segmentation[..., 1] = [[5, 9], [5, 5]]
        arrays, report = transparent_depth_delta(
            original,
            opaque,
            segmentation,
            [5],
            minimum_change_m=0.001,
        )
        self.assertEqual(report["target_visible_pixels"], 3)
        self.assertEqual(report["closer_pixels"], 1)
        self.assertEqual(report["farther_pixels"], 1)
        self.assertEqual(report["newly_valid_pixels"], 1)
        self.assertEqual(report["changed_pixels"], 3)
        self.assertEqual(int(arrays["changed_mask"].sum()), 3)
        self.assertFalse(arrays["target_mask"][0, 1])

    def test_conclusion_requires_forced_opaque_when_target_depth_changes(self):
        conclusion = transparency_conclusion(
            matched_target_geoms=2,
            alpha_changed_count=1,
            target_visible_pixels=100,
            changed_target_pixels=12,
        )
        self.assertTrue(conclusion["conclusive"])
        self.assertEqual(conclusion["status"], "forced_opaque_required")
        self.assertEqual(
            conclusion["formal_cache_policy"],
            "use_forced_opaque_depth_for_pointmap_keep_original_rgb",
        )

    def test_conclusion_accepts_original_depth_only_after_visible_control(self):
        conclusion = transparency_conclusion(
            matched_target_geoms=2,
            alpha_changed_count=1,
            target_visible_pixels=100,
            changed_target_pixels=0,
        )
        self.assertTrue(conclusion["conclusive"])
        self.assertEqual(conclusion["status"], "original_depth_matches_forced_opaque")

    def test_conclusion_is_inconclusive_when_target_is_not_visible(self):
        conclusion = transparency_conclusion(
            matched_target_geoms=2,
            alpha_changed_count=1,
            target_visible_pixels=0,
            changed_target_pixels=0,
        )
        self.assertFalse(conclusion["conclusive"])
        self.assertEqual(
            conclusion["status"], "inconclusive_target_not_visible_in_samples"
        )


if __name__ == "__main__":
    unittest.main()
