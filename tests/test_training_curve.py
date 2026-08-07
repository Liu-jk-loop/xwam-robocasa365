from __future__ import annotations

import unittest

from project_tools.training_curve import build_overfit_curve_report


def _metric_log(*, steps: int = 20, supervision: float = 1.0) -> str:
    lines = []
    for step in range(steps):
        progress = step / max(steps - 1, 1)
        video = 1.0 - 0.5 * progress
        action = 1.2 - 0.6 * progress
        proprio = 1.4 - 0.7 * progress
        total = video + action + proprio
        lines.append(
            "[METRICS] "
            f"Step: {step:07d} - "
            f"train/video_loss: {video:.6f}, "
            f"train/action_loss: {action:.6f}, "
            f"train/proprio_loss: {proprio:.6f}, "
            f"train/action_proprio_supervision_ratio: {supervision:.6f}, "
            "train/depth_loss: 0.000, "
            f"train/loss: {total:.6f}"
        )
    lines.extend(
        [
            f"`Trainer.fit` stopped: `max_steps={steps}` reached.",
            "Run result: /tmp/result.json (pass)",
        ]
    )
    return "\n".join(lines)


class TrainingCurveTest(unittest.TestCase):
    def test_complete_finite_decreasing_curve_passes(self) -> None:
        report = build_overfit_curve_report(
            _metric_log(),
            expected_steps=20,
            window_size=5,
            minimum_relative_drop=0.10,
        )
        self.assertTrue(report["ok"])
        self.assertTrue(report["checks"]["minimum_loss_drop"])
        self.assertTrue(report["checks"]["full_action_proprio_supervision"])

    def test_zero_supervision_fails(self) -> None:
        report = build_overfit_curve_report(
            _metric_log(supervision=0.0),
            expected_steps=20,
            window_size=5,
            minimum_relative_drop=0.10,
        )
        self.assertFalse(report["ok"])
        self.assertTrue(report["checks"]["complete_step_sequence"])
        self.assertFalse(report["checks"]["full_action_proprio_supervision"])

    def test_incomplete_steps_fail(self) -> None:
        lines = _metric_log().splitlines()
        report = build_overfit_curve_report(
            "\n".join(lines[1:]),
            expected_steps=20,
            window_size=5,
            minimum_relative_drop=0.10,
        )
        self.assertFalse(report["ok"])
        self.assertFalse(report["checks"]["complete_step_sequence"])


if __name__ == "__main__":
    unittest.main()
