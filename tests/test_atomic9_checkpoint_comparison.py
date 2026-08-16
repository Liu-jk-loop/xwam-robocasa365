from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from evaluation.launch_robocasa365_m6_policy_pool import (
    _parse_group_checkpoints,
)
from evaluation.robocasa365_m6_topology import (
    load_m6_evaluation_topology,
)
from scripts.aggregate_robocasa365_checkpoint_comparison import (
    aggregate_comparison,
)
from scripts.resolve_robocasa365_eval_checkpoints import (
    resolve_exact_checkpoints,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TOPOLOGY = (
    REPO_ROOT
    / "configs/evaluation/robocasa365_atomic9_step5500_vs_step7000_8server_16client.json"
)
TOPOLOGY_65V75 = (
    REPO_ROOT
    / "configs/evaluation/robocasa365_atomic9_step6500_vs_step7500_8server_16client.json"
)
ATOMIC9 = {
    "OpenStandMixerHead",
    "PickPlaceSinkToCounter",
    "TurnOnElectricKettle",
    "CloseFridge",
    "TurnOnMicrowave",
    "OpenDrawer",
    "CoffeeSetupMug",
    "PickPlaceDrawerToCounter",
    "CloseBlenderLid",
}


def _make_checkpoint(root: Path, step: int, *, ranks: int = 8) -> Path:
    checkpoint = root / f"epoch=0-step={step}.ckpt" / "checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "mp_rank_00_model_states.pt").write_bytes(b"model")
    for rank in range(ranks):
        (
            checkpoint
            / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
        ).write_bytes(b"optimizer")
    return checkpoint.parent


class Atomic9CheckpointComparisonTest(unittest.TestCase):
    def test_topology_duplicates_the_same_atomic9_set_across_two_groups(self) -> None:
        topology = load_m6_evaluation_topology(TOPOLOGY, REPO_ROOT)
        self.assertEqual(topology["schema_version"], 2)
        self.assertEqual(
            {
                name: group["checkpoint_step"]
                for name, group in topology["comparison_groups"].items()
            },
            {"step5500": 5500, "step7000": 7000},
        )
        self.assertEqual(
            [topology["servers"][index]["gpu"] for index in range(8)],
            [0, 0, 1, 1, 2, 2, 3, 3],
        )
        for group_name in ("step5500", "step7000"):
            group_clients = [
                client
                for client in topology["clients"].values()
                if client["comparison_group"] == group_name
            ]
            assigned = [
                task["name"] for client in group_clients for task in client["tasks"]
            ]
            self.assertEqual(set(assigned), ATOMIC9)
            self.assertEqual(len(assigned), 9)
            self.assertEqual(
                Counter(len(client["tasks"]) for client in group_clients),
                Counter({1: 7, 2: 1}),
            )
            paired = next(client for client in group_clients if len(client["tasks"]) == 2)
            self.assertEqual(
                [task["name"] for task in paired["tasks"]],
                ["PickPlaceSinkToCounter", "OpenDrawer"],
            )
        all_assigned = [
            task["name"]
            for client in topology["clients"].values()
            for task in client["tasks"]
        ]
        self.assertNotIn("PickPlaceCounterToStove", all_assigned)

    def test_exact_checkpoint_resolver_requires_all_eight_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint5500 = _make_checkpoint(root, 5500)
            checkpoint7000 = _make_checkpoint(root, 7000)
            _make_checkpoint(root, 6500, ranks=7)
            report = resolve_exact_checkpoints(
                [root],
                {"step5500": 5500, "step7000": 7000},
                expected_world_size=8,
            )
            self.assertTrue(report["ok"], report["errors"])
            self.assertEqual(
                report["groups"]["step5500"]["path"],
                str(checkpoint5500.resolve()),
            )
            self.assertEqual(
                report["groups"]["step7000"]["path"],
                str(checkpoint7000.resolve()),
            )
            missing = resolve_exact_checkpoints(
                [root],
                {"step6500": 6500},
                expected_world_size=8,
            )
            self.assertFalse(missing["ok"])
            self.assertIn("完整step 6500", missing["errors"][0])

    def test_group_checkpoint_parser_rejects_duplicates(self) -> None:
        self.assertEqual(
            _parse_group_checkpoints(
                ["step5500=/tmp/5500.ckpt", "step7000=/tmp/7000.ckpt"]
            ),
            {
                "step5500": "/tmp/5500.ckpt",
                "step7000": "/tmp/7000.ckpt",
            },
        )
        with self.assertRaisesRegex(ValueError, "重复"):
            _parse_group_checkpoints(
                ["step5500=/tmp/a.ckpt", "step5500=/tmp/b.ckpt"]
            )

    def test_aggregator_keeps_results_separate_and_computes_delta(self) -> None:
        topology = load_m6_evaluation_topology(TOPOLOGY, REPO_ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for group_name in ("step5500", "step7000"):
                success = group_name == "step7000"
                for client in topology["clients"].values():
                    if client["comparison_group"] != group_name:
                        continue
                    for task in client["tasks"]:
                        result_path = root / group_name / task["name"] / "result.json"
                        result_path.parent.mkdir(parents=True)
                        result_path.write_text(
                            json.dumps(
                                {
                                    "result": "pass",
                                    "task": task["name"],
                                    "client_id": client["client_id"],
                                    "server_id": client["server_id"],
                                    "comparison_group": group_name,
                                    "split": "target",
                                    "model_seed": 42,
                                    "seed_start": 42,
                                    "episodes_expected": 1,
                                    "episodes_completed": 1,
                                    "n_success": int(success),
                                    "success_rate": float(success),
                                    "mean_inference_time_s": 1.0,
                                    "max_steps": 1000,
                                    "replan_steps": 20,
                                    "action_denoise_steps": 10,
                                    "video_fps": 20,
                                    "episodes": [
                                        {"seed": 42, "success": success}
                                    ],
                                }
                            ),
                            encoding="utf-8",
                        )
            report = aggregate_comparison(
                topology_path=TOPOLOGY,
                output_root=root,
                episodes_per_task=1,
                group_checkpoints={
                    "step5500": "/tmp/step5500.ckpt",
                    "step7000": "/tmp/step7000.ckpt",
                },
                eval_id="unit-55v70",
            )
            self.assertTrue(report["ok"], report["errors"])
            self.assertEqual(report["groups"]["step5500"]["overall"]["tasks"], 9)
            self.assertEqual(report["groups"]["step7000"]["overall"]["tasks"], 9)
            self.assertEqual(
                report["comparison"]["macro_success_rate_delta"], 1.0
            )
            self.assertEqual(len(report["comparison"]["tasks"]), 9)

    def test_sbatch_uses_only_the_evaluation_clone_and_frozen_steps(self) -> None:
        script = (
            REPO_ROOT
            / "deployment/clariden/eval_atomic9_step5500_vs_step7000_xwam.sbatch"
        ).read_text(encoding="utf-8")
        for expected in (
            'REPO="$DEPLOY_STORE/src/xwam-robocasa365-eval"',
            "eval/atomic9-checkpoint-ab",
            "--group-step step5500=5500",
            "--group-step step7000=7000",
            'checkpoint_step5500=$CHECKPOINT_5500 gpus=0,1',
            'checkpoint_step7000=$CHECKPOINT_7000 gpus=2,3',
        ):
            self.assertIn(expected, script)
        self.assertNotIn('REPO="$DEPLOY_STORE/src/xwam-robocasa365"', script)

    def test_step6500_vs_step7500_reuses_the_matched_contract(self) -> None:
        topology = load_m6_evaluation_topology(TOPOLOGY_65V75, REPO_ROOT)
        self.assertEqual(
            {
                name: group["checkpoint_step"]
                for name, group in topology["comparison_groups"].items()
            },
            {"step6500": 6500, "step7500": 7500},
        )
        for group_name in ("step6500", "step7500"):
            assigned = [
                task["name"]
                for client in topology["clients"].values()
                if client["comparison_group"] == group_name
                for task in client["tasks"]
            ]
            self.assertEqual(set(assigned), ATOMIC9)
            self.assertEqual(len(assigned), 9)
        self.assertEqual(topology["model_seed"], 42)
        self.assertEqual(topology["seed_start"], 42)
        self.assertEqual(topology["episodes_per_task"], 50)
        self.assertEqual(topology["replan_steps"], 20)

        script = (
            REPO_ROOT
            / "deployment/clariden/eval_atomic9_step6500_vs_step7500_xwam.sbatch"
        ).read_text(encoding="utf-8")
        for expected in (
            'REPO="$DEPLOY_STORE/src/xwam-robocasa365-eval"',
            "eval/atomic9-checkpoint-ab",
            "--group-step step6500=6500",
            "--group-step step7500=7500",
            'checkpoint_step6500=$CHECKPOINT_6500 gpus=0,1',
            'checkpoint_step7500=$CHECKPOINT_7500 gpus=2,3',
        ):
            self.assertIn(expected, script)
        self.assertNotIn('REPO="$DEPLOY_STORE/src/xwam-robocasa365"', script)


if __name__ == "__main__":
    unittest.main()
