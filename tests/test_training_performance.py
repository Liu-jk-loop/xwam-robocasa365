from __future__ import annotations

import ast
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]


class TrainingPerformanceContractTest(unittest.TestCase):
    @staticmethod
    def _load_text_cache_method():
        source = (REPO_ROOT / "runners/xwam_runner.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        runner = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "XWAMRunner"
        )
        method = next(
            node
            for node in runner.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_encode_text_embeddings"
        )

        class FakeTorch:
            no_grad = staticmethod(nullcontext)

            @staticmethod
            def stack(values, dim=0):
                return list(values)

        module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
        namespace = {"torch": FakeTorch}
        exec(compile(module, "<text-cache-method>", "exec"), namespace)
        return namespace["_encode_text_embeddings"]

    def test_frozen_t5_embeddings_are_cached_without_checkpoint_state(self) -> None:
        runner_path = REPO_ROOT / "runners/xwam_runner.py"
        source = runner_path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(runner_path))

        self.assertIn("self._text_embedding_cache = {}", source)
        self.assertIn("def _encode_text_embeddings(self, texts):", source)
        self.assertIn("computed = self.text_encoder(missing)", source)
        self.assertIn("embedding.detach()", source)
        self.assertIn(
            'self._encode_text_embeddings(batch["prompt"])',
            source,
        )
        self.assertIn('self._encode_text_embeddings([""] * B)', source)
        self.assertNotIn("register_buffer(\"_text_embedding_cache", source)

        model_config = (
            REPO_ROOT / "configs/model/wan22_5b_robocasa365_atomic.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("cache_frozen_text_embeddings: true", model_config)

    def test_frozen_t5_cache_computes_each_unique_prompt_once(self) -> None:
        method = self._load_text_cache_method()
        encoder_calls = []

        class FakeEmbedding(str):
            def detach(self):
                return self

        def encoder(prompts):
            encoder_calls.append(list(prompts))
            return [FakeEmbedding(f"embedding:{prompt}") for prompt in prompts]

        state = SimpleNamespace(
            config=SimpleNamespace(cache_frozen_text_embeddings=True),
            text_encoder=encoder,
            _text_embedding_cache={},
            _text_cache_requests=0,
            _text_cache_hits=0,
            _text_cache_computations=0,
        )
        first = method(state, ["task-a", "task-a", "task-b"])
        second = method(state, ["task-b", ""])

        self.assertEqual(
            encoder_calls,
            [["task-a", "task-b"], [""]],
        )
        self.assertEqual(
            first,
            ["embedding:task-a", "embedding:task-a", "embedding:task-b"],
        )
        self.assertEqual(second, ["embedding:task-b", "embedding:"])
        self.assertEqual(state._text_cache_computations, 3)
        self.assertEqual(set(state._text_embedding_cache), {"task-a", "task-b", ""})

    def test_gh200_formal_profile_uses_eight_workers_and_segment_timing(self) -> None:
        hardware = (
            REPO_ROOT / "configs/hardware/gh200x4_96gb_gbs128.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("num_workers_per_gpu: 8", hardware)
        self.assertIn("enable_segment_timing: true", hardware)
        self.assertIn("segment_timing_interval_steps: 20", hardware)

        runner = (REPO_ROOT / "runners/xwam_runner.py").read_text(
            encoding="utf-8"
        )
        for metric in (
            "data_wait_ms_per_microbatch",
            "t5_ms_per_microbatch",
            "vae_ms_per_microbatch",
            "dit_forward_ms_per_microbatch",
            "backward_ms_per_microbatch",
            "optimizer_ms_per_step",
            "batch_total_ms_per_microbatch",
            "t5_cache_hit_rate",
        ):
            self.assertIn(f'"timing/{metric}"', runner)
        self.assertIn("torch.cuda.Event(enable_timing=True)", runner)
        self.assertIn("torch.cuda.synchronize(self.device)", runner)


if __name__ == "__main__":
    unittest.main()
