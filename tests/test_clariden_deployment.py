from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_ROOT = REPO_ROOT / "deployment" / "clariden"


class ClaridenDeploymentTest(unittest.TestCase):
    def test_environment_contract_keeps_policy_and_simulator_separate(self) -> None:
        contract = json.loads(
            (REPO_ROOT / "configs/environment/xwam_clariden.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(contract["architecture"], "aarch64")
        self.assertEqual(contract["base_contract"]["python"], "3.10")
        self.assertEqual(contract["packages"]["torch"], "2.9.0+cu126")
        self.assertEqual(contract["packages"]["numpy"], "1.23.5")
        self.assertEqual(contract["packages"]["safetensors"], "0.8.0")
        self.assertEqual(contract["packages"]["nvtx"], "0.2.15")
        self.assertTrue(contract["simulator_contract"]["separate_container"])
        for gate in (
            "container_build",
            "gh200_cuda",
            "flash_attn_kernel",
            "checkpoint_discovery",
        ):
            self.assertEqual(contract["validation"][gate], "pass")
        self.assertEqual(contract["validation"]["dataset_smoke"], "pass")
        self.assertEqual(contract["validation"]["nvtx_runtime_overlay"], "pass")
        self.assertEqual(contract["validation"]["training"], "pass")
        self.assertEqual(
            contract["validation"]["multigpu_checkpoint_resume"],
            "pass",
        )
        self.assertEqual(
            contract["validation"]["m6_atomic_seen18_preflight"],
            "cluster-pending",
        )
        self.assertEqual(contract["cluster_evidence"]["training_smoke_result"], "pass")
        self.assertIsNone(contract["cluster_evidence"]["training_smoke_job_id"])
        self.assertIsNone(contract["cluster_evidence"]["training_source_commit"])
        self.assertEqual(
            contract["cluster_evidence"]["multigpu_initial_job_id"], 3046423
        )
        self.assertEqual(
            contract["cluster_evidence"]["multigpu_initial_result"], "pass"
        )
        self.assertEqual(
            contract["cluster_evidence"]["multigpu_resume_result"], "not-run"
        )
        self.assertEqual(
            contract["cluster_evidence"]["multigpu_resume_retry_failure"]["job_id"],
            3047286,
        )
        full_retry = contract["cluster_evidence"]["multigpu_full_retry"]
        self.assertEqual(full_retry["job_id"], 3047744)
        self.assertEqual(full_retry["initial_result"], "pass")
        self.assertEqual(full_retry["resumed_result"], "pass")
        self.assertEqual(full_retry["audit_result"], "fail")
        self.assertTrue(full_retry["checkpoint_optimizer_shards_verified"])
        self.assertEqual(
            full_retry["checkpoint_optimizer_shard_prefix"],
            "bf16_zero_pp_rank_",
        )
        multigpu_pass = contract["cluster_evidence"][
            "multigpu_checkpoint_resume_pass"
        ]
        self.assertEqual(multigpu_pass["job_id"], 3053264)
        self.assertEqual(
            multigpu_pass["source_commit"],
            "a2787ded5106f5178c21010d8378a0d2070e7f88",
        )
        self.assertTrue(multigpu_pass["audit_ok"])

    def test_containerfile_pins_arm64_critical_builds(self) -> None:
        containerfile = (DEPLOY_ROOT / "Containerfile").read_text(encoding="utf-8")
        for expected in (
            "FROM nvcr.io/nvidia/pytorch:24.10-py3",
            "https://download.pytorch.org/whl/cu126",
            '"torch==2.9.0"',
            '"torchvision==0.24.0"',
            '"torchaudio==2.9.0"',
            "6a3617cef035535193f390f86399dde139fa2a53",
            "060c9188beec3a8b62b33a3bfa6d5d2d44975fab",
            "submodule update --init --recursive",
            'TORCH_CUDA_ARCH_LIST="9.0"',
            "--no-build-isolation",
            "transformer-engine transformer-engine-cu12 torch-tensorrt",
            'assert safetensors.__version__ == "0.8.0"',
            'assert importlib.metadata.version("nvtx") == "0.2.15"',
            "assert callable(nvtx.get_domain)",
            'nvtx_domain.push_range(message="probe", category=None)',
        ):
            self.assertIn(expected, containerfile)
        self.assertNotIn("pip install decord", containerfile)
        self.assertNotIn("github.com/robocasa/robocasa", containerfile.lower())
        self.assertNotIn("pip install robocasa", containerfile.lower())
        self.assertLess(
            containerfile.index('checkout "${FLASH_ATTN_SHA}"'),
            containerfile.index("submodule update --init --recursive", containerfile.index("flash-attention")),
        )

    def test_requirements_preserve_validated_xwam_versions(self) -> None:
        requirements = (DEPLOY_ROOT / "requirements-clariden.txt").read_text(
            encoding="utf-8"
        )
        constraints = (DEPLOY_ROOT / "constraints-clariden.txt").read_text(
            encoding="utf-8"
        )
        for expected in (
            "numpy==1.23.5",
            "transformers==4.51.3",
            "diffusers==0.38.0",
            "huggingface-hub==0.36.0",
            "safetensors==0.8.0",
            "lightning==2.6.5",
            "deepspeed==0.19.4",
            "nvtx==0.2.15",
            "pyarrow==16.1.0",
        ):
            self.assertIn(expected, requirements)
        for expected in (
            "torch==2.9.0",
            "torchvision==0.24.0",
            "torchaudio==2.9.0",
            "huggingface-hub==0.36.0",
            "safetensors==0.8.0",
            "nvtx==0.2.15",
        ):
            self.assertIn(expected, constraints)

    def test_t5_default_device_is_not_resolved_during_import(self) -> None:
        source = (REPO_ROOT / "modules/t5.py").read_text(encoding="utf-8")
        self.assertNotIn("device=torch.cuda.current_device()", source)
        self.assertIn("device=None", source)
        self.assertIn("device = torch.cuda.current_device()", source)

    def test_edf_uses_persistent_source_and_iops_caches(self) -> None:
        template = (DEPLOY_ROOT / "xwam.toml.template").read_text(encoding="utf-8")
        self.assertIn("src/xwam-robocasa365", template)
        self.assertIn("/iopsstor/scratch/cscs/zjingchen/terry_nys/cache", template)
        self.assertIn("python/xwam-nvtx-0.2.15", template)
        self.assertIn('HF_HUB_OFFLINE = "1"', template)
        self.assertNotIn("MUJOCO_GL", template)

    def test_cluster_scripts_pass_shell_syntax(self) -> None:
        for script in (
            "prepare_xwam.sh",
            "build_xwam.sbatch",
            "validate_xwam.sbatch",
            "smoke_batch_xwam.sbatch",
            "smoke_train_xwam.sbatch",
            "prepare_runtime_overlay_xwam.sbatch",
            "smoke_train_resume_xwam.sbatch",
            "prepare_m6_data_xwam.sbatch",
        ):
            result = subprocess.run(
                ["bash", "-n", str(DEPLOY_ROOT / script)],
                cwd=REPO_ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

        build_script = (DEPLOY_ROOT / "build_xwam.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("unsquashfs -s", build_script)
        self.assertIn("ENROOT_STATUS", build_script)

    def test_clariden_train_smoke_is_one_gpu_one_step_without_checkpoint(self) -> None:
        hardware = (
            REPO_ROOT / "configs/hardware/gh200_96gb_debug.yaml"
        ).read_text(encoding="utf-8")
        experiment = (
            REPO_ROOT / "configs/experiment/robocasa365_clariden_single_step.yaml"
        ).read_text(encoding="utf-8")
        script = (DEPLOY_ROOT / "smoke_train_xwam.sbatch").read_text(
            encoding="utf-8"
        )
        for expected in (
            "devices: 1",
            "batch_size_per_gpu: 1",
            "deepspeed_offload_optimizer: true",
            "deepspeed_fp32_optimizer_states: true",
        ):
            self.assertIn(expected, hardware)
        for expected in (
            "num_training_steps: 1",
            "trainer_max_steps: 1",
            "train_subset_size: 1",
            "clean_action_ratio: 0.0",
            "enable_checkpointing: false",
        ):
            self.assertIn(expected, experiment)
        self.assertIn("torch.cuda.get_device_capability(0) == (9, 0)", script)
        self.assertIn("callable(nvtx.get_domain)", script)
        self.assertIn("domain.push_range(message='probe', category=None)", script)
        self.assertIn("module.is_relative_to(overlay)", script)
        self.assertIn("[PASS] X-WAM Clariden one-step train", script)

    def test_runtime_overlay_is_pinned_hashed_and_atomically_published(self) -> None:
        requirements = (DEPLOY_ROOT / "runtime-overlay-clariden.txt").read_text(
            encoding="utf-8"
        )
        script = (DEPLOY_ROOT / "prepare_runtime_overlay_xwam.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("nvtx==0.2.15", requirements)
        self.assertIn(
            "sha256:a4f50832fd90a1b480a9deef6e4cd48015b61869095b54dd1a7afe87b4138c6a",
            requirements,
        )
        for expected in (
            "--require-hashes",
            "--no-deps",
            "mktemp -d",
            'mv "$STAGING" "$OVERLAY"',
            "callable(nvtx.get_domain)",
            'domain.push_range(message="probe", category=None)',
            "xwam_runtime_overlay.txt",
        ):
            self.assertIn(expected, script)

    def test_clariden_four_gpu_resume_gate_is_debug_only_and_audited(self) -> None:
        hardware = (
            REPO_ROOT / "configs/hardware/gh200x4_96gb_resume_debug.yaml"
        ).read_text(encoding="utf-8")
        experiment = (
            REPO_ROOT
            / "configs/experiment/robocasa365_clariden_4gpu_resume_gate.yaml"
        ).read_text(encoding="utf-8")
        script = (DEPLOY_ROOT / "smoke_train_resume_xwam.sbatch").read_text(
            encoding="utf-8"
        )
        for expected in (
            "devices: 4",
            "batch_size_per_gpu: 1",
            "global_batch_size: 4",
            "deepspeed_offload_optimizer: true",
            "deepspeed_fp32_optimizer_states: true",
            "deepspeed_exclude_frozen_parameters: true",
            "audit_optimizer_state_dtype: true",
            "allow_distributed_generator_state: true",
        ):
            self.assertIn(expected, hardware)
        for expected in (
            "num_training_steps: 4",
            "trainer_max_steps: 2",
            "train_subset_size: 8",
            "save_interval: 2",
            "save_top_k: -1",
        ):
            self.assertIn(expected, experiment)
        for expected in (
            "#SBATCH --gpus-per-node=4",
            'SOURCE_JOB_ID="${XWAM_INITIAL_JOB_ID:-$SLURM_JOB_ID}"',
            'REUSE_INITIAL=true',
            "trainer_max_steps=4",
            "resume_checkpoint='$CHECKPOINT'",
            "unset SLURM_NTASKS",
            'env INITIAL_RESULT="$INITIAL_RESULT" CHECKPOINT_RECORD="$CHECKPOINT_RECORD"',
            "resolve_clariden_initial_checkpoint.py",
            'INITIAL_METADATA="$INITIAL_METADATA"',
            "--allow-orchestration-only-commit-delta",
            "audit_clariden_4gpu_resume.py",
            "[PASS] X-WAM Clariden 4xGH200 step 2 to 4 resume gate",
        ):
            self.assertIn(expected, script)
        self.assertNotIn('CHECKPOINT="$(python - "$INITIAL_RESULT"', script)
        self.assertNotIn("payload['result']", script)

    def test_clariden_m6_data_preflight_is_atomic_only_and_machine_audited(self) -> None:
        script = (DEPLOY_ROOT / "prepare_m6_data_xwam.sbatch").read_text(
            encoding="utf-8"
        )
        for expected in (
            "/datasets/robocasa/v1.0/pretrain/atomic",
            "build_robocasa365_m6_training_manifest.py",
            "compute_robocasa365_global_stats.py",
            "audit_robocasa365_m6_preflight.py",
            "--global-batch-size 128",
            "--epochs 5",
            'test -z "$(git status --porcelain)"',
            "grep -q '\"ok\": true'",
            "[PASS] X-WAM Clariden M6 Atomic-Seen 18 manifest, global stats and schedule",
        ):
            self.assertIn(expected, script)
        self.assertNotIn("/composite", script)


if __name__ == "__main__":
    unittest.main()
