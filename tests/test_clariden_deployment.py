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
        self.assertTrue(contract["simulator_contract"]["separate_container"])

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
            "pyarrow==16.1.0",
        ):
            self.assertIn(expected, requirements)
        for expected in (
            "torch==2.9.0",
            "torchvision==0.24.0",
            "torchaudio==2.9.0",
            "huggingface-hub==0.36.0",
            "safetensors==0.8.0",
        ):
            self.assertIn(expected, constraints)

    def test_edf_uses_persistent_source_and_iops_caches(self) -> None:
        template = (DEPLOY_ROOT / "xwam.toml.template").read_text(encoding="utf-8")
        self.assertIn("src/xwam-robocasa365", template)
        self.assertIn("/iopsstor/scratch/cscs/zjingchen/terry_nys/cache", template)
        self.assertIn('HF_HUB_OFFLINE = "1"', template)
        self.assertNotIn("MUJOCO_GL", template)

    def test_cluster_scripts_pass_shell_syntax(self) -> None:
        for script in ("prepare_xwam.sh", "build_xwam.sbatch", "validate_xwam.sbatch"):
            result = subprocess.run(
                ["bash", "-n", str(DEPLOY_ROOT / script)],
                cwd=REPO_ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
