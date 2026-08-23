from __future__ import annotations

import json
import os
import subprocess
import tempfile
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
        self.assertEqual(contract["packages"]["wandb"], "0.23.1")
        self.assertEqual(contract["packages"]["sentry_sdk"], "2.58.0")
        self.assertEqual(contract["packages"]["gitpython"], "3.1.58")
        self.assertEqual(contract["packages"]["gitdb"], "4.0.12")
        self.assertEqual(contract["packages"]["smmap"], "5.0.3")
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
            "pass",
        )
        self.assertEqual(
            contract["validation"]["m6_gh200_formal_profile_gate"],
            "cluster-pending",
        )
        self.assertEqual(
            contract["validation"]["wandb_runtime_overlay"],
            "pass",
        )
        self.assertEqual(
            contract["validation"]["m6_atomic_seen18_evaluation"],
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
        multigpu_pass = contract["cluster_evidence"]["multigpu_checkpoint_resume_pass"]
        self.assertEqual(multigpu_pass["job_id"], 3053264)
        self.assertEqual(
            multigpu_pass["source_commit"],
            "a2787ded5106f5178c21010d8378a0d2070e7f88",
        )
        self.assertTrue(multigpu_pass["audit_ok"])
        m6_preflight = contract["cluster_evidence"]["m6_atomic_seen18_preflight_pass"]
        self.assertEqual(m6_preflight["job_id"], 3053322)
        self.assertEqual(m6_preflight["task_count"], 18)
        self.assertEqual(m6_preflight["total_valid_clips"], 419706)
        self.assertEqual(m6_preflight["num_training_steps"], 16390)
        self.assertEqual(
            contract["cluster_evidence"]["wandb_runtime_overlay_result"],
            "pass",
        )
        self.assertIsNone(contract["cluster_evidence"]["wandb_runtime_overlay_job_id"])
        self.assertEqual(
            contract["cluster_evidence"]["wandb_runtime_overlay_validation_job_id"],
            3054130,
        )
        formal_failure = contract["cluster_evidence"]["wandb_formal_run_failure"]
        self.assertEqual(formal_failure["job_id"], 3054130)
        self.assertEqual(formal_failure["global_step"], 0)
        self.assertEqual(formal_failure["wandb_api_key_auth"], "pass")
        self.assertFalse(formal_failure["wandb_entity_explicit"])
        entity_failure = contract["cluster_evidence"]["wandb_entity_preflight_failure"]
        self.assertEqual(entity_failure["job_id"], 3054165)
        self.assertEqual(entity_failure["phase"], "outer_preflight")
        self.assertEqual(entity_failure["global_step"], 0)

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
            'assert importlib.metadata.version("wandb") == "0.23.1"',
            'assert importlib.metadata.version("sentry-sdk") == "2.58.0"',
            'assert importlib.metadata.version("GitPython") == "3.1.58"',
            'assert importlib.metadata.version("gitdb") == "4.0.12"',
            'assert importlib.metadata.version("smmap") == "5.0.3"',
            "assert callable(nvtx.get_domain)",
            'nvtx_domain.push_range(message="probe", category=None)',
        ):
            self.assertIn(expected, containerfile)
        self.assertNotIn("pip install decord", containerfile)
        self.assertNotIn("github.com/robocasa/robocasa", containerfile.lower())
        self.assertNotIn("pip install robocasa", containerfile.lower())
        self.assertLess(
            containerfile.index('checkout "${FLASH_ATTN_SHA}"'),
            containerfile.index(
                "submodule update --init --recursive",
                containerfile.index("flash-attention"),
            ),
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
            "wandb==0.23.1",
            "sentry-sdk==2.58.0",
            "GitPython==3.1.58",
            "gitdb==4.0.12",
            "smmap==5.0.3",
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
            "wandb==0.23.1",
            "sentry-sdk==2.58.0",
            "GitPython==3.1.58",
            "gitdb==4.0.12",
            "smmap==5.0.3",
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
            "error_trap.sh",
            "archive_debug_logs_xwam.sh",
            "prepare_xwam.sh",
            "build_xwam.sbatch",
            "validate_xwam.sbatch",
            "smoke_batch_xwam.sbatch",
            "smoke_train_xwam.sbatch",
            "prepare_runtime_overlay_xwam.sbatch",
            "prepare_wandb_overlay_xwam.sbatch",
            "smoke_train_resume_xwam.sbatch",
            "prepare_m6_data_xwam.sbatch",
            "smoke_m6_gate_xwam.sbatch",
            "train_m6_formal_xwam.sbatch",
            "train_m6_formal_xwam_8gpu.sbatch",
            "train_close_fridge_ab_xwam.sbatch",
            "eval_close_fridge_ab_xwam.sbatch",
            "eval_fastwam_atomic9_xwam_b_shared4.sbatch",
            "eval_m6_atomic18_xwam.sbatch",
            "eval_atomic9_rgbd_step12000_xwam.sbatch",
            "eval_atomic9_rgbd_step8500_xwam.sbatch",
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

        build_script = (DEPLOY_ROOT / "build_xwam.sbatch").read_text(encoding="utf-8")
        self.assertIn("unsquashfs -s", build_script)
        self.assertIn("ENROOT_STATUS", build_script)

        eval_script = (DEPLOY_ROOT / "eval_m6_atomic18_xwam.sbatch").read_text(
            encoding="utf-8"
        )
        for expected in (
            "#SBATCH --export=ALL",
            "#SBATCH --error=",
            'EVAL_NAMESPACE="${XWAM_EVAL_NAMESPACE:-atomic18}"',
            'EVAL_ROOT="$DEPLOY_IOPS/x-wam-eval/$EVAL_NAMESPACE/$EVAL_ID"',
            'ROBOCASA_ROOT="$DEPLOY_STORE/src/robocasa"',
            'ROBOSUITE_ROOT="$DEPLOY_STORE/src/robosuite"',
            "fixtures/sinks/Sink025/model.xml",
            "probe_robocasa365_eval_runtime.py",
            'PYTHONPATH="$ROBOCASA_ROOT:$ROBOSUITE_ROOT:$REPO',
            "evaluation_contract.txt",
            'LOG_ROOT="$EVAL_ROOT/logs"',
            'RESULT_ROOT="$EVAL_ROOT/results"',
            'SUMMARY_STEM="${XWAM_EVAL_SUMMARY_STEM:-atomic18}"',
            'SUMMARY_CSV="$EVAL_ROOT/summary_${SUMMARY_STEM}.csv"',
            '"$LOG_ROOT/server_launcher.log"',
            '--log-root "$LOG_ROOT"',
            'EVAL_ID="$EVAL_ID"',
            'CHECKPOINT="$CHECKPOINT"',
            "--eval-id \"$EVAL_ID\"",
            "--checkpoint \"$CHECKPOINT\"",
        ):
            self.assertIn(expected, eval_script)
        self.assertNotIn('EVAL_ROOT="$DEPLOY_STORE/evaluations/xwam', eval_script)

        close_fridge_eval = (
            DEPLOY_ROOT / "eval_close_fridge_ab_xwam.sbatch"
        ).read_text(encoding="utf-8")
        for expected in (
            "#SBATCH --gpus-per-node=1",
            "#SBATCH --export=ALL",
            'REPO="${XWAM_EVAL_REPO:-$DEPLOY_STORE/src/xwam-robocasa365-eval}"',
            'EVAL_ROOT="$DEPLOY_IOPS/x-wam-eval/close-fridge-ab/$EVAL_ID"',
            "final-step=1000.ckpt",
            'CUDA_DEVICE="${XWAM_EVAL_CUDA_DEVICE:-0}"',
            'STEP_GPUS="${XWAM_EVAL_STEP_GPUS:-1}"',
            "--server-id 5",
            "--client-id 5",
            '--cuda-visible-devices "$CUDA_DEVICE"',
            "--single-task-checkpoint",
            "--single-task-name CloseFridge",
            '--cuda-device "$CUDA_DEVICE"',
            '--gpus-per-node="$STEP_GPUS"',
            'export MUJOCO_EGL_DEVICE_ID="$CUDA_DEVICE"',
            'if srun --overlap --exact',
            'if wait "$POLICY_STEP_PID"; then',
            'export XWAM_PHASE=close_fridge_eval_client',
            'inspect $LOG_ROOT/clients/client_05.log',
            "--episodes \"$EPISODES\"",
            "summary.json",
            "Independent evaluation repo must be clean",
        ):
            self.assertIn(expected, close_fridge_eval)
        self.assertNotIn("export MUJOCO_EGL_DEVICE_ID=0", close_fridge_eval)
        self.assertNotIn("set +e\nsrun --overlap --exact", close_fridge_eval)

        shared_eval = (
            DEPLOY_ROOT / "eval_fastwam_atomic9_xwam_b_shared4.sbatch"
        ).read_text(encoding="utf-8")
        for expected in (
            "#SBATCH --gpus-per-node=4",
            "#SBATCH --cpus-per-task=96",
            "eval_robocasa365_atomic9_6s9c_reserve4_use3.sbatch",
            '"export NUM_GPUS=3" "export NUM_SERVERS=6" "export NUM_CLIENTS=9"',
            "FastWAM script explicitly binds a process to GPU 3",
            "close_fridge_rgb_ratio00_seed42_4gpu",
            "XWAM_EVAL_STEP_GPUS=4",
            "XWAM_EVAL_CUDA_DEVICE=3",
            "FastWAM GPUs        : 0,1,2",
            "X-WAM-B GPU         : 3",
            "X-WAM-B ports       : 12005 / 13005",
            "FastWAM return code",
            "X-WAM-B return code",
        ):
            self.assertIn(expected, shared_eval)
        self.assertNotIn("XWAM_EVAL_CUDA_DEVICE=0", shared_eval)

        client_pool = (
            REPO_ROOT / "evaluation/launch_robocasa365_m6_client_pool.py"
        ).read_text(encoding="utf-8")
        for expected in (
            'physical_egl_device = args.cuda_visible_devices.split(",", 1)[0]',
            'environment["MUJOCO_EGL_DEVICE_ID"] = physical_egl_device',
            'environment["XWAM_ROBOCASA_IMPORT_EGL_DEVICE"] = physical_egl_device',
            'environment["EGL_DEVICE_ID"] = "0"',
            'environment["ROBOSUITE_RENDER_GPU_DEVICE_ID"] = "0"',
            'environment["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices',
        ):
            self.assertIn(expected, client_pool)

        task_client = (
            REPO_ROOT / "evaluation/run_robocasa365_m6_client.py"
        ).read_text(encoding="utf-8")
        self.assertIn('import robocasa  # noqa: F401', task_client)
        self.assertIn('os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"', task_client)
        self.assertLess(
            task_client.index("import robocasa  # noqa: F401"),
            task_client.index('os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"'),
        )

    def test_debug_log_archiver_preserves_formal_logs(self) -> None:
        script = (DEPLOY_ROOT / "archive_debug_logs_xwam.sh").read_text(
            encoding="utf-8"
        )
        for debug_prefix in (
            '"build-*"',
            '"validate-*"',
            '"wandb-overlay-*"',
            '"train4-resume-*"',
            '"m6-data-*"',
            '"m6-gate-*"',
            '"m6-formal-3053803*"',
        ):
            self.assertIn(debug_prefix, script)
        self.assertNotIn('"m6-formal-*"', script)
        self.assertNotIn("rm ", script)
        self.assertIn('DEBUG_LOG_ROOT="$LOG_ROOT/debug"', script)
        self.assertIn("target already exists", script)

        with tempfile.TemporaryDirectory() as tmp:
            deploy_store = Path(tmp)
            log_root = deploy_store / "logs/xwam"
            log_root.mkdir(parents=True)
            (log_root / "build-1.log").write_text("build", encoding="utf-8")
            (log_root / "m6-gate-2-audit.json").write_text("gate", encoding="utf-8")
            (log_root / "m6-formal-3.log").write_text("formal", encoding="utf-8")
            (log_root / "m6-formal-3053803-failure.txt").write_text(
                "preflight failure", encoding="utf-8"
            )
            result = subprocess.run(
                ["bash", str(DEPLOY_ROOT / "archive_debug_logs_xwam.sh")],
                cwd=REPO_ROOT,
                env={**os.environ, "XWAM_DEPLOY_STORE": str(deploy_store)},
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((log_root / "debug/build-1.log").is_file())
            self.assertTrue((log_root / "debug/m6-gate-2-audit.json").is_file())
            self.assertTrue(
                (log_root / "debug/m6-formal-3053803-failure.txt").is_file()
            )
            self.assertTrue((log_root / "m6-formal-3.log").is_file())
            self.assertIn("Archived 3 X-WAM debug log files", result.stdout)

        formal_script = (DEPLOY_ROOT / "train_m6_formal_xwam.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "$DEPLOY_STORE/logs/xwam/debug/m6-gate-3053436-audit.json",
            formal_script,
        )

    def test_clariden_error_trap_records_phase_command_and_exit_code(self) -> None:
        helper = DEPLOY_ROOT / "error_trap.sh"
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "failure.txt"
            env = {
                **os.environ,
                "XWAM_FAILURE_REPORT": str(report),
                "XWAM_PHASE": "unit_test_phase",
                "XWAM_MAIN_LOG": "/tmp/main.log",
                "XWAM_TRAIN_LOG": "/tmp/train.log",
                "WANDB_API_KEY": "must-not-appear",
            }
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'set -Eeuo pipefail; source "$1"; xwam_install_err_trap; false',
                    "bash",
                    str(helper),
                ],
                env=env,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 1)
            payload = report.read_text(encoding="utf-8")
            self.assertIn("phase=unit_test_phase", payload)
            self.assertIn("exit_code=1", payload)
            self.assertIn("command=false", payload)
            self.assertIn("failure_report=", result.stderr)
            self.assertNotIn("must-not-appear", payload + result.stderr)

    def test_clariden_train_smoke_is_one_gpu_one_step_without_checkpoint(self) -> None:
        hardware = (REPO_ROOT / "configs/hardware/gh200_96gb_debug.yaml").read_text(
            encoding="utf-8"
        )
        experiment = (
            REPO_ROOT / "configs/experiment/robocasa365_clariden_single_step.yaml"
        ).read_text(encoding="utf-8")
        script = (DEPLOY_ROOT / "smoke_train_xwam.sbatch").read_text(encoding="utf-8")
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

    def test_wandb_overlay_is_pinned_probed_and_atomically_published(self) -> None:
        requirements = (DEPLOY_ROOT / "wandb-overlay-clariden.txt").read_text(
            encoding="utf-8"
        )
        script = (DEPLOY_ROOT / "prepare_wandb_overlay_xwam.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("wandb==0.23.1", requirements)
        self.assertIn("sentry-sdk==2.58.0", requirements)
        self.assertIn("GitPython==3.1.58", requirements)
        self.assertIn("gitdb==4.0.12", requirements)
        self.assertIn("smmap==5.0.3", requirements)
        self.assertIn(
            "sha256:6cc984cf85feb2f8ee0451d76bc9fb7f39da94956bb8183e30d26284cf203b65",
            requirements,
        )
        self.assertIn(
            "sha256:688d1c704ddecf382ea3326f21a67453d4caa95592d722b7c780a36a9d23109e",
            requirements,
        )
        for dependency_hash in (
            "sha256:d331e722577f0fd7fc1f857419b3ecc07af66282b933d2a4d95f84a042fdd50f",
            "sha256:67073e15955400952c6565cc3e707c554a4eea2e428946f7a4c162fab9bd9bcf",
            "sha256:c106e05d5a61449cf6ba9a1e650227ecfb141590d2a98412103ff35d89fc7b2f",
        ):
            self.assertIn(dependency_hash, requirements)
        for expected in (
            "--require-hashes",
            "--no-deps",
            "mktemp -d",
            'mv "$STAGING" "$OVERLAY"',
            '"GitPython": "3.1.58"',
            '"gitdb": "4.0.12"',
            '"sentry-sdk": "2.58.0"',
            '"smmap": "5.0.3"',
            '"wandb": "0.23.1"',
            "validate_declared_dependencies",
            "dependency_errors.extend(errors)",
            "assert not dependency_errors, dependency_errors",
            "distribution_root.is_relative_to(overlay)",
            "wandb_module.is_relative_to(overlay)",
            "sentry_module.is_relative_to(overlay)",
            'mode="offline"',
            'run.log({"probe": 1.0}, step=0)',
            "xwam_wandb_overlay.txt",
            "error_trap.sh",
            "xwam_install_err_trap",
            "wandb_overlay_existing_probe",
            "wandb_overlay_install",
            "wandb_overlay_staging_probe",
            "wandb_overlay_final_probe",
            "sentry_sdk=%s",
            "gitpython=%s",
            "gitdb=%s",
            "smmap=%s",
            'FAILURE_REPORT="$DEPLOY_STORE/logs/xwam/wandb-overlay-${SLURM_JOB_ID}-failure.txt"',
        ):
            self.assertIn(expected, script)

    def test_clariden_four_gpu_resume_gate_is_debug_only_and_audited(self) -> None:
        hardware = (
            REPO_ROOT / "configs/hardware/gh200x4_96gb_resume_debug.yaml"
        ).read_text(encoding="utf-8")
        experiment = (
            REPO_ROOT / "configs/experiment/robocasa365_clariden_4gpu_resume_gate.yaml"
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
            "REUSE_INITIAL=true",
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

    def test_clariden_m6_data_preflight_is_atomic_only_and_machine_audited(
        self,
    ) -> None:
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

    def test_clariden_m6_formal_profile_gate_is_four_gpu_and_audited(self) -> None:
        script = (DEPLOY_ROOT / "smoke_m6_gate_xwam.sbatch").read_text(encoding="utf-8")
        for expected in (
            "#SBATCH --gpus-per-node=4",
            "gh200x4_96gb_gbs128.yaml",
            "gh200x4_96gb_gbs128_balanced.yaml",
            "gh200x4_96gb_gbs128_safe.yaml",
            "XWAM_M6_HARDWARE_CONFIG",
            "robocasa365_m6_gh200_gate.yaml",
            "robocasa365_m6_atomic_seen18_manifest.json",
            "robocasa365_m6_atomic_seen18_global_stats.json",
            "trainer_max_steps=4",
            "resume_checkpoint='$CHECKPOINT'",
            "audit_robocasa365_m6_gate.py",
            'test -z "$(git status --porcelain)"',
            "[PASS] X-WAM Clariden M6 4xGH200 formal-profile step 2 to 4 gate",
        ):
            self.assertIn(expected, script)
        self.assertNotIn("deepspeed_exclude_frozen_parameters=true", script)
        self.assertNotIn("deepspeed_stage=2", script)

    def test_clariden_m6_formal_training_is_restartable_and_machine_audited(
        self,
    ) -> None:
        script = (DEPLOY_ROOT / "train_m6_formal_xwam.sbatch").read_text(
            encoding="utf-8"
        )
        for expected in (
            "#SBATCH --gpus-per-node=4",
            "#SBATCH --time=12:00:00",
            "#SBATCH --export=ALL",
            "TOTAL_STEPS=16390",
            'CHUNK_STEPS="$TOTAL_STEPS"',
            'HOT_CHECKPOINT_ROOT="$DEPLOY_IOPS/xwam_run/$EXP_NAME/checkpoints"',
            'DURABLE_CHECKPOINT_ROOT="$DEPLOY_STORE/checkpoints/xwam/$EXP_NAME/checkpoints"',
            "gh200x4_96gb_gbs128.yaml",
            "robocasa365_m6_gh200_rgb_formal.yaml",
            "plan_robocasa365_m6_formal_chunk.py",
            "--quarantine-incomplete-root",
            '"checkpoint_dir=$HOT_CHECKPOINT_ROOT"',
            '"durable_checkpoint_dir=$DURABLE_CHECKPOINT_ROOT"',
            '"final_checkpoint_dir=$DURABLE_CHECKPOINT_ROOT"',
            '--expected-rolling-checkpoint-root "$HOT_CHECKPOINT_ROOT"',
            '--expected-durable-checkpoint-root "$DURABLE_CHECKPOINT_ROOT"',
            "flock -n 9",
            "audit_robocasa365_m6_formal_chunk.py",
            "audit_robocasa365_m6_formal_training.py",
            "error_trap.sh",
            "xwam_install_err_trap",
            "XWAM_PHASE=wandb_preflight",
            "XWAM_PHASE=training",
            "XWAM_PHASE=chunk_audit",
            'FAILURE_REPORT="$DEPLOY_STORE/logs/xwam/${LOG_STEM}-${SLURM_JOB_ID}-failure.txt"',
            'WANDB_OVERLAY="$DEPLOY_IOPS/python/xwam-wandb-0.23.1"',
            'WANDB_RUN_ID_FILE="$EXP_DIR/.wandb_run_id"',
            "WANDB_MODE=online",
            "WANDB_RESUME=allow",
            'if [[ -z "${WANDB_API_KEY:-}" ]]',
            "unset WANDB_IDENTITY_TOKEN_FILE",
            'assert os.environ.get("WANDB_API_KEY")',
            "wandb.login(verify=True)",
            '--expected-wandb-run-id "$WANDB_RUN_ID"',
            "resume_checkpoint=$RESUME_CHECKPOINT",
            'test -z "$(git status --porcelain)"',
            "[PASS] X-WAM Clariden M6 formal chunk completed",
            "[PASS] X-WAM Clariden M6 formal 5-epoch training completed",
        ):
            self.assertIn(expected, script)
        self.assertNotIn("CHUNK_STEPS=1000", script)
        self.assertNotIn(
            "hardware_config=configs/hardware/gh200x4_96gb_gbs128_safe.yaml", script
        )
        self.assertNotIn("deepspeed_stage=2", script)

    def test_clariden_m6_eight_gpu_training_is_a_separate_two_node_run(self) -> None:
        wrapper = (DEPLOY_ROOT / "train_m6_formal_xwam_8gpu.sbatch").read_text(
            encoding="utf-8"
        )
        shared = (DEPLOY_ROOT / "train_m6_formal_xwam.sbatch").read_text(
            encoding="utf-8"
        )
        hardware = (REPO_ROOT / "configs/hardware/gh200x8_96gb_gbs128.yaml").read_text(
            encoding="utf-8"
        )
        for expected in (
            "#SBATCH --nodes=2",
            "#SBATCH --ntasks-per-node=1",
            "#SBATCH --gpus-per-node=4",
            "#SBATCH --export=ALL",
            "XWAM_M6_NUM_NODES=2",
            "gh200x8_96gb_gbs128.yaml",
            "robocasa365_m6_atomic_seen18_rgb_seed42_8gpu",
        ):
            self.assertIn(expected, wrapper)
        for expected in (
            'srun --nodes="$FORMAL_NUM_NODES"',
            "--ntasks-per-node=1",
            "python -m torch.distributed.run",
            '--nnodes "$FORMAL_NUM_NODES"',
            '--nproc-per-node "$FORMAL_DEVICES_PER_NODE"',
            '--node-rank "$NODE_RANK"',
            '--master-addr "$MASTER_ADDR"',
            '--expected-world-size "$FORMAL_WORLD_SIZE"',
        ):
            self.assertIn(expected, shared)
        for expected in (
            "devices: 4",
            "batch_size_per_gpu: 16",
            "accumulate_grad_batches: 1",
            "global_batch_size: 128",
            "deepspeed_stage: 1",
            "formal_world_size: 8",
            "formal_num_nodes: 2",
            "formal_devices_per_node: 4",
        ):
            self.assertIn(expected, hardware)

    def test_close_fridge_ab_changes_only_the_clean_action_ratio(self) -> None:
        ratio05 = (
            REPO_ROOT
            / "configs/experiment/robocasa365_close_fridge_ratio05.yaml"
        ).read_text(encoding="utf-8")
        ratio00 = (
            REPO_ROOT
            / "configs/experiment/robocasa365_close_fridge_ratio00.yaml"
        ).read_text(encoding="utf-8")
        hardware = (
            REPO_ROOT
            / "configs/hardware/gh200x4_96gb_gbs128_single_task.yaml"
        ).read_text(encoding="utf-8")
        data = (
            REPO_ROOT / "configs/data/robocasa365_close_fridge_rgb.yaml"
        ).read_text(encoding="utf-8")
        script = (DEPLOY_ROOT / "train_close_fridge_ab_xwam.sbatch").read_text(
            encoding="utf-8"
        )

        normalized05 = "\n".join(ratio05.splitlines()[1:]).replace(
            "close_fridge_rgb_ratio05_seed42_4gpu",
            "close_fridge_rgb_ratioXX_seed42_4gpu",
        ).replace("clean_action_ratio: 0.5", "clean_action_ratio: X")
        normalized00 = "\n".join(ratio00.splitlines()[1:]).replace(
            "close_fridge_rgb_ratio00_seed42_4gpu",
            "close_fridge_rgb_ratioXX_seed42_4gpu",
        ).replace("clean_action_ratio: 0.0", "clean_action_ratio: X")
        self.assertEqual(normalized05, normalized00)

        for config, expected_ratio in ((ratio05, "0.5"), (ratio00, "0.0")):
            self.assertIn(f"clean_action_ratio: {expected_ratio}", config)
            self.assertIn("num_training_steps: 3000", config)
            self.assertIn("trainer_max_steps: 1000", config)
            self.assertIn("save_interval: 500", config)
            self.assertIn("save_top_k: 2", config)
            self.assertIn("checkpoint_post_save_barrier: true", config)
            self.assertIn("enable_wandb: true", config)

        for expected in (
            "task_name: CloseFridge",
            "augment: true",
            "normalization: panda_omron_v1",
        ):
            self.assertIn(expected, data)
        for expected in (
            "devices: 4",
            "batch_size_per_gpu: 16",
            "accumulate_grad_batches: 2",
            "global_batch_size: 128",
            "use_gradient_checkpointing: true",
            "deepspeed_stage: 1",
            "deepspeed_fp32_optimizer_states: true",
            "distributed_timeout_minutes: 90",
            "m6_formal_guard: false",
        ):
            self.assertIn(expected, hardware)
        for expected in (
            "#SBATCH --nodes=1",
            "#SBATCH --gpus-per-node=4",
            "#SBATCH --time=12:00:00",
            "#SBATCH --export=ALL",
            'VARIANT="${XWAM_CF_VARIANT:-}"',
            "ratio05)",
            "ratio00)",
            'TARGET_STEPS="${XWAM_CF_TARGET_STEPS:-1000}"',
            "1000|3000",
            "--expected-world-size 4",
            'srun --nodes=1 \\\n  --ntasks=1',
            'HOT_CHECKPOINT_ROOT="$DEPLOY_IOPS/xwam_run/$EXP_NAME/checkpoints"',
            'FINAL_CHECKPOINT_ROOT="$DEPLOY_STORE/checkpoints/xwam/$EXP_NAME/checkpoints"',
            "Git worktree must be clean before CloseFridge A/B training",
            "WANDB_API_KEY must be exported before sbatch submission",
            "[PASS] CloseFridge $VARIANT reached step",
        ):
            self.assertIn(expected, script)
        self.assertNotIn("python -m torch.distributed.run", script)
        self.assertNotIn("--nnodes 2", script)


if __name__ == "__main__":
    unittest.main()
