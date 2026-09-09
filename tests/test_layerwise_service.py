"""Launcher checks and disposable CPU process cleanup; no model/NPU is run."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

SPEC = importlib.util.spec_from_file_location(
    "layerwise_service_under_test",
    Path(__file__).resolve().parents[1] / "tools" / "run_layerwise_service.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def args(**overrides):
    data = dict(plan_kind="layerwise", plan_path=Path("/drrqr-plans/a.json"),
                plan_sha256="a" * 64, diagnostic_worker=None,
                diagnostic_worker_sha256=None,
                diagnostic_worker_class="LayerwiseProbeWorker")
    data.update(overrides)
    return SimpleNamespace(**data)


class LayerwiseServiceTests(unittest.TestCase):
    def test_shared_envelope_matches_uniform_control_and_mixed_maximum(self):
        reference = SimpleNamespace(old_head_k_dim=128, num_key_heads=16,
                                    num_value_heads=48, head_v_dim=128,
                                    target_head_k_dim=64,
                                    keep_indices=((0, (1,)), (1, (1,))))
        uniform = MODULE.describe_plan(reference, "uniform")
        all64 = SimpleNamespace(reference=reference, layer_dims=((0, 64), (1, 64)),
                                max_head_k_dim=64)
        mixed = SimpleNamespace(reference=reference,
                                layer_dims=((0, 128), (1, 32), (2, 32), (3, 64)),
                                max_head_k_dim=128)
        detail = MODULE.describe_plan(all64, "layerwise")
        self.assertEqual(uniform["shared_hf_head_k_dim"], detail["shared_hf_head_k_dim"])
        self.assertEqual(MODULE.service_command(64), MODULE.service_command(detail["shared_hf_head_k_dim"]))
        hetero = MODULE.describe_plan(mixed, "layerwise")
        self.assertEqual(hetero["sum_layer_head_k_dims"], 256)
        self.assertEqual(hetero["shared_hf_head_k_dim"], 128)
        self.assertEqual(hetero["layer_head_k_dims"]["1"], 32)

    def test_unaligned_width_or_wrong_max_rejected(self):
        reference = SimpleNamespace(old_head_k_dim=128, num_key_heads=16,
                                    num_value_heads=48, head_v_dim=128)
        for dims, maximum in ((((0, 104),), 104), (((0, 64),), 128)):
            with self.subTest(dims=dims), self.assertRaises(ValueError):
                MODULE.describe_plan(SimpleNamespace(reference=reference,
                    layer_dims=dims, max_head_k_dim=maximum), "layerwise")

    def test_common_flags_and_stale_capture_removed(self):
        env, flags, _, worker_args, diagnostic = MODULE.service_environment(
            args(), Path("/drrqr-results/new"),
            {"ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3",
             "VLLM_ASCEND_DRRQR_CAPTURE_DIR": "/old",
             "DRRQR_LAYERWISE_OLD": "stale"})
        self.assertEqual(flags["VLLM_ASCEND_DRRQR_CONV_LAYOUT"], "1")
        self.assertEqual(flags["VLLM_ASCEND_DRRQR_PREFILL_MC2_MIXED"], "1")
        self.assertEqual(flags["VLLM_ASCEND_DRRQR_CAPTURE_ENABLE"], "0")
        self.assertNotIn("VLLM_ASCEND_DRRQR_CAPTURE_DIR", env)
        self.assertNotIn("DRRQR_LAYERWISE_OLD", env)
        self.assertEqual(worker_args, [])
        self.assertIsNone(diagnostic)

    def test_namespace_determinism_and_source_shadow_rejected(self):
        for key, value in (
            ("ASCEND_RT_VISIBLE_DEVICES", "4,5,6,7"),
            ("DRRQR_DIAGNOSTIC_DETERMINISM", "1"),
            ("VLLM_BATCH_INVARIANT", "1"),
            ("HCCL_DETERMINISTIC", "strict"),
            ("PYTHONPATH", str(MODULE.ROOT)),
        ):
            inherited = {"ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3", key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                MODULE.service_environment(args(), Path("/drrqr-results/new"), inherited)

    def test_worker_whitelist_and_bound_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tools").mkdir()
            worker = root / "tools/layerwise_probe_worker.py"
            worker.write_text("class LayerwiseProbeWorker: pass\n")
            config = args(diagnostic_worker=worker,
                          diagnostic_worker_sha256=MODULE.digest(worker))
            with patch.object(MODULE, "ROOT", root):
                env, _, _, command, diagnostic = MODULE.service_environment(
                    config, Path("/drrqr-results/new"),
                    {"ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3"})
                self.assertEqual(command, ["--worker-cls", "layerwise_probe_worker.LayerwiseProbeWorker"])
                self.assertEqual(env["DRRQR_LAYERWISE_PROBE_DIR"], "/drrqr-results/new/worker-probe")
                self.assertEqual(diagnostic["worker_sha256"], MODULE.digest(worker))
                config.diagnostic_worker_sha256 = "b" * 64
                with self.assertRaises(ValueError):
                    MODULE.service_environment(config, Path("/drrqr-results/new"),
                        {"ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3"})
                config.diagnostic_worker = root / "tools/runtime_determinism_worker.py"
                with self.assertRaises(ValueError):
                    MODULE.service_environment(config, Path("/drrqr-results/new"),
                        {"ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3"})

    def test_build_validates_wheel_snapshot_installed_and_live_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            build_dir = Path(tmp) / "build"
            installed = Path(tmp) / "site-packages/drror_vllm_ascend"
            content = b'"""fixture package"""\n'
            expected = {"drror_vllm_ascend/__init__.py": sha(content),
                        "pyproject.toml": sha(b"fixture\n")}
            for base in (root, build_dir / "source"):
                (base / "drror_vllm_ascend").mkdir(parents=True)
                (base / "drror_vllm_ascend/__init__.py").write_bytes(content)
                (base / "pyproject.toml").write_bytes(b"fixture\n")
            installed.mkdir(parents=True)
            (installed / "__init__.py").write_bytes(content)
            wheel = build_dir / "fixture.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("drror_vllm_ascend/__init__.py", content)
                archive.writestr("fixture.dist-info/METADATA", "Name: fixture\nVersion: 0.1.12\n")
            manifest = build_dir / "build-manifest.json"
            manifest.write_text(json.dumps({"schema": "drrqr-local-wheel-build/v1",
                "source_root": str(root), "source_sha256": expected,
                "wheel": str(wheel), "wheel_sha256": MODULE.digest(wheel)}))
            config = args(build_manifest=manifest,
                          build_manifest_sha256=MODULE.digest(manifest),
                          plugin_version="0.1.12")
            with patch.object(MODULE, "ROOT", root), \
                 patch.object(MODULE.importlib.util, "find_spec",
                              return_value=SimpleNamespace(origin=str(installed / "__init__.py"))), \
                 patch.object(MODULE.importlib.metadata, "version", return_value="0.1.12"):
                self.assertEqual(MODULE.verify_build(config)["module_sha256"], {
                    "drror_vllm_ascend/__init__.py": sha(content)})
                for target in (installed / "__init__.py",
                               build_dir / "source/drror_vllm_ascend/__init__.py",
                               root / "drror_vllm_ascend/__init__.py"):
                    target.write_bytes(b"changed\n")
                    with self.subTest(target=str(target)), self.assertRaises(ValueError):
                        MODULE.verify_build(config)
                    target.write_bytes(content)
                (installed / "extra.py").write_text("pass\n")
                with self.assertRaises(ValueError):
                    MODULE.verify_build(config)

    def test_bound_document_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "doc.json"
            path.write_text("{}")
            checksum = MODULE.digest(path)
            self.assertEqual(MODULE.read_bound_json(path, checksum), {})
            path.write_text('{"changed": true}')
            with self.assertRaises(ValueError):
                MODULE.read_bound_json(path, checksum)

    def test_protocol_is_fixed_and_profiler_absent(self):
        command = MODULE.service_command(128)
        self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "4")
        self.assertIn("--no-enable-prefix-caching", command)
        self.assertIn("--async-scheduling", command)
        self.assertNotIn("--profiler-config", command)
        override = json.loads(command[command.index("--hf-overrides") + 1])
        self.assertEqual(override, {"text_config": {"linear_key_head_dim": 128}})

    def test_main_precreates_probe_directory_only_for_diagnostic_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_root = Path(tmp)
            for enabled in (False, True):
                output = result_root / ("diagnostic" if enabled else "ordinary")
                config = args(output_dir=output, reference_manifest_sha256="a" * 64,
                              max_service_seconds=1800, term_grace_seconds=30)
                diagnostic = {"output_dir": str(output / "worker-probe")} if enabled else None
                def inspect_before_launch(command, env, result, budget, grace):
                    self.assertTrue(result.is_dir())
                    self.assertEqual((result / "worker-probe").is_dir(), enabled)
                    self.assertTrue((result / "service-manifest.json").is_file())
                    return 0
                with self.subTest(enabled=enabled), \
                     patch.object(MODULE, "RESULT_ROOT", result_root), \
                     patch.object(MODULE, "parse_args", return_value=(None, config)), \
                     patch.object(MODULE, "service_environment",
                                  return_value=({}, {}, {}, [], diagnostic)), \
                     patch.object(MODULE, "verify_build", return_value={}), \
                     patch.object(MODULE, "verify_runtime", return_value={}), \
                     patch.object(MODULE, "verify_plan", return_value={"shared_hf_head_k_dim": 64}), \
                     patch.object(MODULE.socket, "socket"), \
                     patch.object(MODULE, "run_service", side_effect=inspect_before_launch):
                    self.assertEqual(MODULE.main(), 0)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux process groups/subreaper")
    def test_stop_request_reaps_own_group_and_preserves_unrelated_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                         start_new_session=True)
            try:
                # The disposable child creates an orphan in the same group and a
                # stop request. No model, HTTP service, torch or NPU is imported.
                code = ("import pathlib,subprocess,sys,time; "
                        "subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
                        "pathlib.Path('stop.request').touch(); time.sleep(30)")
                result = MODULE.run_service([sys.executable, "-c", code], os.environ.copy(),
                                            output, 10, 1)
                self.assertEqual(result, 0)
                self.assertIsNone(bystander.poll())
                record = json.loads((output / "service-exit.json").read_text())
                self.assertEqual(record["stop_reason"], "stop_request")
                self.assertEqual(record["exit_code"], -signal.SIGTERM)
                self.assertFalse(record["group_remaining"])
                self.assertFalse(MODULE.group_exists(record["process_group"]))
            finally:
                bystander.terminate()
                bystander.wait(timeout=5)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux process groups/subreaper")
    def test_timeout_escalation_is_bounded_and_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            code = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)"
            started = time.monotonic()
            result = MODULE.run_service([sys.executable, "-c", code], os.environ.copy(),
                                        output, 1, 1)
            self.assertEqual(result, 124)
            self.assertLess(time.monotonic() - started, 6)
            record = json.loads((output / "service-exit.json").read_text())
            self.assertEqual(record["signals"], ["SIGTERM", "SIGKILL"])
            self.assertEqual(record["exit_code"], -signal.SIGKILL)
            self.assertFalse(record["group_remaining"])


if __name__ == "__main__":
    unittest.main()
