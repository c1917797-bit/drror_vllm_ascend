"""Synthetic A/A gate tests; no torch, NPU or model process is imported."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("uniform_repeatability_under_test",
                                            ROOT / "tools/compare_uniform_repeatability.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "uniform_fixture_source", ROOT / "tests/test_layerwise_model_control.py")
F = importlib.util.module_from_spec(FIXTURE_SPEC)
FIXTURE_SPEC.loader.exec_module(F)


def make_uniform_fixtures(root):
    first, repeat = F.fixtures(root)
    first_manifest = json.loads((first / "service-manifest.json").read_text())
    for side, directory in enumerate((first, repeat)):
        manifest = json.loads((directory / "service-manifest.json").read_text())
        for name in ("plan_kind", "plan_path", "plan_sha256", "source_plans"):
            manifest[name] = first_manifest[name]
        for name in ("VLLM_ASCEND_DRRQR_PLAN_PATH", "VLLM_ASCEND_DRRQR_PLAN_SHA256"):
            manifest["plugin_flags"][name] = first_manifest["plugin_flags"][name]
        manifest_sha = F.write(directory / "service-manifest.json", manifest)
        sequence_path = directory / "integration-sequence.json"
        sequence = json.loads(sequence_path.read_text())
        sequence["service_manifest_sha256"] = manifest_sha
        F.write(sequence_path, sequence)
        for rank in range(4):
            for phase in ("load", "cache"):
                path = directory / "worker-probe" / f"rank{rank}.{phase}.json"
                report = json.loads(path.read_text())
                report.update(plan_sha256=manifest["plan_sha256"], plan_class="DrrqrPlan",
                              parent_worker={"path": "/runtime/worker.py", "sha256": F.SHA})
                if phase == "load":
                    report["model_class"] = {"path": "/runtime/model.py", "sha256": F.SHA}
                    for layer in report["layers"]:
                        base = layer["parameters"][0]
                        layer["parameters"] = []
                        layer["module_class"] = {"path": "/runtime/gdn.py", "sha256": F.SHA}
                        layer["method_sources"] = {"forward": {"path": "/runtime/gdn.py", "sha256": F.SHA}}
                        for index, name in enumerate(sorted(M.PARAMETER_NAMES)):
                            p = copy.deepcopy(base)
                            p["name"] = name
                            p["data_ptr"] += index * 128
                            p["storage_ptr"] += index * 128
                            layer["parameters"].append(p)
                    report["parameter_bytes_hashed"] = 48 * 7 * 18
                else:
                    report["num_blocks"] = 925
                    for layer in report["layers"]:
                        states = layer["states"]
                        total_elements = 925 * (6144 + 98304)
                        for state in states:
                            state["shape"][0] = 925
                            state["numel"] = math.prod(state["shape"])
                            state["storage_nbytes"] = total_elements * 2
                            if state["index"] == 1:
                                state["storage_offset"] = 925 * 6144
                                state["data_ptr"] = state["storage_ptr"] + state["storage_offset"] * 2
                F.write(path, report)
    return first, repeat


class UniformRepeatabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.first, self.repeat = make_uniform_fixtures(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def change(self, relative, mutate, directory=None):
        path = (directory or self.repeat) / relative
        data = json.loads(path.read_text())
        mutate(data)
        return F.write(path, data)

    def test_complete_same_uniform_control_passes(self):
        report = M.compare_runs(self.first, self.repeat)
        self.assertEqual(report["status"], "passed", report)
        self.assertEqual(len(report["input_sha256"]), 25)
        self.assertTrue(report["selected_logprobs_exact"])
        self.assertEqual(report["selected_logprob_count"], 128)
        self.assertTrue(all(r["parameter_counts"] == [336, 336] for r in report["ranks"]))
        self.assertTrue(all(r["observed_num_blocks"] == [925, 925] for r in report["ranks"]))
        self.assertEqual(report["ab_status_unchanged"].split(";")[0], "failed")

    def test_validator_exact_bytes_must_match_before_execution(self):
        path = self.root / "tampered-validator.py"
        path.write_text("raise RuntimeError('must not execute')\n")
        with self.assertRaisesRegex(ValueError, "validator bytes changed"):
            M.load_validator(path)

    def test_first_isolated_request_and_each_token_are_reported(self):
        self.change("integration-sequence.json",
                    lambda d: d["results"][0]["selected_logprobs"].__setitem__(3, -0.95))
        report = M.compare_runs(self.first, self.repeat)
        self.assertEqual(report["status"], "failed")
        first = report["first_isolated_request"]
        self.assertTrue(first["isolated_first_request"])
        self.assertEqual(len(first["tokens"]), 32)
        self.assertAlmostEqual(first["tokens"][3]["delta"], 0.05)
        self.assertIn("selected_logprob_max_abs", report["failures"])
        self.assertTrue(report["ab_status_unchanged"].startswith("failed"))

    def test_global_mean_and_token_gates_are_preserved(self):
        self.change("integration-sequence.json",
                    lambda d: [r.update(selected_logprobs=[-0.99] * 32) for r in d["results"]])
        report = M.compare_runs(self.first, self.repeat)
        self.assertEqual(report["status"], "failed")
        self.assertIn("selected_logprob_mean_abs", report["failures"])
        self.change("integration-sequence.json",
                    lambda d: d["results"][1]["token_ids"].__setitem__(4, 99))
        self.assertIn("request1.token_ids", M.compare_runs(self.first, self.repeat)["failures"])

    def test_nonexact_logprob_within_existing_gate_is_recorded(self):
        self.change("integration-sequence.json",
                    lambda d: d["results"][0]["selected_logprobs"].__setitem__(0, -0.99))
        report = M.compare_runs(self.first, self.repeat)
        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["selected_logprobs_exact"])

    def test_plan_or_manifest_binding_difference_is_rejected(self):
        self.change("service-manifest.json", lambda d: d.update(plan_kind="layerwise"))
        report = M.compare_runs(self.first, self.repeat)
        self.assertEqual(report["status"], "failed")
        self.assertIn("incomplete_or_invalid_input", report["failures"])

    def test_missing_exit_and_failed_phase_fail_closed(self):
        path = self.repeat / "service-exit.json"
        original = path.read_bytes()
        path.unlink()
        self.assertEqual(M.compare_runs(self.first, self.repeat)["status"], "failed")
        path.write_bytes(original)
        self.change("worker-probe/rank3.cache.json", lambda d: d.update(status="failed"))
        self.assertEqual(M.compare_runs(self.first, self.repeat)["status"], "failed")

    def test_dispatch_identity_difference_is_reported(self):
        self.change("worker-probe/rank1.load.json",
                    lambda d: d["layers"][0]["method_sources"]["forward"].update(sha256="b" * 64))
        report = M.compare_runs(self.first, self.repeat)
        self.assertIn("rank1.layer0.load.method_sources", report["failures"])

    def test_missing_parameter_is_not_hidden_by_equal_remaining_weights(self):
        self.change("worker-probe/rank0.load.json", lambda d: d["layers"][0]["parameters"].pop())
        self.assertEqual(M.compare_runs(self.first, self.repeat)["status"], "failed")

    def test_cache_block_difference_is_recorded_as_observed(self):
        self.change("worker-probe/rank0.cache.json", lambda d: d.update(num_blocks=924))
        report = M.compare_runs(self.first, self.repeat)
        self.assertIn("rank0.num_blocks", report["failures"])
        self.assertEqual(report["ranks"][0]["observed_num_blocks"], [925, 924])
        self.assertFalse(report["ranks"][0]["num_blocks_match_expected"])

    def test_wheel_change_is_not_confused_with_repeatability(self):
        digest = self.change("service-manifest.json", lambda d: d.update(wheel_sha256="b" * 64))
        self.change("integration-sequence.json", lambda d: d.update(service_manifest_sha256=digest))
        report = M.compare_runs(self.first, self.repeat)
        self.assertIn("service.wheel_sha256", report["failures"])

    def test_output_is_exclusive_and_failure_persists(self):
        output = self.root / "repeatability.json"
        self.change("integration-sequence.json",
                    lambda d: d["results"][0]["selected_logprobs"].__setitem__(0, -0.9))
        arguments = ["--first-dir", str(self.first), "--repeat-dir", str(self.repeat), "--output", str(output)]
        with patch.object(M, "RESULT_ROOT", self.root):
            self.assertEqual(M.main(arguments), 1)
            original = output.read_bytes()
            with self.assertRaises(SystemExit):
                M.main(arguments)
            self.assertEqual(output.read_bytes(), original)
        self.assertEqual(json.loads(output.read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()

