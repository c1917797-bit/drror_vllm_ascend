"""Synthetic closed-report comparison tests; no torch, service or NPU imports."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "layerwise_model_control_comparator_under_test",
    Path(__file__).resolve().parents[1] / "tools/compare_layerwise_model_control.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
SHA = "a" * 64


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor(shape, rank, pointer, offset=0, storage_bytes=None):
    stride, product = [], 1
    for n in reversed(shape):
        stride.insert(0, product)
        product *= n
    return {"shape": shape, "stride": stride, "dtype": "torch.bfloat16",
            "device": f"npu:{rank}", "numel": math.prod(shape), "element_size": 2,
            "storage_offset": offset, "storage_nbytes": storage_bytes or product * 2,
            "data_ptr": pointer + offset * 2, "storage_ptr": pointer, "contiguous": True}


def fixtures(root):
    dirs = [root / "legacy", root / "layerwise"]
    plan_path, layer_path = root / "legacy-plan.json", root / "layer-plan.json"
    plan = {"schema": "ascend-drrqr-plan/v1", "target_model_id": "Qwen/Qwen3.8-27B",
            "model_type": "qwen3_5_text", "target_head_k_dim": 64, "old_head_k_dim": 128,
            "num_key_heads": 16, "num_value_heads": 48, "head_v_dim": 128,
            "layer_types": ["linear_attention" if i in M.LAYERS else "full_attention" for i in range(64)],
            "keep_indices": {str(i): [h * 128 + j for h in range(16) for j in range(64)] for i in M.LAYERS}}
    plan_sha = write(plan_path, plan)
    refs = {"64": {"path": str(plan_path), "sha256": plan_sha}}
    layer_sha = write(layer_path, {"schema": "ascend-drrqr-layerwise-manifest/v1",
                                   "layer_head_k_dims": M.DIMS, "source_plans": refs})
    command = ["python", "-m", "vllm.entrypoints.openai.api_server"]
    for flag, value in {"--model": "/cache/austinov/Qwen3.8-27B", "--tensor-parallel-size": "4",
                        "--dtype": "bfloat16", "--max-model-len": "40960",
                        "--max-num-batched-tokens": "40960", "--max-num-seqs": "32",
                        "--worker-cls": "layerwise_probe_worker.LayerwiseProbeWorker",
                        "--hf-overrides": json.dumps({"text_config": {"linear_key_head_dim": 64}})}.items():
        command.extend([flag, value])
    command.extend(["--no-enable-prefix-caching", "--async-scheduling"])
    for side, directory in enumerate(dirs):
        ppath, psha = (plan_path, plan_sha) if side == 0 else (layer_path, layer_sha)
        manifest = {"schema": "drrqr-layerwise-service/v1",
                    "plan_kind": "uniform" if side == 0 else "layerwise",
                    "shared_hf_head_k_dim": 64, "layer_head_k_dims": M.DIMS,
                    "num_linear_layers": 48, "sum_layer_head_k_dims": 3072,
                    "common_optimizations": M.COMMON, "profiler_config": None,
                    "physical_npu_contract": [4, 5, 6, 7], "container_npu_namespace": [0, 1, 2, 3],
                    "wheel_sha256": SHA, "build_manifest_sha256": SHA,
                    "reference_manifest_sha256": SHA, "wrapper_sha256": SHA,
                    "plan_sha256": psha, "plan_path": str(ppath), "source_plans": None if side == 0 else refs,
                    "module_sha256": {"x.py": SHA}, "build_source_sha256": {"x.py": SHA},
                    "source_sha256": {"/runtime/x.py": SHA}, "plugin_version": "0.1.12",
                    "command": command, "diagnostic_worker": {"class": "LayerwiseProbeWorker",
                                                             "worker_sha256": SHA},
                    "inherited_runtime_environment": {},
                    "plugin_flags": {
                        "VLLM_ASCEND_DRRQR_ENABLE": "1", "VLLM_ASCEND_DRRQR_CAPTURE_ENABLE": "0",
                        "VLLM_ASCEND_DRRQR_STRICT": "1", "VLLM_ASCEND_DRRQR_REQUIRE_RUNTIME_HOOKS": "1",
                        "VLLM_ASCEND_DRRQR_CONV_LAYOUT": "1", "VLLM_ASCEND_DRRQR_PREFILL_MC2": "1",
                        "VLLM_ASCEND_DRRQR_PREFILL_MC2_MIXED": "1",
                        "VLLM_ASCEND_DRRQR_PLAN_PATH": str(ppath), "VLLM_ASCEND_DRRQR_PLAN_SHA256": psha,
                        "VLLM_ASCEND_DRRQR_EVIDENCE_FILE": str(directory / "evidence.jsonl")}}
        manifest_sha = write(directory / "service-manifest.json", manifest)
        write(directory / "service-process.json", {"pid": 500 + side, "pgid": 500 + side})
        write(directory / "service-exit.json",
              {"schema": "drrqr-layerwise-service-exit/v1", "stop_reason": "stop_request",
               "group_remaining": False, "exit_code": -15, "started_epoch_s": 1,
               "ended_epoch_s": 2, "service_pid": 500 + side, "process_group": 500 + side})
        for rank in range(4):
            for phase in ("load", "cache"):
                report = {"schema": "ascend-drrqr-layerwise-worker-probe/v1",
                          "status": "passed", "phase": phase, "rank": rank, "local_rank": rank,
                          "plan_sha256": psha, "plan_class": "DrrqrPlan" if side == 0 else "LayerwiseDrrqrPlan",
                          "protocol": M.PROTOCOL, "worker_source": {"sha256": SHA},
                          "pid": 1000 + side * 100 + rank, "started_unix_ns": 1, "finished_unix_ns": 2,
                          "runtime_sources": {name: {"path": "/runtime/" + name, "sha256": SHA}
                                              for name in ("worker", "runner", "gdn")},
                          "layer_count": 48, "layers": []}
                for i in M.LAYERS:
                    ptr = 1000000000 + side * 1000000000 + i * 1000000
                    record = {"layer": i, "module_name": f"layers.{i}.linear_attn",
                              "prefix": f"layers.{i}.linear_attn",
                              "expected_attributes": {"head_k_dim": 64, "layer_idx": i,
                                                      "tp_size": 4, "tp_rank": rank},
                              "local_qkv_rows": 2048, "projection_layout": "qkv_and_z",
                              "state_shapes": [[3, 2048], [12, 128, 64]],
                              "state_dtypes": ["torch.bfloat16", "torch.bfloat16"]}
                    if phase == "load":
                        record["parameters"] = [{"name": "conv1d.weight", "sha256": SHA,
                                                 **tensor([3, 3], rank, ptr)}]
                    else:
                        record.update(cache_group_index=i % 3, runner_cache_index=i,
                                      binding_identity_verified=True,
                                      mamba_spec={"class": "vllm.v1.kv_cache_interface.MambaSpec",
                                                  "shapes": record["state_shapes"], "dtypes": record["state_dtypes"],
                                                  "block_size": 16, "page_size_padded": 208896,
                                                  "page_size_bytes": 208896, "mamba_type": "gdn_attention",
                                                  "mamba_cache_mode": "none", "num_speculative_blocks": 0})
                        record["states"] = [
                            {"index": 0, **tensor([2, 3, 2048], rank, ptr, storage_bytes=417792)},
                            {"index": 1, **tensor([2, 12, 128, 64], rank, ptr, offset=12288,
                                                storage_bytes=417792)}]
                    report["layers"].append(record)
                if phase == "load":
                    report["parameter_bytes_hashed"] = 48 * 18
                else:
                    report.update(num_blocks=2, cache_values_read=False)
                write(directory / "worker-probe" / f"rank{rank}.{phase}.json", report)
        sequence = {"schema": "drrqr-layerwise-integration-sequence/v1", "status": "requests_complete",
                    "service_manifest_sha256": manifest_sha, "script_sha256": SHA,
                    "protocol": {"thinking": False, "temperature": 0, "seed": 0,
                                 "requests": 4, "batches": [1, 3], "output_tokens": 32,
                                 "ignore_eos": True, "logprobs": 5, "input_tokens": [10, 20, 30, 40]},
                    "comparison_criteria": M.CRITERIA, "results": [
                        {"index": i, "input_tokens": (i + 1) * 10, "prompt_sha256": SHA,
                         "token_ids": list(range(32)), "selected_logprobs": [-1.0] * 32}
                        for i in range(4)]}
        write(directory / "integration-sequence.json", sequence)
    return dirs


class LayerwiseModelControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.legacy, self.layerwise = fixtures(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def change(self, relative, mutate):
        path = self.layerwise / relative
        data = json.loads(path.read_text())
        mutate(data)
        write(path, data)

    def test_complete_exact_control_passes_and_hashes_every_input(self):
        report = M.compare_dirs(self.legacy, self.layerwise)
        self.assertEqual(report["status"], "passed", report)
        self.assertTrue(report["selected_logprobs_exact"])
        self.assertEqual(report["selected_logprob_count"], 128)
        self.assertEqual(len(report["input_sha256"]), 26)
        self.assertTrue(all(r["cache_alias_pattern_exact"] for r in report["ranks"]))

    def test_missing_exit_and_partial_worker_fail_closed(self):
        exit_path = self.layerwise / "service-exit.json"
        saved = exit_path.read_bytes()
        exit_path.unlink()
        self.assertEqual(M.compare_dirs(self.legacy, self.layerwise)["status"], "failed")
        exit_path.write_bytes(saved)
        self.change("worker-probe/rank2.cache.json", lambda d: d.update(status="running"))
        report = M.compare_dirs(self.legacy, self.layerwise)
        self.assertEqual(report["status"], "failed")
        self.assertIn("incomplete_or_invalid_input", report["failures"])

    def test_incomplete_layer_and_parameter_byte_coverage_fail(self):
        self.change("worker-probe/rank1.load.json", lambda d: d["layers"].pop())
        self.assertEqual(M.compare_dirs(self.legacy, self.layerwise)["status"], "failed")

    def test_weight_hash_difference_is_not_hidden_by_equal_shape(self):
        self.change("worker-probe/rank0.load.json",
                    lambda d: d["layers"][0]["parameters"][0].update(sha256="b" * 64))
        report = M.compare_dirs(self.legacy, self.layerwise)
        self.assertEqual(report["status"], "failed")
        self.assertIn("rank0.layer0.load.parameters_exact", report["failures"])

    def test_alias_difference_detected_while_pointer_values_are_ignored(self):
        def separate_storage(d):
            state = d["layers"][0]["states"][1]
            state["storage_ptr"] += 500000
            state["data_ptr"] += 500000
        self.change("worker-probe/rank0.cache.json", separate_storage)
        report = M.compare_dirs(self.legacy, self.layerwise)
        self.assertEqual(report["status"], "failed")
        self.assertIn("rank0.cache_alias_pattern", report["failures"])

    def test_within_fixed_logprob_tolerance_reports_nonexact(self):
        self.change("integration-sequence.json",
                    lambda d: d["results"][0]["selected_logprobs"].__setitem__(0, -0.99))
        report = M.compare_dirs(self.legacy, self.layerwise)
        self.assertEqual(report["status"], "passed", report)
        self.assertFalse(report["selected_logprobs_exact"])
        self.assertAlmostEqual(report["selected_logprob_max_abs"], 0.01)

    def test_each_logprob_threshold_and_token_identity_are_enforced(self):
        path = self.layerwise / "integration-sequence.json"
        base = json.loads(path.read_text())
        for label, mutate, failure in (
            ("maximum", lambda d: d["results"][0]["selected_logprobs"].__setitem__(0, -0.97),
             "selected_logprob_max_abs"),
            ("mean", lambda d: [r.update(selected_logprobs=[-0.99] * 32) for r in d["results"]],
             "selected_logprob_mean_abs"),
            ("tokens", lambda d: d["results"][0]["token_ids"].__setitem__(0, 99), "request0.token_ids"),
        ):
            data = copy.deepcopy(base)
            mutate(data)
            write(path, data)
            with self.subTest(label=label):
                report = M.compare_dirs(self.legacy, self.layerwise)
                self.assertEqual(report["status"], "failed")
                self.assertIn(failure, report["failures"])

    def test_bound_plan_tamper_rejected(self):
        path = self.root / "layer-plan.json"
        plan = json.loads(path.read_text())
        plan["source_plans"]["64"]["sha256"] = "b" * 64
        write(path, plan)
        report = M.compare_dirs(self.legacy, self.layerwise)
        self.assertEqual(report["status"], "failed")
        self.assertIn("plan document hash changed", report["error"]["message"])

    def test_same_wheel_and_common_optimization_required(self):
        path = self.layerwise / "service-manifest.json"
        data = json.loads(path.read_text())
        data["wheel_sha256"] = "b" * 64
        digest = write(path, data)
        self.change("integration-sequence.json", lambda d: d.update(service_manifest_sha256=digest))
        report = M.compare_dirs(self.legacy, self.layerwise)
        self.assertEqual(report["status"], "failed")
        self.assertIn("service.wheel_sha256", report["failures"])

    def test_output_is_exclusive_and_failure_report_is_written(self):
        output = self.root / "comparison.json"
        with patch.object(M, "RESULT_ROOT", self.root):
            self.assertEqual(M.main(["--legacy-dir", str(self.legacy), "--layerwise-dir",
                                     str(self.layerwise), "--output", str(output)]), 0)
            with self.assertRaises(SystemExit):
                M.main(["--legacy-dir", str(self.legacy), "--layerwise-dir",
                        str(self.layerwise), "--output", str(output)])
        self.assertEqual(json.loads(output.read_text())["status"], "passed")

    def test_nonfinite_or_missing_logprob_fails_closed(self):
        self.change("integration-sequence.json",
                    lambda d: d["results"][0]["selected_logprobs"].__setitem__(0, float("nan")))
        self.assertEqual(M.compare_dirs(self.legacy, self.layerwise)["status"], "failed")

    def test_malformed_worker_field_is_a_failed_report(self):
        self.change("worker-probe/rank0.load.json", lambda d: d.update(worker_source=[]))
        report = M.compare_dirs(self.legacy, self.layerwise)
        self.assertEqual(report["status"], "failed")
        self.assertIn("incomplete_or_invalid_input", report["failures"])


if __name__ == "__main__":
    unittest.main()
