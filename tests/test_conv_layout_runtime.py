"""Runtime integration gates for the opt-in load-time layout preparation."""
import os
import sys
import types
import unittest
from unittest import mock

import torch

from drror_vllm_ascend import envs
from drror_vllm_ascend.envs import DrrqrConfig
from drror_vllm_ascend.patches import conv_layout_runtime as runtime


class Gdn(torch.nn.Module):
    def __init__(self, dk=64):
        super().__init__()
        self.tp_size, self.num_k_heads, self.num_v_heads = 4, 16, 48
        self.head_k_dim, self.head_v_dim = dk, 128
        self.conv1d = torch.nn.Module()
        self.conv1d.weight = torch.nn.Parameter(torch.randn(2 * 4 * dk + 1536, 1, 4,
                                                          dtype=torch.bfloat16), requires_grad=False)


class Model:
    def __init__(self, missing=False, dk=64):
        self.targets = [(f"language_model.layers.{i}.linear_attn", Gdn(dk))
                        for i in sorted(runtime.LINEAR_LAYERS)]
        if missing:
            self.targets.pop()

    def named_modules(self):
        return iter(self.targets)


class ConvRuntimeTests(unittest.TestCase):
    def test_independent_opt_in_requires_evidence_but_not_pruning_plan(self):
        with mock.patch.dict(os.environ, {"VLLM_ASCEND_DRRQR_CONV_LAYOUT": "1",
                                         "VLLM_ASCEND_DRRQR_EVIDENCE_FILE": "/tmp/evidence.jsonl"}, clear=True):
            cfg = envs.get_config()
            self.assertTrue(cfg.conv_layout)
            self.assertFalse(cfg.enable)
            self.assertEqual(cfg.plan_path, "")
        with mock.patch.dict(os.environ, {"VLLM_ASCEND_DRRQR_CONV_LAYOUT": "1"}, clear=True):
            with self.assertRaisesRegex(ValueError, "EVIDENCE_FILE"):
                envs.get_config()

    def test_capture_conflict(self):
        with mock.patch.dict(os.environ, {"VLLM_ASCEND_DRRQR_CONV_LAYOUT": "1",
                                         "VLLM_ASCEND_DRRQR_CAPTURE_ENABLE": "1"}, clear=True):
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                envs.get_config()

    def test_all_targets_exact_and_no_missing_target_mutation(self):
        for missing in (False, True):
            model = Model(missing)
            before = [m.conv1d.weight.detach().clone() for _, m in model.targets]
            if missing:
                with self.assertRaisesRegex(RuntimeError, "all 48"):
                    runtime.prepare_model(model, Gdn, expected_dk=64, require_npu=False)
                self.assertTrue(all(m.conv1d.weight.is_contiguous() for _, m in model.targets))
            else:
                rows = runtime.prepare_model(model, Gdn, expected_dk=64, require_npu=False)
                self.assertEqual(len(rows), 48)
                self.assertTrue(all(r["exact_values_verified"] for r in rows))
            for (_, module), original in zip(model.targets, before):
                self.assertTrue(torch.equal(module.conv1d.weight, original))

    def test_dk32_all_targets_exact(self):
        model = Model(dk=32)
        before = [module.conv1d.weight.detach().clone() for _, module in model.targets]
        rows = runtime.prepare_model(model, Gdn, expected_dk=32, require_npu=False)
        self.assertEqual(len(rows), 48)
        self.assertTrue(all(row["shape"] == [1792, 1, 4] for row in rows))
        self.assertTrue(all(torch.equal(module.conv1d.weight, original)
                            for (_, module), original in zip(model.targets, before)))

    def test_runner_order_reload_rejection_and_idempotent_install(self):
        calls = []
        model = Model()
        class Runner:
            def load_model(self):
                calls.append("original_load_and_postprocess")
                return "loaded"
            def get_model(self):
                return model
        class NoopOffloader:
            pass
        noop = NoopOffloader()
        module = types.ModuleType("vllm.model_executor.offloader.base")
        module.NoopOffloader = NoopOffloader
        module.get_offloader = lambda: noop
        cfg = DrrqrConfig(conv_layout=True, evidence_file="/tmp/not-written")
        runtime.install_runner_patch(Runner, Model, Gdn, cfg, lambda: calls.append("verify"))
        first = Runner.load_model
        runtime.install_runner_patch(Runner, Model, Gdn, cfg, lambda: calls.append("must_not_replace"))
        self.assertIs(first, Runner.load_model)
        instance = Runner()
        instance.vllm_config = object()
        with (mock.patch.dict(sys.modules, {module.__name__: module}),
              mock.patch.object(runtime, "validate_config", return_value=64),
              mock.patch.object(runtime, "prepare_model", side_effect=lambda *a, **k: calls.append("prepare") or []),
              mock.patch.object(runtime, "emit_evidence", side_effect=lambda *a, **k: calls.append("evidence"))):
            self.assertEqual(instance.load_model(), "loaded")
            self.assertEqual(calls, ["original_load_and_postprocess", "verify", "prepare", "evidence"])
            with self.assertRaisesRegex(RuntimeError, "reload"):
                instance.load_model()

    def test_cpu_parameters_rejected_in_real_runtime_gate(self):
        with self.assertRaisesRegex(RuntimeError, "resident on NPU"):
            runtime.prepare_model(Model(), Gdn, expected_dk=64)


if __name__ == "__main__":
    unittest.main()
