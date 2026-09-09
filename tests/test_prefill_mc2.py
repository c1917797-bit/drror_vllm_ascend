import copy
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from drror_vllm_ascend.envs import DrrqrConfig, get_config
from drror_vllm_ascend.patches import prefill_mc2 as mc2


def context(rows=8192, **changes):
    meta = NS(num_prefills=1, num_decodes=0, num_decode_tokens=0, num_actual_tokens=rows)
    value = NS(in_profile_run=False, capturing=False, cudagraph_runtime_mode=NS(name="NONE"),
               attn_metadata={"a": meta})
    for key, val in changes.items():
        setattr(value, key, val)
    return value


class Row:
    def __init__(self, k=1536):
        self.weight = torch.empty((5120, k), dtype=torch.bfloat16, device="meta")
        self.tp_size = 4
        self.input_is_parallel = self.reduce_results = self.return_bias = True
        self.bias = self.custom_op = self.out_dtype = None
        self.original_calls = 0

    def forward(self, input_, **kwargs):
        self.original_calls += 1
        return "original", kwargs


def model_rows():
    entries = []
    for index in range(64):
        entries.append(("model.layers." + str(index) + ".mlp.down_proj", Row(4352)))
        kind = ".self_attn.o_proj" if (index + 1) % 4 == 0 else ".linear_attn.out_proj"
        entries.append(("model.layers." + str(index) + kind, Row()))
    return entries


class PrefillMC2Tests(unittest.TestCase):
    def test_pure_prefill_only(self):
        self.assertEqual(mc2.prefill_rows(context(), 8192), 8192)
        for changed in (context(in_profile_run=True), context(capturing=True),
                        context(cudagraph_runtime_mode=NS(name="FULL")),
                        context(attn_metadata={}), context(attn_metadata=[]), context(8191)):
            self.assertEqual(mc2.prefill_rows(changed, 8192), 0)
        for key, value in (("num_decodes", 1), ("num_decode_tokens", 1), ("num_prefills", 0),
                           ("num_actual_tokens", None), ("num_prefills", torch.tensor(1))):
            c = context()
            setattr(c.attn_metadata["a"], key, value)
            self.assertEqual(mc2.prefill_rows(c, 8192), 0)
        c = context()
        c.attn_metadata["b"] = copy.copy(c.attn_metadata["a"])
        c.attn_metadata["b"].num_decodes = 1
        self.assertEqual(mc2.prefill_rows(c, 8192), 0)

    def test_all_target_coverage_and_no_mutation(self):
        rows = model_rows()
        model = NS(named_modules=lambda: rows)
        targets = mc2.collect_targets(model, Row, require_npu=False)
        self.assertEqual(len(targets), 128)
        self.assertEqual(sum(item[3] == 2048 for item in targets), 64)
        rows.pop()
        with self.assertRaises(RuntimeError):
            mc2.collect_targets(model, Row, require_npu=False)
        self.assertFalse(any(hasattr(layer, "_drrqr_prefill_mc2") for _, layer in rows))

    def test_unsupported_target(self):
        for field, value in (("tp_size", 2), ("reduce_results", False), ("input_is_parallel", False),
                             ("custom_op", object()), ("bias", object()), ("out_dtype", torch.float32)):
            rows = model_rows()
            setattr(rows[0][1], field, value)
            with self.assertRaises(RuntimeError):
                mc2.collect_targets(NS(named_modules=lambda: rows), Row, require_npu=False)

    def test_dispatch_fallback_and_context_reset(self):
        class Linear(Row):
            pass
        class Runner:
            def _model_forward(self, num_tokens_padded, input_ids=None):
                if self.fail:
                    raise RuntimeError("model failure")
                return self.layer.forward(self.x)

        current = context()
        calls = []
        config = DrrqrConfig(evidence_file="/tmp/unused-evidence")
        bindings = mc2.install_dispatch(Runner, Linear, config, get_context=lambda: current,
                                        fused=lambda x, w, h: calls.append((x.shape, w.shape, h)) or "fused")
        layer = Linear()
        layer._drrqr_prefill_mc2 = {"name": "layer", "k": 1536, "min_rows": 8192,
                                   "hcom": "test-group", "observed": False, "source_sha256": "unused"}
        bindings["layer"] = (layer, layer._drrqr_prefill_mc2)
        runner = Runner()
        runner.layer, runner.fail = layer, False
        runner.x = torch.empty((8192, 1536), dtype=torch.bfloat16, device="meta")
        with patch.object(mc2, "emit_evidence") as evidence:
            self.assertEqual(runner._model_forward(8192), ("fused", None))
            self.assertEqual(len(calls), 1)
            self.assertEqual(evidence.call_count, 1)
            self.assertEqual(mc2._PREFILL_ROWS.get(), 0)
            # Mixed metadata and direct calls outside the runner preserve exact original route.
            current.attn_metadata["a"].num_decodes = 1
            self.assertEqual(runner._model_forward(8192)[0], "original")
            self.assertEqual(layer.forward(runner.x)[0], "original")
            self.assertEqual(len(calls), 1)
            runner.x = torch.empty((16, 1536), dtype=torch.bfloat16, device="meta")
            self.assertEqual(runner._model_forward(16)[0], "original")
            runner.fail = True
            with self.assertRaises(RuntimeError):
                runner._model_forward(8192)
            self.assertEqual(mc2._PREFILL_ROWS.get(), 0)

    def test_compiled_phase_is_not_frozen_from_dummy_batch(self):
        class Linear:
            def __init__(self):
                self.weight = torch.randn((8, 4))
                self.return_bias = True

            def forward(self, x, **kwargs):
                return torch.nn.functional.linear(x, self.weight), None

        class Runner:
            def _model_forward(self, num_tokens_padded):
                return self.compiled(self.x)

        library = torch.library.Library("drrqr_mc2_cpu_test", "DEF")
        library.define("linear(Tensor x, Tensor weight, str key) -> Tensor")
        def make_opaque(dispatch):
            library.impl("linear", dispatch, "CPU")
            library.impl("linear", lambda x, w, key: x.new_empty((x.shape[0], w.shape[0])), "Meta")
            return torch.ops.drrqr_mc2_cpu_test.linear

        current = context(2048, in_profile_run=True)
        calls = []
        layer = Linear()
        bindings = mc2.install_dispatch(
            Runner, Linear, DrrqrConfig(), get_context=lambda: current,
            fused=lambda x, w, h: calls.append("fused") or (x @ w + 1),
            make_opaque=make_opaque,
        )
        layer._drrqr_prefill_mc2 = {"name": "cpu-test", "k": 4, "min_rows": 2048,
                                   "hcom": "cpu", "observed": True, "source_sha256": "unused"}
        bindings["cpu-test"] = (layer, layer._drrqr_prefill_mc2)
        runner = Runner()
        runner.x = torch.randn((2048, 4))
        runner.compiled = torch.compile(lambda x: layer.forward(x)[0], backend="eager", fullgraph=True, dynamic=True)
        reference = runner.x @ layer.weight.t()
        self.assertTrue(torch.equal(runner._model_forward(2048), reference))
        self.assertEqual(calls, [])
        current.in_profile_run = False
        self.assertTrue(torch.equal(runner._model_forward(2048), reference + 1))
        self.assertEqual(calls, ["fused"])
        current.attn_metadata["a"].num_decodes = 1
        self.assertTrue(torch.equal(runner._model_forward(2048), reference))
        self.assertEqual(calls, ["fused"])


    def test_opt_in_and_capture_conflict(self):
        with patch.dict("os.environ", {"VLLM_ASCEND_DRRQR_PREFILL_MC2": "1"}, clear=True):
            with self.assertRaises(ValueError):
                get_config()
        with patch.dict("os.environ", {"VLLM_ASCEND_DRRQR_PREFILL_MC2": "1",
                                      "VLLM_ASCEND_DRRQR_EVIDENCE_FILE": "/tmp/evidence"}, clear=True):
            self.assertTrue(get_config().prefill_mc2)
        with patch.dict("os.environ", {"VLLM_ASCEND_DRRQR_PREFILL_MC2": "1",
                                      "VLLM_ASCEND_DRRQR_CAPTURE_ENABLE": "1"}, clear=True):
            with self.assertRaises(ValueError):
                get_config()
