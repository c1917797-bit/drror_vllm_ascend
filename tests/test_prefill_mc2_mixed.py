"""Change-specific routing tests for the explicit mixed-batch opt-in."""
import copy
import io
from contextlib import redirect_stderr
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from drror_vllm_ascend.envs import DrrqrConfig, get_config
from drror_vllm_ascend.patches import prefill_mc2 as mc2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_conv_layout_service import parse_args


def context(rows=8292, decodes=4):
    item = NS(num_prefills=2, num_decodes=decodes,
              num_decode_tokens=decodes, num_actual_tokens=rows)
    return NS(in_profile_run=False, capturing=False, cudagraph_runtime_mode=NS(name="NONE"),
              attn_metadata={"gdn": item, "full": copy.copy(item)})


class MixedMC2Tests(unittest.TestCase):
    def test_default_off_and_metadata_contract(self):
        c = context()
        self.assertEqual(mc2.prefill_rows(c, 8292), 0)
        self.assertEqual(mc2.prefill_rows(c, 8292, allow_mixed=True), 8292)
        for name, value in (("num_decodes", -1), ("num_decodes", 5),
                            ("num_decode_tokens", 8292), ("num_prefills", 0),
                            ("num_prefills", 8292), ("num_actual_tokens", 8291),
                            ("num_decodes", torch.tensor(4))):
            bad = context()
            setattr(bad.attn_metadata["full"], name, value)
            self.assertEqual(mc2.prefill_rows(bad, 8292, allow_mixed=True), 0)
        for key in ("in_profile_run", "capturing"):
            bad = context()
            setattr(bad, key, True)
            self.assertEqual(mc2.prefill_rows(bad, 8292, allow_mixed=True), 0)
        bad = context()
        bad.cudagraph_runtime_mode.name = "FULL"
        self.assertEqual(mc2.prefill_rows(bad, 8292, allow_mixed=True), 0)
        self.assertEqual(mc2.prefill_rows(context(103), 103, allow_mixed=True), 0)
        self.assertEqual(mc2.prefill_rows(context(8292, 0), 8292, allow_mixed=True), 8292)

    def test_mixed_runtime_dispatch_and_compiled_transition(self):
        class Linear:
            def __init__(self):
                self.weight = torch.randn((8, 4))
                self.return_bias = True
            def forward(self, x, **kwargs):
                return torch.nn.functional.linear(x, self.weight), None

        class Runner:
            def _model_forward(self, num_tokens_padded):
                if self.fail:
                    raise RuntimeError("model failure")
                return self.compiled(self.x)

        library = torch.library.Library("drrqr_mc2_mixed_cpu_test", "DEF")
        library.define("linear(Tensor x, Tensor weight, str key) -> Tensor")
        def make_opaque(dispatch):
            library.impl("linear", dispatch, "CPU")
            library.impl("linear", lambda x, w, key: x.new_empty((x.shape[0], w.shape[0])), "Meta")
            return torch.ops.drrqr_mc2_mixed_cpu_test.linear

        c = context(2048, 4)
        c.in_profile_run = True
        calls = []
        layer = Linear()
        config = DrrqrConfig(prefill_mc2=True, prefill_mc2_mixed=True)
        bindings = mc2.install_dispatch(
            Runner, Linear, config, get_context=lambda: c,
            fused=lambda x, w, h: calls.append(mc2._PREFILL_PHASE.get()) or (x @ w + 1),
            make_opaque=make_opaque)
        binding = {"name": "mixed", "k": 4, "min_rows": 2048, "hcom": "cpu",
                   "observed": False, "observed_mixed": False, "source_sha256": "unused"}
        layer._drrqr_prefill_mc2 = binding
        bindings["mixed"] = (layer, binding)
        runner = Runner()
        runner.fail = False
        runner.x = torch.randn((2048, 4))
        runner.compiled = torch.compile(lambda x: layer.forward(x)[0],
                                        backend="eager", fullgraph=True, dynamic=True)
        reference = runner.x @ layer.weight.t()
        with patch.object(mc2, "emit_evidence") as evidence:
            self.assertTrue(torch.equal(runner._model_forward(2048), reference))
            c.in_profile_run = False
            for _ in range(2):
                self.assertTrue(torch.equal(runner._model_forward(2048), reference + 1))
                self.assertEqual(mc2._PREFILL_ROWS.get(), 0)
                self.assertEqual(mc2._PREFILL_PHASE.get(), "none")
            self.assertEqual(calls, ["mixed_prefill", "mixed_prefill"])
            self.assertEqual([e.args[1] for e in evidence.call_args_list],
                             ["prefill_mc2_called", "prefill_mc2_mixed_called"])
            # Pure decode is still original even with many rows.
            for item in c.attn_metadata.values():
                item.num_prefills = 0
                item.num_decodes = item.num_decode_tokens = 2048
            self.assertTrue(torch.equal(runner._model_forward(2048), reference))
            # Direct call has no runner marker.
            self.assertTrue(torch.equal(layer.forward(runner.x)[0], reference))
            c = context(2048, 4)
            runner.x = torch.randn((2048, 5))
            with self.assertRaisesRegex(RuntimeError, "live input"):
                runner._model_forward(2048)
            self.assertEqual(mc2._PREFILL_ROWS.get(), 0)
            self.assertEqual(mc2._PREFILL_PHASE.get(), "none")
            runner.fail = True
            with self.assertRaisesRegex(RuntimeError, "model failure"):
                runner._model_forward(2048)
            self.assertEqual(mc2._PREFILL_ROWS.get(), 0)
            self.assertEqual(mc2._PREFILL_PHASE.get(), "none")
            self.assertEqual(len(calls), 2)
        with self.assertRaisesRegex(RuntimeError, "another configuration"):
            mc2.install_dispatch(Runner, Linear, DrrqrConfig(), get_context=lambda: c, fused=None)

    def test_opt_in_requires_main_flag(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(get_config().prefill_mc2_mixed)
        with patch.dict("os.environ", {"VLLM_ASCEND_DRRQR_PREFILL_MC2_MIXED": "1"}, clear=True):
            with self.assertRaisesRegex(ValueError, "requires"):
                get_config()
        with patch.dict("os.environ", {"VLLM_ASCEND_DRRQR_PREFILL_MC2": "1",
                                     "VLLM_ASCEND_DRRQR_PREFILL_MC2_MIXED": "1",
                                     "VLLM_ASCEND_DRRQR_EVIDENCE_FILE": "/tmp/evidence"}, clear=True):
            self.assertTrue(get_config().prefill_mc2_mixed)

    def test_launcher_version_and_diagnostic_boundaries(self):
        base = ["--variant", "dk64", "--layout", "packed", "--build-manifest", "/unused",
                "--output-dir", "/drrqr-results/unused"]
        valid = ["--plugin-version", "0.1.11", "--prefill-mc2", "--prefill-mc2-mixed"]
        self.assertTrue(parse_args(base + valid)[1].prefill_mc2_mixed)
        self.assertFalse(parse_args(base + ["--plugin-version", "0.1.11"])[1].prefill_mc2_mixed)
        for flags in (["--prefill-mc2-mixed"],
                      ["--plugin-version", "0.1.8", "--prefill-mc2", "--prefill-mc2-mixed"],
                      valid + ["--diagnostic-mc2-coverage"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(base + flags)
        dk32 = ["--variant", "dk32", "--layout", "packed",
                "--build-manifest", "/unused", "--output-dir", "/drrqr-results/unused",
                "--plan-path", "/drrqr-results/dk32.json", "--plan-sha", "a" * 64]
        parsed = parse_args(dk32 + valid)[1]
        self.assertEqual(parsed.variant, "dk32")
        self.assertEqual(parsed.plan_path, Path("/drrqr-results/dk32.json"))
        self.assertEqual(parsed.plan_sha, "a" * 64)
