import sys
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from mc2_coverage import install_observer, snapshot_context
from collect_mc2_coverage import summarize_records


def context(prefills=1, decodes=0, decode_tokens=0, actual=8192):
    value = NS(num_prefills=prefills, num_decodes=decodes,
               num_decode_tokens=decode_tokens, num_actual_tokens=actual)
    return NS(attn_metadata={"a": value, "b": value}, in_profile_run=False,
              capturing=False, cudagraph_runtime_mode=NS(name="NONE"))


class CoverageTests(unittest.TestCase):
    def test_context_phases_and_no_tensor_conversion(self):
        for c, phase, prompt in ((context(), "prefill", 8192),
                                 (context(decodes=2, decode_tokens=2), "mixed", 8190),
                                 (context(0, 16, 16, 16), "decode", 0)):
            value = snapshot_context(c, 8192, 0)
            self.assertEqual(value["phase"], phase)
            self.assertEqual(value["prefill_rows"], prompt)
            self.assertEqual(value["metadata"][0]["entries"], 2)
        c = context()
        c.attn_metadata["a"].num_prefills = object()
        self.assertEqual(snapshot_context(c, 8192, 0)["phase"], "unknown")
        self.assertIsNone(snapshot_context(c, 8192, 0)["actual_rows"])

    def setup_observer(self):
        bindings = {"down": (None, {"observed": False}), "out": (None, {"observed": False})}
        ordinary, records = [], []
        current = context()
        adapter = NS(emit_evidence=lambda *a, **k: ordinary.append((a, k)),
                     prefill_rows=lambda c, n: n if c.attn_metadata["a"].num_decodes == 0 else 0)
        class Runner:
            fail = False
            result = object()
            def _model_forward(self, num_tokens_padded):
                if self.fail:
                    raise RuntimeError("expected model failure")
                if adapter.prefill_rows(current, num_tokens_padded):
                    for name, (_, binding) in bindings.items():
                        if not binding["observed"]:
                            adapter.emit_evidence(None, "prefill_mc2_called", name=name,
                                                  k=4352 if name == "down" else 1536, rows=num_tokens_padded)
                            binding["observed"] = True
                return self.result
        runner = Runner()
        active = [False]
        install_observer(runner, adapter, bindings, lambda: current, lambda: active[0], records.append, max_steps=8)
        return runner, bindings, ordinary, records, current, active

    def test_return_identity_repeated_calls_and_once_only_evidence(self):
        runner, bindings, ordinary, records, current, active = self.setup_observer()
        self.assertIs(runner._model_forward(8192), runner.result)
        self.assertEqual(len(ordinary), 2)
        self.assertEqual(records, [])
        active[0] = True
        for _ in range(2):
            self.assertIs(runner._model_forward(8192), runner.result)
        self.assertEqual(len(ordinary), 2)
        self.assertEqual([r["fused_calls_by_k"] for r in records], [{"4352": 1, "1536": 1}] * 2)
        self.assertTrue(all(b["observed"] for _, b in bindings.values()))
        current.attn_metadata["a"].num_decodes = 1
        current.attn_metadata["a"].num_decode_tokens = 1
        self.assertIs(runner._model_forward(8192), runner.result)
        self.assertEqual(records[-1]["fused_calls_by_k"], {})
        self.assertEqual(records[-1]["phase"], "mixed")

    def test_exception_restores_state_and_does_not_claim_completion(self):
        runner, bindings, ordinary, records, current, active = self.setup_observer()
        runner._model_forward(8192)
        active[0], runner.fail = True, True
        with self.assertRaisesRegex(RuntimeError, "expected model failure"):
            runner._model_forward(8192)
        self.assertEqual(records[-1]["status"], "failed")
        self.assertTrue(all(b["observed"] for _, b in bindings.values()))
        runner.fail = False
        self.assertIs(runner._model_forward(8192), runner.result)
        self.assertEqual(records[-1]["status"], "complete")
        self.assertEqual(len(ordinary), 2)

    def test_coverage_aggregation_and_reject_inconsistent_counts(self):
        a = snapshot_context(context(), 8192, 8192)
        a.update(sequence=1, status="complete", fused_calls_by_k={"4352": 64, "1536": 64},
                 fused_row_calls_by_k={"4352": 8192 * 64, "1536": 8192 * 64})
        b = snapshot_context(context(decodes=2, decode_tokens=2), 8192, 0)
        b.update(sequence=2, status="complete", fused_calls_by_k={}, fused_row_calls_by_k={})
        result = summarize_records([a, b])
        self.assertEqual(result["observed_prefill_rows"], 16382)
        self.assertAlmostEqual(result["fused_row_fraction_of_observed_prefill_by_k"]["4352"], 8192 / 16382)
        b["fused_calls_by_k"] = {"4352": 1}
        with self.assertRaises(ValueError):
            summarize_records([a, b])
        with self.assertRaises(ValueError):
            summarize_records([a, a])


if __name__ == "__main__":
    unittest.main()
