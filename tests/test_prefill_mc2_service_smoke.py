import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("mc2_smoke", Path(__file__).resolve().parents[1] / "tools/prefill_mc2_service_smoke.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def events(enabled=True):
    result = []
    names = ["model.layers." + str(i) + "." + kind for i in range(64)
             for kind in ("mlp.down_proj", "self_attn.o_proj" if (i + 1) % 4 == 0 else "linear_attn.out_proj")]
    for pid in range(4):
        result += [{"event": "plugin_build_active", "pid": pid, "version": "0.1.8", "prefill_mc2": enabled},
                   {"event": "conv_weight_layout_prepared", "pid": pid,
                    "layers": [{"layer": i, "exact_values_verified": True} for i in range(64) if (i + 1) % 4]}]
        if enabled:
            result.append({"event": "prefill_mc2_prepared", "pid": pid, "weights_modified": False,
                           "source_sha256": "hash", "layers": [{"name": n} for n in names]})
            result.extend({"event": "prefill_mc2_called", "pid": pid, "name": n, "phase": "pure_prefill",
                           "rows": 32768, "source_sha256": "hash"} for n in names)
    return result


class MC2ServiceSmokeTests(unittest.TestCase):
    def test_enabled_and_control(self):
        self.assertEqual(smoke.validate_evidence(events(), True, "hash")["fused_calls"], 512)
        self.assertEqual(smoke.validate_evidence(events(False), False, "hash")["fused_calls"], 0)

    def test_missing_or_wrong_call(self):
        for change in ("missing", "duplicate", "phase", "hash"):
            values = events()
            if change == "missing":
                values.pop()
            elif change == "duplicate":
                values[-1] = copy.deepcopy(values[-2])
            elif change == "phase":
                values[-1]["phase"] = "decode"
            else:
                values[-1]["source_sha256"] = "wrong"
            with self.assertRaises(ValueError):
                smoke.validate_evidence(values, True, "hash")

    def test_control_rejects_treatment(self):
        with self.assertRaises(ValueError):
            smoke.validate_evidence(events(), False, "hash")
