import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("nz_probe", Path(__file__).resolve().parents[1] / "tools/nz_weight_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def records():
    return [dict(case, weight_roundtrip_exact=True, weight_formats={"nd": 2, "nz": 29},
                 numerical={name: {"finite": True, "relative_l2": 0.0, "normalized_max_abs": 0.0}
                            for name in ("nz_vs_nd_all_elements", "nd_vs_cpu_fp32_sample", "nz_vs_cpu_fp32_sample")},
                 graph_matches_eager={"nd": True, "nz": True} if case["phase"] == "decode" else {},
                 timing_ms_per_call={"nd": [1.0] * 4, "nz": [0.8] * 4})
            for case in probe.matrix()]


class NZProbeTests(unittest.TestCase):
    def test_matrix(self):
        self.assertEqual(len(probe.matrix()), 16)
        self.assertEqual(probe.validate(records()), "micro_screen_complete")

    def test_duplicate_missing(self):
        cases = records()
        cases[-1] = copy.deepcopy(cases[0])
        with self.assertRaises(ValueError):
            probe.validate(cases)

    def test_numeric_failure(self):
        for field, value in (("finite", False), ("relative_l2", 1.0), ("relative_l2", float("nan")),
                             ("normalized_max_abs", 1.0)):
            cases = records()
            cases[0]["numerical"]["nz_vs_nd_all_elements"][field] = value
            with self.assertRaises(ValueError):
                probe.validate(cases)

    def test_weight_graph_timing_failures(self):
        for key, value in (("weight_roundtrip_exact", False), ("weight_formats", {"nd": 2, "nz": 2}),
                           ("graph_matches_eager", {"nz": False}),
                           ("timing_ms_per_call", {"nd": [0.0] * 4, "nz": [0.8] * 4})):
            cases = records()
            cases[0][key] = value
            with self.assertRaises(ValueError):
                probe.validate(cases)
