import copy
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import matmul_allreduce_probe as probe


def ranks(rows=None):
    result = []
    for rank in range(4):
        cases = [dict(c, weight_format=2,
                      numerical={name: {"finite": True, "relative_l2": 0.0, "normalized_max_abs": 0.0}
                                 for name in probe.NUMERIC_NAMES},
                      device_ms={"separate": [1.0 + rank] * 4, "fused": [0.5 + rank] * 4},
                      wall_ms={"separate": [1.1 + rank] * 4, "fused": [0.6 + rank] * 4})
                 for c in probe.matrix(rows)]
        result.append({"rank": rank, "status": "rank_complete", "cases": cases})
    return result


class MatmulAllreduceProbeTests(unittest.TestCase):
    def test_observed_rows_do_not_change_default_matrix(self):
        shape_rows = [32769, 40960]
        values = ranks(shape_rows)
        result = probe.summarize(values, shape_rows, "observed-mixed-linear")
        self.assertEqual(len(result), 4)
        self.assertEqual({v["m"] for v in result}, set(shape_rows))
        self.assertTrue(all(v["phase"] == "observed-mixed-linear" for v in result))
        self.assertEqual(len(probe.matrix()), 6)
        with self.assertRaises(ValueError):
            probe.summarize(values)

    def test_complete_and_slowest_rank(self):
        rows = probe.summarize(ranks())
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]["device_ms_median_max_rank"], {"separate": 4.0, "fused": 3.5})

    def test_missing_duplicate_rank(self):
        values = ranks()
        values[-1] = copy.deepcopy(values[0])
        with self.assertRaises(ValueError):
            probe.summarize(values)

    def test_case_coverage(self):
        values = ranks()
        values[1]["cases"][-1] = copy.deepcopy(values[1]["cases"][0])
        with self.assertRaises(ValueError):
            probe.summarize(values)

    def test_numeric_and_timing_failures(self):
        for change in ("nan", "threshold", "missing_metric", "timing", "format"):
            values = ranks()
            case = values[3]["cases"][0]
            if change == "nan":
                case["numerical"]["fused_vs_separate_all_elements"]["relative_l2"] = float("nan")
            elif change == "threshold":
                case["numerical"]["fused_vs_separate_all_elements"]["relative_l2"] = 0.1
            elif change == "missing_metric":
                case["numerical"] = {}
            elif change == "format":
                case["weight_format"] = 29
            else:
                case["wall_ms"]["fused"] = [1.0]
            with self.assertRaises(ValueError):
                probe.summarize(values)
