"""Coverage guards for the fixed layout equivalence protocol."""
import copy
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    "layout_comparison_under_test",
    Path(__file__).resolve().parents[1] / "tools" / "compare_layout_diagnostics.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LayoutComparisonTests(unittest.TestCase):
    def setUp(self):
        cases = [{"id": f"input2048-{i}", "input_tokens": 2048, "output_tokens": 128,
                  "concurrency": 16} for i in range(16)]
        cases += [{"id": f"input32768-{i}", "input_tokens": 32768, "output_tokens": 128,
                   "concurrency": 1} for i in range(2)]
        rows = [{"id": c["id"], "token_ids": [1] * 128} for c in cases]
        self.probe = {"cases": cases, "rounds": [copy.deepcopy(rows), copy.deepcopy(rows)]}

    def test_valid_and_empty_coverage(self):
        self.assertEqual(len(MODULE.index_rounds(self.probe)[0]), 18)
        with self.assertRaises(ValueError):
            MODULE.index_rounds({"cases": [], "rounds": [[], []]})

    def test_missing_duplicate_and_wrong_length_rejected(self):
        for mutation in ("missing", "duplicate", "short", "boolean"):
            probe = copy.deepcopy(self.probe)
            if mutation == "missing":
                probe["rounds"][0].pop()
            elif mutation == "duplicate":
                probe["rounds"][0][-1] = probe["rounds"][0][0]
            elif mutation == "short":
                probe["rounds"][0][0]["token_ids"].pop()
            else:
                probe["rounds"][0][0]["token_ids"][0] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                MODULE.index_rounds(probe)

    def test_command_only_normalizes_profile_artifact_path(self):
        a = ["serve", "--profiler-config", '{"torch_profiler_dir": "/a", "max_iterations": 30}']
        b = ["serve", "--profiler-config", '{"torch_profiler_dir": "/b", "max_iterations": 30}']
        c = ["serve", "--profiler-config", '{"torch_profiler_dir": "/b", "max_iterations": 31}']
        self.assertEqual(MODULE.command_key(a), MODULE.command_key(b))
        self.assertNotEqual(MODULE.command_key(a), MODULE.command_key(c))


if __name__ == "__main__":
    unittest.main()

