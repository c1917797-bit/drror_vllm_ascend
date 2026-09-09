"""CPU-only validation of diagnostic comparison coverage."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    "layout_isolation_under_test",
    Path(__file__).resolve().parents[1] / "tools" / "collect_layout_isolation.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LayoutIsolationSummaryTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"id": "a", "token_ids": [1] * 128},
                     {"id": "b", "token_ids": [2] * 128}]

    def test_stability_and_first_difference(self):
        self.assertEqual(MODULE.summarize([self.rows, self.rows])["stable_cases"], 2)
        changed = [self.rows[0], {"id": "b", "token_ids": [2] * 127 + [3]}]
        summary = MODULE.summarize([self.rows, changed])
        self.assertEqual(summary["stable_cases"], 1)
        self.assertEqual(summary["cases"][1]["first_different_position"], 127)

    def test_rejects_changed_coverage(self):
        with self.assertRaises(ValueError):
            MODULE.summarize([self.rows, self.rows[::-1]])
        duplicate = [self.rows[0], self.rows[0]]
        with self.assertRaises(ValueError):
            MODULE.summarize([duplicate, duplicate])

    def test_rejects_incomplete_output(self):
        with self.assertRaises(ValueError):
            MODULE.summarize([[{"id": "a", "token_ids": [1]}]])


if __name__ == "__main__":
    unittest.main()

