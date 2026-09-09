import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from conv_layout_graph_probe import BATCHES, DKS, STEPS, validate


def records():
    fields = ("graph_original_output", "graph_packed_output", "eager_packed_output",
              "graph_original_state", "graph_packed_state", "eager_packed_state",
              "storage_pointers_unchanged")
    return [{"dk": dk, "batch": batch,
             "steps": [{"step": step, **dict.fromkeys(fields, True)} for step in range(STEPS)]}
            for dk in DKS for batch in BATCHES]


class GraphReportTests(unittest.TestCase):
    def test_complete_matrix(self):
        self.assertEqual(validate(records()), "graph_operator_parity_pass")

    def test_missing_duplicate_cases(self):
        for invalid in ([], records()[:-1], records()[:-1] + [records()[0]]):
            with self.assertRaises(ValueError):
                validate(invalid)

    def test_incomplete_step_sequence(self):
        value = records()
        value[0]["steps"].pop()
        with self.assertRaises(ValueError):
            validate(value)
        value = records()
        value[0]["steps"][1]["step"] = 0
        with self.assertRaises(ValueError):
            validate(value)

    def test_each_numerical_or_storage_failure_rejected(self):
        value = records()
        for field in value[0]["steps"][0]:
            if field == "step":
                continue
            modified = copy.deepcopy(value)
            modified[0]["steps"][0][field] = False
            with self.assertRaises(ValueError):
                validate(modified)

