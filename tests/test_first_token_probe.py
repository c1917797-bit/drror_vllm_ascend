"""First-token diagnostic response validation."""
import copy
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    "first_token_probe_under_test",
    Path(__file__).resolve().parents[1] / "tools" / "collect_first_token_probe.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FirstTokenProbeTests(unittest.TestCase):
    def setUp(self):
        self.data = {"usage": {"prompt_tokens": 2048, "completion_tokens": 1},
                     "choices": [{"index": 0, "finish_reason": "length", "token_ids": [42],
                                  "logprobs": {"top_logprobs": [{"answer": -0.1}],
                                               "token_logprobs": [-0.1]}}]}

    def test_valid_response(self):
        self.assertEqual(MODULE.validate_response(self.data, 2048)["token_id"], 42)

    def test_invalid_lengths_or_token_type(self):
        for mutation in ("input", "output", "tokens", "boolean"):
            data = copy.deepcopy(self.data)
            if mutation == "input":
                data["usage"]["prompt_tokens"] = 2047
            elif mutation == "output":
                data["usage"]["completion_tokens"] = 2
            elif mutation == "tokens":
                data["choices"][0]["token_ids"] = []
            else:
                data["choices"][0]["token_ids"] = [True]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                MODULE.validate_response(data, 2048)

    def test_missing_and_nonfinite_probabilities(self):
        for value in (None, {"top_logprobs": [{"answer": float("nan")}],
                            "token_logprobs": [-0.1]}):
            data = copy.deepcopy(self.data)
            data["choices"][0]["logprobs"] = value
            with self.assertRaises(ValueError):
                MODULE.validate_response(data, 2048)


if __name__ == "__main__":
    unittest.main()

