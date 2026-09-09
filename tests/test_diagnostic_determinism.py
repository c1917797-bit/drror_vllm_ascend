"""No operator substitution and before-communicator ordering for deterministic diagnostics."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

SPEC = importlib.util.spec_from_file_location(
    "diagnostic_determinism_under_test",
    Path(__file__).resolve().parents[1] / "tools" / "diagnostic_determinism.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DiagnosticDeterminismTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.active = False
        def enable(value, *, warn_only):
            self.calls.append((value, warn_only))
            self.active = value
        self.torch = SimpleNamespace(
            distributed=SimpleNamespace(is_initialized=lambda: False),
            use_deterministic_algorithms=enable,
            are_deterministic_algorithms_enabled=lambda: self.active)
        self.env = {"DRRQR_DIAGNOSTIC_DETERMINISM": "1", "VLLM_BATCH_INVARIANT": "0",
                    "HCCL_DETERMINISTIC": "strict", "LCCL_DETERMINISTIC": "1"}

    def test_explicit_settings_preserve_operator_path(self):
        result = MODULE.apply_settings(self.torch, self.env)
        self.assertEqual(self.calls, [(True, True)])
        self.assertFalse(result["operator_substitutions"])

    def test_missing_opt_in_or_collective_setting_rejected(self):
        for key in ("DRRQR_DIAGNOSTIC_DETERMINISM", "HCCL_DETERMINISTIC", "LCCL_DETERMINISTIC"):
            env = dict(self.env)
            del env[key]
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                MODULE.apply_settings(self.torch, env)
        self.assertEqual(self.calls, [])

    def test_general_batch_invariant_mode_rejected(self):
        self.env["VLLM_BATCH_INVARIANT"] = "1"
        with self.assertRaises(RuntimeError):
            MODULE.apply_settings(self.torch, self.env)
        self.assertEqual(self.calls, [])

    def test_already_initialized_communicator_rejected(self):
        self.torch.distributed.is_initialized = lambda: True
        with self.assertRaises(RuntimeError):
            MODULE.apply_settings(self.torch, self.env)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()

