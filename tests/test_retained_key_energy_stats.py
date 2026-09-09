"""Software checks for retained-energy statistics, not NPU qualification."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
spec = importlib.util.spec_from_file_location("retained_stats", Path(__file__).resolve().parents[1] / "tools/retained_key_energy_stats.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_equal_energy_retention():
    values = torch.ones(7, 4, 128)
    keep = torch.arange(64).repeat(4, 1)
    assert torch.equal(module.fractions(values, keep), torch.full((7, 4), .5, dtype=torch.float64))


def test_head_mapping():
    values = torch.ones(2, 4, 128)
    values[:, 1, :64] = 2
    keep = torch.arange(64).repeat(4, 1)
    result = module.fractions(values, keep)
    assert torch.allclose(result[:, 1], torch.full((2,), .8, dtype=torch.float64))
    assert result[0, 0] == .5


@pytest.mark.parametrize("kind", ["nonfinite", "zero", "duplicate", "negative"])
def test_invalid_inputs(kind):
    values = torch.ones(1, 4, 128)
    keep = torch.arange(64).repeat(4, 1)
    if kind == "nonfinite":
        values[0, 0, 0] = float("nan")
    elif kind == "zero":
        values[0, 0] = 0
    elif kind == "duplicate":
        keep[0, 1] = 0
    else:
        keep[0, 0] = -1
    with pytest.raises(ValueError):
        module.fractions(values, keep)


def test_describe_constant():
    result = module.describe(torch.full((8,), .75))
    assert result["count"] == 8 and result["mean"] == .75
    assert result["cv"] == 0 and result["p95"] == .75


def test_describe_empty():
    with pytest.raises(ValueError):
        module.describe(torch.empty(0))
