import json

import pytest

from tools.analyze_layerwise_energy_budget import analyze


def _write_plan(path, dim, values, *, capture_hash="a" * 64):
    layers = {}
    for layer, value in enumerate(values):
        layers[str(layer)] = {
            "ranks": {
                "0": {
                    "rrqr_by_local_head": [
                        {"retained_energy_fraction": value},
                    ],
                },
            },
        }
    path.write_text(json.dumps({
        "schema": "ascend-drrqr-plan/v1",
        "old_head_k_dim": 128,
        "target_head_k_dim": dim,
        "source_config_sha256": "b" * 64,
        "source_index_sha256": "c" * 64,
        "layer_types": ["linear_attention"] * len(values),
        "provenance": {
            "selection": {
                "selection_objective": "energy-kernel",
                "capture_manifest_sha256": capture_hash,
                "layers": layers,
            },
        },
    }))


def test_optimizer_finds_better_exact_budget_assignment(tmp_path):
    dk32 = tmp_path / "dk32.json"
    dk64 = tmp_path / "dk64.json"
    # One Dk128 layer must be balanced by two Dk32 layers to preserve an
    # average Dk64 budget: 128 + 32 + 32 == 3 * 64.
    _write_plan(dk32, 32, [0.20, 0.98, 0.98])
    _write_plan(dk64, 64, [0.50, 0.99, 0.99])

    report = analyze(dk32, dk64, prototype_gate_pp=1.0)

    assert report["uniform64_mean_retained_energy_fraction"] == pytest.approx(
        (0.50 + 0.99 + 0.99) / 3
    )
    assert report["optimized_mean_retained_energy_fraction"] == pytest.approx(
        (1.0 + 0.98 + 0.98) / 3
    )
    assert report["assignment_counts"] == {"32": 2, "64": 0, "128": 1}
    assert report["decision"] == "proxy_promising_runtime_feasibility_still_required"


def test_incompatible_capture_binding_is_rejected(tmp_path):
    dk32 = tmp_path / "dk32.json"
    dk64 = tmp_path / "dk64.json"
    _write_plan(dk32, 32, [0.50], capture_hash="a" * 64)
    _write_plan(dk64, 64, [0.75], capture_hash="d" * 64)

    with pytest.raises(ValueError, match="capture_manifest_sha256"):
        analyze(dk32, dk64)


def test_small_proxy_gain_rejects_runtime_prototype(tmp_path):
    dk32 = tmp_path / "dk32.json"
    dk64 = tmp_path / "dk64.json"
    _write_plan(dk32, 32, [0.70, 0.70])
    _write_plan(dk64, 64, [0.90, 0.90])

    report = analyze(dk32, dk64, prototype_gate_pp=1.0)

    assert report["optimized_gain_over_uniform64_pp"] == pytest.approx(0.0)
    assert report["assignment_counts"] == {"32": 0, "64": 2, "128": 0}
    assert report["decision"] == "reject_mixed_dimension_runtime_prototype"
