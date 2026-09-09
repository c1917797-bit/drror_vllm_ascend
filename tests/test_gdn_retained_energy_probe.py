"""CPU-only protocol and algebra tests; no NPU workload or model inference."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[1] / "tools/gdn_retained_energy_probe.py"
_spec = importlib.util.spec_from_file_location("retained_energy_probe_under_test", SOURCE)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


def test_fixed_protocol_and_golden_reference():
    assert (probe.D_K, probe.H_QK, probe.H_V, probe.D_V) == (64, 4, 12, 128)
    assert probe.BATCHES == (1, 16) and probe.STEPS == 8
    assert probe.DECODE_RTOL == 3e-3 and probe.DECODE_ATOL == 1e-2
    assert probe.RELATIVE_L2_LIMIT == 2 ** -7
    assert probe.NORMALIZED_MAX_LIMIT == 2 ** -6
    assert Path(probe.decode_golden.__code__.co_filename).name == "gdn_golden_parity.py"
    assert probe.checksum(probe.GOLDEN_SOURCE) == probe.GOLDEN_SHA256


@pytest.mark.parametrize("batch", (1, 16))
def test_unique_rotating_indices_and_actual_length_protocol(batch):
    generator = torch.Generator().manual_seed(19)
    previous = None
    for step in range(8):
        data = probe.cpu_inputs(batch, step, generator, probe.R_CASES["retained_key"])
        indices = data["indices"]
        assert indices.dtype == torch.int32
        assert len(set(indices.tolist())) == batch
        assert bool((indices >= 0).all() and (indices < batch + 3).all())
        assert data["lengths"].tolist() == [0] + [1] * batch
        assert data["r"].tolist() == [0.5] * 3 + [0.75] * 3 + [0.875] * 3 + [1.0] * 3
        assert data["r"].dtype == data["g"].dtype == torch.float32
        assert all(data[key].dtype == torch.bfloat16 for key in ("q", "k", "v", "beta_base"))
        assert tuple(data["q"].shape) == (batch, 4, 64)
        assert tuple(data["v"].shape) == (batch, 12, 128)
        if previous is not None:
            assert not torch.equal(data["beta_base"], previous)
        previous = data["beta_base"]


@pytest.mark.parametrize("bad", ([1, 1, 1], [0, 1, 1, 1], [1.01, 1, 1, 1],
                                  [float("nan"), 1, 1, 1], [float("inf"), 1, 1, 1]))
def test_invalid_retained_energy_rejected(bad):
    with pytest.raises(ValueError):
        probe.value_head_energy(bad)


def test_beta_conversion_and_same_operation_passed_to_native(monkeypatch):
    beta = torch.linspace(0.01, 0.99, 12).reshape(1, 12).to(torch.bfloat16)
    r = probe.value_head_energy(probe.R_CASES["retained_key"])
    data = {"beta_base": beta, "r": r}
    state = object()
    observed = {}

    def native(inputs, actual_state, actual_beta):
        observed.update(inputs=inputs, state=actual_state, beta=actual_beta)
        return torch.tensor([4.0])

    monkeypatch.setattr(probe, "native_recurrent", native)
    output, product, effective = probe.compensated_operation(data, state)
    assert product.dtype == torch.float32 and effective.dtype == torch.bfloat16
    assert torch.equal(product, beta.float() * r)
    assert probe.exact_finite(effective, (beta.float() * r).to(torch.bfloat16))
    assert observed["inputs"] is data and observed["state"] is state
    assert observed["beta"] is effective and output.item() == 4.0
    _, identity = probe.quantized_beta(beta, probe.value_head_energy((1, 1, 1, 1)))
    assert probe.exact_finite(identity, beta)


@pytest.mark.parametrize("kind", ("beta_dtype", "r_dtype", "r_shape"))
def test_beta_dtype_and_head_shape_fail_closed(kind):
    beta = torch.ones((1, 12), dtype=torch.bfloat16)
    r = torch.ones(12, dtype=torch.float32)
    if kind == "beta_dtype":
        beta = beta.float()
    elif kind == "r_dtype":
        r = r.to(torch.bfloat16)
    else:
        r = r.reshape(1, 12)
    with pytest.raises(ValueError):
        probe.quantized_beta(beta, r)


def test_finite_bytes_distinguish_signed_zero_dtype_shape():
    plus = torch.tensor([0.0])
    minus = torch.tensor([-0.0])
    assert not probe.exact_finite(plus, minus)
    assert not probe.exact_finite(plus, plus.to(torch.bfloat16))
    assert not probe.exact_finite(plus, plus.reshape(1, 1))
    for bad in (float("nan"), float("inf"), -float("inf")):
        tensor = torch.tensor([bad])
        assert not probe.exact_finite(tensor, tensor)
        assert not probe.error_metrics(tensor, tensor)["passed"]


def test_metrics_have_independent_declared_gates():
    zeros = torch.zeros(2)
    result = probe.error_metrics(zeros, zeros)
    assert result["passed"] and result["relative_l2"] == result["normalized_max"] == 0.0
    reference = torch.full((10000,), 0.1)
    actual = reference.clone()
    actual[0] += 0.008
    result = probe.error_metrics(actual, reference)
    assert result["allclose"]
    assert result["relative_l2"] < probe.RELATIVE_L2_LIMIT
    assert result["normalized_max"] > probe.NORMALIZED_MAX_LIMIT
    assert not result["passed"]
    assert not probe.error_metrics(actual.to(torch.bfloat16), reference)["passed"]


def test_unselected_slots_detect_changes_and_duplicates():
    before = torch.randn(4, 1, 2, 1)
    after = before.clone()
    indices = torch.tensor([1], dtype=torch.int32)
    after[1].add_(1)
    assert probe.unselected_unchanged(before, after, indices)
    after[3].add_(1)
    assert not probe.unselected_unchanged(before, after, indices)
    with pytest.raises(ValueError):
        probe.unselected_unchanged(before, before, torch.tensor([1, 1]))


def test_reused_golden_respects_indirect_state_slots():
    state = torch.zeros((4, 1, 2, 1), dtype=torch.float32)
    qk = torch.ones((1, 1, 1), dtype=torch.bfloat16)
    value = torch.tensor([[[1.0, 2.0]]], dtype=torch.bfloat16)
    beta = torch.tensor([[0.5]], dtype=torch.bfloat16)
    indices = torch.tensor([2], dtype=torch.int32)
    out, updated = probe.decode_golden(
        qk, qk, value, state, beta, 1.0, torch.ones(1, dtype=torch.int32),
        indices, torch.zeros((1, 1)))
    assert out.tolist() == [[[0.5, 1.0]]]
    assert updated[2, 0, :, 0].tolist() == [0.5, 1.0]
    assert probe.unselected_unchanged(state, updated, indices)


def projected_step(state, key, value, alpha, beta):
    decayed = alpha * state
    return decayed + beta * (value - decayed @ key).unsqueeze(-1) * key.unsqueeze(0)


@pytest.mark.parametrize("rk", (0.5, 0.75, 0.875, 1.0))
def test_constant_r_state_and_query_output_algebra(rk):
    generator = torch.Generator().manual_seed(12)
    c, rq = rk ** 0.5, 0.6
    state = torch.randn((3, 2), generator=generator, dtype=torch.float64)
    transformed = c * state
    for _ in range(8):
        key = torch.randn(2, generator=generator, dtype=torch.float64)
        key = key / key.norm()
        query = torch.randn(2, generator=generator, dtype=torch.float64)
        query = query / query.norm()
        value = torch.randn(3, generator=generator, dtype=torch.float64)
        state = projected_step(state, c * key, value, 0.8, 0.5)
        transformed = projected_step(transformed, key, value, 0.8, rk * 0.5)
        torch.testing.assert_close(transformed, c * state, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            state @ (rq ** 0.5 * query),
            (rq / rk) ** 0.5 * (transformed @ query), rtol=1e-12, atol=1e-12)


def test_token_varying_r_counterexample_not_removed_by_rmsnorm():
    state = torch.zeros((2, 1), dtype=torch.float64)
    naive = state.clone()
    key = torch.ones(1, dtype=torch.float64)
    for r, value in ((0.25, (1.0, 0.0)), (1.0, (0.0, 1.0))):
        v = torch.tensor(value, dtype=torch.float64)
        state = projected_step(state, r ** 0.5 * key, v, 1.0, 0.5)
        naive = projected_step(naive, key, v, 1.0, r * 0.5)
    assert state[:, 0].tolist() == [0.125, 0.5]
    assert naive[:, 0].tolist() == [0.0625, 0.5]
    norm = lambda x: x / x.square().mean().sqrt()
    assert not torch.allclose(norm(state), norm(naive))


def test_rmsnorm_epsilon_prevents_exact_scale_invariance():
    x = torch.tensor([1e-4, 2e-4], dtype=torch.float64)
    rms = lambda v, eps: v / (v.square().mean() + eps).sqrt()
    torch.testing.assert_close(rms(x, 0.0), rms(0.1 * x, 0.0))
    assert not torch.allclose(rms(x, 1e-6), rms(0.1 * x, 1e-6))


def test_fresh_output_rejects_old_directory_and_symlink(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "RESULTS_ROOT", tmp_path)
    output = probe.fresh_output(tmp_path / "fresh")
    (output / "old").write_text("preserve")
    with pytest.raises(ValueError):
        probe.fresh_output(output)
    alias = tmp_path / "alias"
    alias.symlink_to(output, target_is_directory=True)
    with pytest.raises(ValueError):
        probe.fresh_output(alias / "child")
    assert (output / "old").read_text() == "preserve"


def test_initialization_failure_is_saved_without_native_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "RESULTS_ROOT", tmp_path)
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0")
    for flag in probe.DISABLED_FLAGS:
        monkeypatch.setenv(flag, "1")
    monkeypatch.setenv("VLLM_PLUGINS", "drror_vllm_ascend")
    called = []

    def fail(report, phase):
        called.append("initialize")
        assert all(probe.os.environ[flag] == "0" for flag in probe.DISABLED_FLAGS)
        assert probe.os.environ["VLLM_PLUGINS"] == ""
        raise RuntimeError("intentional CPU-only initialization failure")

    monkeypatch.setattr(probe, "initialize_npu", fail)
    monkeypatch.setattr(probe, "loaded_libraries", lambda: [])
    out = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="intentional CPU-only"):
        probe.main(["--output-dir", str(out)])
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "failed" and report["cases"] == []
    assert "not real" in report["limits"].lower()
    assert called == ["initialize"]
    saved = (out / "report.json").read_bytes()
    with pytest.raises(ValueError):
        probe.main(["--output-dir", str(out)])
    assert (out / "report.json").read_bytes() == saved


def test_graph_cleanup_preserves_original_error():
    class BrokenGraph:
        def reset(self):
            raise RuntimeError("reset failed")
    record = {}
    probe.reset_graph(BrokenGraph(), record, ValueError("original"))
    assert "reset failed" in record["graph_cleanup_error"]
    with pytest.raises(RuntimeError, match="reset failed"):
        probe.reset_graph(BrokenGraph(), {}, None)
