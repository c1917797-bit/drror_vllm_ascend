"""CPU tests for diagnostic recording only; these make no NPU runtime claim."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def probe(monkeypatch):
    # Do not import/initialize Ascend runtime during helper tests.
    stub = ModuleType("vllm_ascend.worker.worker")
    class FakeNPUWorker:
        def load_model(self):
            self.base_load_calls = getattr(self, "base_load_calls", 0) + 1
            if getattr(self, "load_error", None):
                raise self.load_error

        def initialize_from_config(self, config):
            self.base_cache_calls = getattr(self, "base_cache_calls", 0) + 1
            if getattr(self, "cache_error", None):
                raise self.cache_error

    stub.NPUWorker = FakeNPUWorker
    monkeypatch.setitem(sys.modules, "vllm_ascend.worker.worker", stub)
    path = Path(__file__).resolve().parents[1] / "tools/layerwise_probe_worker.py"
    spec = importlib.util.spec_from_file_location("isolated_layerwise_probe_worker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_parameter_hash_preserves_noncontiguous_numeric_values(probe, dtype):
    tensor = torch.nn.Parameter(torch.arange(24, dtype=dtype).view(4, 6).t(), requires_grad=False)
    before = tensor.detach().clone()
    metadata = probe.tensor_metadata(tensor)
    result = probe.parameter_identity("conv.weight", tensor)
    expected = before.contiguous().view(torch.uint8).numpy().tobytes()
    assert result["sha256"] == hashlib.sha256(expected).hexdigest()
    assert result["shape"] == [6, 4]
    assert result["contiguous"] is False
    assert probe.tensor_metadata(tensor) == metadata
    assert torch.equal(tensor, before)
    tensor[0, 0] += 1
    assert probe.parameter_identity("conv.weight", tensor)["sha256"] != result["sha256"]


def test_inference_parameter_and_scalar_hash(probe):
    with torch.inference_mode():
        tensor = torch.nn.Parameter(torch.tensor(1, dtype=torch.bfloat16), requires_grad=False)
    result = probe.parameter_identity("scalar", tensor)
    assert result["version"] is None and result["inference_tensor"]
    assert result["sha256"] == hashlib.sha256(bytes([128, 63])).hexdigest()


def test_success_report_exclusive_and_rank_separated(probe, tmp_path):
    with probe.phase_report(str(tmp_path), 0, 0, "load") as report:
        report["plan_sha256"] = "a" * 64
    path = tmp_path / "rank0.load.json"
    original = path.read_bytes()
    data = json.loads(original)
    assert data["status"] == "passed"
    assert data["plan_sha256"] == "a" * 64
    with pytest.raises(FileExistsError):
        with probe.phase_report(str(tmp_path), 0, 0, "load"):
            pytest.fail("existing result must not enter")
    assert path.read_bytes() == original
    with probe.phase_report(str(tmp_path), 1, 1, "load"):
        pass
    assert (tmp_path / "rank1.load.json").is_file()


def test_failed_report_preserves_partial_evidence_and_raises(probe, tmp_path):
    with pytest.raises(ValueError, match="intentional"):
        with probe.phase_report(str(tmp_path), 3, 3, "cache") as report:
            report["layers"].append({"layer": 0, "states": []})
            raise ValueError("intentional test failure")
    data = json.loads((tmp_path / "rank3.cache.json").read_text())
    assert data["status"] == "failed"
    assert data["layers"] == [{"layer": 0, "states": []}]
    assert data["error"]["type"] == "ValueError"
    assert "intentional test failure" in data["traceback"]


def test_probe_does_not_create_missing_result_directory(probe, tmp_path):
    target = tmp_path / "not-created"
    with pytest.raises(RuntimeError, match="already exist"):
        with probe.phase_report(str(target), 0, 0, "load"):
            pass
    assert not target.exists()


def fake_module(dk=64, qkvz=True):
    ref = SimpleNamespace(
        target_head_k_dim=dk, head_v_dim=8, num_key_heads=4,
        num_value_heads=4, hidden_size=2, conv_kernel_dim=4,
    )
    packed, value = 2 * dk + 8, 8
    module = SimpleNamespace(
        layer_idx=0, tp_size=4, tp_rank=2, head_k_dim=dk, head_v_dim=8,
        num_k_heads=4, num_v_heads=4, hidden_size=2, conv_kernel_size=4,
        key_dim=4 * dk, value_dim=32, num_spec=0,
        conv1d=SimpleNamespace(weight=torch.empty(packed, 1, 4)),
        get_state_shape=lambda: ((3, packed), (1, 8, dk)),
        get_state_dtype=lambda: (torch.bfloat16, torch.bfloat16),
    )
    if qkvz:
        module.in_proj_qkvz = SimpleNamespace(weight=torch.empty(packed + value, 2))
    else:
        module.in_proj_qkvz = None
        module.in_proj_qkv = SimpleNamespace(weight=torch.empty(packed, 2))
        module.in_proj_z = SimpleNamespace(weight=torch.empty(value, 2))
    return module, ref


@pytest.mark.parametrize("qkvz", [True, False])
def test_module_shapes_accept_only_selected_local_width(probe, qkvz):
    module, plan = fake_module(qkvz=qkvz)
    record = probe.check_module(module, plan, 0, 2)
    assert record["local_qkv_rows"] == 136
    assert record["state_shapes"][1] == [1, 8, 64]
    module.head_k_dim = 32
    with pytest.raises(RuntimeError, match="head_k_dim"):
        probe.check_module(module, plan, 0, 2)


def test_layerwise_dimension_overrides_global_envelope(probe):
    module, reference = fake_module(dk=32)
    reference.target_head_k_dim = 64
    plan = SimpleNamespace(reference=reference, layer_dim=lambda layer: 32)
    record = probe.check_module(module, plan, 0, 2)
    assert record["state_shapes"][1] == [1, 8, 32]
    module.get_state_shape = lambda: ((3, 72), (1, 8, 64))
    with pytest.raises(RuntimeError, match="recurrent state"):
        probe.check_module(module, plan, 0, 2)


def test_module_shapes_reject_wrong_tp_weight_partition(probe):
    module, plan = fake_module()
    module.in_proj_qkvz.weight = torch.empty(145, 2)
    with pytest.raises(RuntimeError, match="QKVZ weight"):
        probe.check_module(module, plan, 0, 2)


def test_parent_load_failure_is_saved_and_second_load_rejected(probe, tmp_path, monkeypatch):
    monkeypatch.setenv("DRRQR_LAYERWISE_PROBE_DIR", str(tmp_path))
    worker = probe.LayerwiseProbeWorker()
    worker.rank = worker.local_rank = 0
    worker._admit = lambda report: None
    worker.load_error = ValueError("base load failed")
    with pytest.raises(ValueError, match="base load failed"):
        worker.load_model()
    path = tmp_path / "rank0.load.json"
    original = path.read_bytes()
    assert json.loads(original)["status"] == "failed"
    assert worker.base_load_calls == 1
    with pytest.raises(RuntimeError, match="only once"):
        worker.load_model()
    assert path.read_bytes() == original and worker.base_load_calls == 1


def test_parent_cache_failure_is_saved_and_second_binding_rejected(probe, tmp_path, monkeypatch):
    monkeypatch.setenv("DRRQR_LAYERWISE_PROBE_DIR", str(tmp_path))
    worker = probe.LayerwiseProbeWorker()
    worker.rank = worker.local_rank = 2
    worker._admit = lambda report: None
    worker._layerwise_probe_load_passed = True
    worker.cache_error = ValueError("base cache failed")
    with pytest.raises(ValueError, match="base cache failed"):
        worker.initialize_from_config(None)
    path = tmp_path / "rank2.cache.json"
    original = path.read_bytes()
    assert json.loads(original)["status"] == "failed"
    with pytest.raises(RuntimeError, match="only once"):
        worker.initialize_from_config(None)
    assert path.read_bytes() == original and worker.base_cache_calls == 1
