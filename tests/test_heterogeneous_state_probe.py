"""CPU-only checks for arena bookkeeping; these are not native NPU results."""
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def probe(monkeypatch):
    stub = ModuleType("torch_npu")
    stub.__version__ = "test-stub"
    stub.npu = SimpleNamespace(set_compile_mode=lambda **kwargs: None)
    monkeypatch.setitem(sys.modules, "torch_npu", stub)
    script = Path(__file__).resolve().parents[1] / "tools/heterogeneous_state_probe.py"
    spec = importlib.util.spec_from_file_location("isolated_arena_probe", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_exact_distinguishes_signed_zero(probe, dtype):
    positive = torch.tensor([0.0], dtype=dtype)
    negative = torch.tensor([-0.0], dtype=dtype)
    assert torch.equal(positive, negative)  # The historical v4 predicate.
    assert not probe.exact(positive, negative)
    assert probe.exact(positive, positive.clone())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_exact_compares_logical_bytes_of_noncontiguous_values(probe, dtype):
    value = torch.arange(12, dtype=dtype).view(3, 4).t()
    assert not value.is_contiguous()
    assert probe.exact(value, value.contiguous())
    changed = value.contiguous()
    changed[0, 0] = 42
    assert not probe.exact(value, changed)
    assert probe.exact(torch.tensor(1.0, dtype=dtype), torch.tensor(1.0, dtype=dtype))


def test_exact_rejects_dtype_and_shape_mismatch(probe):
    assert not probe.exact(torch.ones(4, dtype=torch.bfloat16), torch.ones(4, dtype=torch.float32))
    assert not probe.exact(torch.ones(2, 2), torch.ones(4))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_exact_rejects_nonfinite_even_with_identical_bytes(probe, value):
    tensor = torch.tensor([value])
    assert not probe.exact(tensor, tensor.clone())


@pytest.mark.parametrize("dtype,guard", [(torch.bfloat16, 256), (torch.float32, 128)])
def test_typed_arena_preserves_512_byte_guards_and_state_offsets(probe, monkeypatch, dtype, guard):
    real_full = torch.full
    def cpu_full(*args, **kwargs):
        assert kwargs["device"] == "npu:0"
        return real_full(*args, **{**kwargs, "device": "cpu"})
    monkeypatch.setattr(probe.torch, "full", cpu_full)
    dks, slots = (128, 32, 32, 64), 4
    storage, states, guards = probe.arena(dks, slots, dtype)
    assert probe.guard_elements(dtype) == guard
    assert len(guards) == 5
    assert all(g.numel() == guard and g.numel() * g.element_size() == 512 for g in guards)
    for dk, state in zip(dks, states):
        assert state.dtype == dtype
        assert tuple(state.shape) == (slots, 12, 128, dk)
        assert (state.data_ptr() - storage.data_ptr()) % 512 == 0
        state.fill_(0.5)
    assert all(torch.equal(g, torch.full_like(g, probe.SENTINEL)) for g in guards)


def test_metadata_detects_layout_dtype_pointer_and_storage_offset_changes(probe):
    storage = torch.arange(12, dtype=torch.float32)
    view = storage[2:10].view(2, 4)
    original = probe.tensor_metadata(view)
    assert original == {
        "shape": [2, 4], "stride": [4, 1], "dtype": "torch.float32", "device": "cpu",
        "data_ptr": view.data_ptr(), "storage_ptr": storage.data_ptr(), "storage_offset": 2,
    }
    assert probe.tensor_metadata(view.t()) != original
    assert probe.tensor_metadata(view.clone()) != original
    assert probe.tensor_metadata(view.view(torch.int32)) != original
    assert probe.tensor_metadata(storage[3:11].view(2, 4)) != original


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_unselected_slot_checks_are_independent_of_reference_parity(probe, dtype):
    before = torch.ones(5, 2, dtype=dtype)
    after = before.clone()
    after[[0, 3]] = 7
    assert probe.unchanged_slots(before, after, [0, 3])
    # Even if every execution path made the same erroneous write, this fails.
    after[4, 0] = 8
    assert not probe.unchanged_slots(before, after, [0, 3])
    with pytest.raises(ValueError, match="slot range"):
        probe.unchanged_slots(before, after, [5])


def test_unselected_slots_preserve_signed_zero(probe):
    before = torch.zeros(3, 1)
    after = before.clone()
    after[2] = -0.0
    assert not probe.unchanged_slots(before, after, [0])
    assert probe.unchanged_slots(before, after, [2])


def test_graph_cleanup_does_not_mask_original_exception(probe):
    primary = ValueError("original numerical failure")
    class BrokenGraph:
        def reset(self):
            raise RuntimeError("cleanup failed")
    record = {}
    with pytest.raises(ValueError) as observed:
        try:
            raise primary
        finally:
            probe.reset_graph(BrokenGraph(), record, sys.exc_info()[1])
    assert observed.value is primary
    assert "cleanup failed" in record["graph_cleanup_error"]


def test_graph_cleanup_failure_rejects_otherwise_clean_case(probe):
    class BrokenGraph:
        def reset(self):
            raise RuntimeError("cleanup failed")
    record = {"status": "passed", "passed": True}
    with pytest.raises(RuntimeError, match="cleanup failed"):
        probe.reset_graph(BrokenGraph(), record, None)
    assert record["status"] == "failed" and record["passed"] is False
    assert record["failed_phase"] == "graph_reset"


@pytest.mark.parametrize("explicit", [False, True])
def test_legacy_two_argument_allocator_and_explicit_typed_path(probe, monkeypatch, explicit):
    assert probe.STATE_DTYPE == torch.bfloat16 and probe.GUARD == 256
    assert inspect.signature(probe.arena).parameters["state_dtype"].default == torch.bfloat16
    assert inspect.signature(probe.case).parameters["state_dtype"].default is None
    real_to, real_zeros, real_full = torch.Tensor.to, torch.zeros, torch.full
    def cpu_to(self, *args, **kwargs):
        if args and args[0] == "npu:0":
            args = ("cpu", *args[1:])
        return real_to(self, *args, **kwargs)
    def remap(factory):
        def make(*args, **kwargs):
            if kwargs.get("device") == "npu:0":
                kwargs["device"] = "cpu"
            return factory(*args, **kwargs)
        return make
    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    monkeypatch.setattr(torch, "zeros", remap(real_zeros))
    monkeypatch.setattr(torch, "full", remap(real_full))
    calls = []
    def stop_allocator(*args):
        calls.append(args)
        raise RuntimeError("stop before any native operation")
    monkeypatch.setattr(probe, "arena", stop_allocator)
    record = {}
    arguments = ("uniform64", (64,), 1)
    if explicit:
        arguments += (torch.float32,)
    with pytest.raises(RuntimeError, match="stop before"):
        probe.case(*arguments, record=record)
    assert len(calls) == 1
    assert calls[0] == ((64,), 4, torch.float32) if explicit else calls[0] == ((64,), 4)
    assert record["status"] == "failed" and record["failed_phase"] == "allocation"


def configure_fake_main(probe, monkeypatch, tmp_path, *, dtype=None, enable=True):
    root = tmp_path / "results"
    root.mkdir()
    source = tmp_path / "gdn.py"
    source.write_text("# fake runtime source for control-flow tests")
    maps = tmp_path / "maps"
    maps.write_text("")
    def mapped_path(value):
        if str(value) == "/drrqr-results":
            return root
        if str(value) == "/vllm-workspace/vllm-ascend/vllm_ascend/ops/gdn.py":
            return source
        if str(value) == "/proc/self/maps":
            return maps
        return Path(value)
    real_checksum = probe.checksum
    monkeypatch.setattr(probe, "Path", mapped_path)
    monkeypatch.setattr(probe, "checksum", lambda path: probe.ASCEND_GDN_SHA256 if Path(path) == source else real_checksum(path))
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")
    out = root / "run"
    argv = [probe.__file__, "--output-dir", str(out)]
    if dtype:
        argv += ["--state-dtype", dtype]
    monkeypatch.setattr(sys, "argv", argv)
    events = []
    probe.torch_npu.npu.set_compile_mode = lambda **kw: events.append(("compile_mode", kw))
    native_utils = ModuleType("vllm_ascend.utils")
    def enable_custom_op():
        events.append(("enable_custom_op", None))
        return enable
    native_utils.enable_custom_op = enable_custom_op
    triton_utils = ModuleType("vllm_ascend.ops.triton.triton_utils")
    triton_utils.init_device_properties_triton = lambda: events.append(("triton_properties", None))
    monkeypatch.setitem(sys.modules, "vllm_ascend.utils", native_utils)
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.triton.triton_utils", triton_utils)
    monkeypatch.setattr(probe.torch, "npu", SimpleNamespace(set_device=lambda device: events.append(("set_device", device))), raising=False)
    monkeypatch.setattr(probe.torch.ops, "_C_ascend", SimpleNamespace(
        npu_recurrent_gated_delta_rule=SimpleNamespace(default=SimpleNamespace(_schema="mock schema"))), raising=False)
    calls = []
    def fake_case(name, dks, batch, state_dtype, *, record, progress):
        calls.append((name, tuple(dks), batch, state_dtype))
        record.update(status="passed", passed=True, phase="complete")
        progress()
        return record
    monkeypatch.setattr(probe, "case", fake_case)
    return out, events, calls


@pytest.mark.parametrize("dtype,expected,guard", [(None, torch.bfloat16, 256), ("float32", torch.float32, 128)])
def test_main_preserves_golden_order_and_single_dtype_matrix(probe, monkeypatch, tmp_path, dtype, expected, guard):
    out, events, calls = configure_fake_main(probe, monkeypatch, tmp_path, dtype=dtype)
    probe.main()
    report = json.loads((out / "report.json").read_text())
    assert events == [
        ("compile_mode", {"jit_compile": False}), ("enable_custom_op", None),
        ("set_device", 0), ("triton_properties", None),
    ]
    assert len(calls) == 4 and all(c[-1] == expected for c in calls)
    assert [(c[0], c[2]) for c in calls] == [
        ("uniform64", 1), ("uniform64", 16), ("heterogeneous", 1), ("heterogeneous", 16),
    ]
    assert report["schema"] == "drrqr-heterogeneous-state-probe/v5"
    assert report["protocol"]["guard_elements"] == guard
    assert report["protocol"]["state_dtype"] == ("bfloat16" if dtype is None else dtype)
    assert report["protocol"]["steps"] == 8
    assert "signed zero" in report["historical_predicate"]
    original = (out / "report.json").read_bytes()
    with pytest.raises(SystemExit):
        probe.main()
    assert (out / "report.json").read_bytes() == original


def test_initialization_failure_has_receipt_and_no_case_or_device_call(probe, monkeypatch, tmp_path):
    out, events, calls = configure_fake_main(probe, monkeypatch, tmp_path, enable=False)
    with pytest.raises(RuntimeError, match="native custom operators unavailable"):
        probe.main()
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "failed" and report["phase"] == "enable_custom_op"
    assert report["operator_schema"] is None
    assert report["error_type"] == "RuntimeError" and "completed_unix_ns" in report
    assert calls == []
    assert [event[0] for event in events] == ["compile_mode", "enable_custom_op"]

