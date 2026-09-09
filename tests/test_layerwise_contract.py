from dataclasses import replace
from types import SimpleNamespace
import hashlib
import json

import pytest
import torch

from drror_vllm_ascend.layerwise import LayerwiseDrrqrPlan
from drror_vllm_ascend.plan import DrrqrPlan
from drror_vllm_ascend import plan as plan_module
from drror_vllm_ascend.layerwise import SCHEMA
from drror_vllm_ascend.patches.layerwise import install_layerwise_constructor_patch
from drror_vllm_ascend.runtime_plan import load_bound_plan
from drror_vllm_ascend.envs import DrrqrConfig


def bound_manifest(tmp_path, mismatched_capture=False):
    refs = {}
    for dk in (32, 64):
        base = plan(dk)
        data = {
            "schema": plan_module.SCHEMA, "target_model_id": plan_module.TARGET_MODEL_ID,
            "model_type": plan_module.TARGET_TEXT_MODEL_TYPE,
            "source_config_sha256": plan_module.OFFICIAL_CONFIG_SHA256,
            "source_index_sha256": plan_module.OFFICIAL_INDEX_SHA256,
            "selection_objective": "energy-kernel",
            "provenance": {
                "method": "DRRQR", "official_commit": plan_module.OFFICIAL_COMMIT,
                "official_rrqr_sha256": plan_module.OFFICIAL_RRQR_SHA256,
                "calibration_sha256": "d" * 64,
                "selection": {"capture_manifest_sha256": ("f" if mismatched_capture and dk == 32 else "e") * 64},
            },
            "keep_indices": {str(k): list(v) for k, v in base.keep_indices},
            "layer_types": list(base.layer_types),
        }
        for field in ("old_head_k_dim", "target_head_k_dim", "num_key_heads",
                      "num_value_heads", "head_v_dim", "conv_kernel_dim", "hidden_size"):
            data[field] = getattr(base, field)
        source = tmp_path / f"dk{dk}.json"
        source.write_text(json.dumps(data))
        refs[str(dk)] = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    manifest = tmp_path / "layerwise.json"
    manifest.write_text(json.dumps({"schema": SCHEMA, "source_plans": refs,
                                   "layer_head_k_dims": {"0": 128, "1": 32, "2": 32, "3": 64}}))
    return manifest, hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_bound_manifest_roundtrip(tmp_path):
    path, digest = bound_manifest(tmp_path)
    result = LayerwiseDrrqrPlan.load(str(path), digest)
    assert dict(result.layer_dims) == {0: 128, 1: 32, 2: 32, 3: 64}
    # Legacy serving intentionally rejects the new offline contract.
    with pytest.raises(ValueError):
        DrrqrPlan.load(str(path), digest)
    with pytest.raises(ValueError, match="hash mismatch"):
        LayerwiseDrrqrPlan.load(str(path), "0" * 64)
    loaded = load_bound_plan(
        DrrqrConfig(plan_path=str(path), plan_sha256=digest)
    )
    assert isinstance(loaded, LayerwiseDrrqrPlan)
    assert loaded.digest == digest


def test_bound_manifest_rejects_capture_mix(tmp_path):
    path, digest = bound_manifest(tmp_path, mismatched_capture=True)
    with pytest.raises(ValueError, match="provenance"):
        LayerwiseDrrqrPlan.load(str(path), digest)


def plan(dk):
    return DrrqrPlan(
        digest="a" * 64, source_config_sha256="b" * 64, source_index_sha256="c" * 64,
        old_head_k_dim=128, target_head_k_dim=dk, num_key_heads=4,
        num_value_heads=4, head_v_dim=8, conv_kernel_dim=4, hidden_size=2,
        layer_types=("linear_attention", "linear_attention", "linear_attention",
                     "linear_attention", "full_attention"),
        keep_indices=tuple((layer, tuple(head * 128 + j for head in range(4) for j in range(dk)))
                           for layer in range(4)))


def weights():
    width = 2 * 4 * 128 + 4 * 8
    for layer in range(4):
        yield f"model.language_model.layers.{layer}.linear_attn.in_proj_qkv.weight", torch.arange(width * 2).float().view(width, 2)
        yield f"model.language_model.layers.{layer}.linear_attn.conv1d.weight", torch.arange(width * 4).float().view(width, 1, 4)


def test_uniform_contract_is_exact_identity_of_old_transformer():
    base = plan(64)
    mixed = LayerwiseDrrqrPlan.compose({64: base}, {i: 64 for i in range(4)})
    for (name, actual), (expected_name, expected) in zip(mixed.transform_weights(weights()), base.transform_weights(weights())):
        assert name == expected_name
        assert torch.equal(actual, expected)


def test_heterogeneous_weights_keep_dense_layer_and_v_rows():
    mixed = LayerwiseDrrqrPlan.compose({32: plan(32), 64: plan(64)}, {0: 128, 1: 32, 2: 32, 3: 64})
    originals = dict(weights())
    for name, actual in mixed.transform_weights(weights()):
        layer = int(name.split(".")[3])
        assert actual.shape[0] == mixed.layer_shapes(layer)["global_qkv_rows"]
        assert torch.equal(actual[-32:], originals[name][-32:])
        if layer == 0:
            assert torch.equal(actual, originals[name])
    assert mixed.max_head_k_dim == 128


def test_layer_config_does_not_mutate_shared_config_or_other_layers():
    mixed = LayerwiseDrrqrPlan.compose({32: plan(32), 64: plan(64)}, {0: 128, 1: 32, 2: 32, 3: 64})
    config = SimpleNamespace(linear_key_head_dim=128, nested={"x": []})
    local = mixed.isolated_layer_config(config, "model.language_model.layers.1.linear_attn")
    local.nested["x"].append(1)
    assert local.linear_key_head_dim == 32
    assert config.linear_key_head_dim == 128 and config.nested["x"] == []
    assert mixed.isolated_layer_config(config, "model.layers.0.linear_attn").linear_key_head_dim == 128
    with pytest.raises(ValueError):
        mixed.isolated_layer_config(config, "model.layers.4.linear_attn")


def test_constructor_patch_binds_each_layer_without_mutating_shared_config():
    mixed = LayerwiseDrrqrPlan.compose(
        {32: plan(32), 64: plan(64)},
        {0: 128, 1: 32, 2: 32, 3: 64},
    )

    class Attention:
        def __init__(
            self,
            config,
            vllm_config,
            prefix="",
            gqa_interleaved_layout=False,
        ):
            self.dk = config.linear_key_head_dim
            self.prefix = prefix
            self.marker = config.nested

    original = Attention.__init__
    try:
        install_layerwise_constructor_patch(Attention, mixed, ("p", "h"))
        shared = SimpleNamespace(linear_key_head_dim=128, nested={"x": []})
        first = Attention(shared, object(), "model.layers.0.linear_attn")
        second = Attention(shared, object(), "model.layers.1.linear_attn")
        last = Attention(shared, object(), "model.layers.3.linear_attn")
        assert (first.dk, second.dk, last.dk) == (128, 32, 64)
        second.marker["x"].append(1)
        assert shared.linear_key_head_dim == 128
        assert shared.nested == {"x": []}
        install_layerwise_constructor_patch(Attention, mixed, ("p", "h"))
        with pytest.raises(RuntimeError, match="another plan"):
            install_layerwise_constructor_patch(Attention, mixed, ("q", "h"))
    finally:
        Attention.__init__ = original


@pytest.mark.parametrize("dims", [
    {0: 64}, {0: 64, 1: 64, 2: 64, 3: 64, 4: 64},
    {0: 104, 1: 64, 2: 64, 3: 64}, {0: True, 1: 64, 2: 64, 3: 64},
])
def test_reject_invalid_layer_widths(dims):
    with pytest.raises(ValueError):
        LayerwiseDrrqrPlan.compose({64: plan(64)}, dims)


def test_reject_mismatched_sources():
    with pytest.raises(ValueError, match="different model contracts"):
        LayerwiseDrrqrPlan.compose({32: replace(plan(32), hidden_size=3), 64: plan(64)},
                                  {0: 32, 1: 64, 2: 64, 3: 64})


def test_reject_unused_sources_and_cross_head_indices():
    with pytest.raises(ValueError, match="exactly match"):
        LayerwiseDrrqrPlan.compose({32: plan(32), 64: plan(64)}, {i: 64 for i in range(4)})
    base = plan(64)
    keeps = list(base.keep_indices)
    layer, values = keeps[0]
    bad = list(values)
    bad[0] = 127 + 128
    keeps[0] = (layer, tuple(bad))
    with pytest.raises(ValueError):
        LayerwiseDrrqrPlan.compose({64: replace(base, keep_indices=tuple(keeps))},
                                  {i: 64 for i in range(4)})


@pytest.mark.parametrize("widths", [{i: 64 for i in range(4)}, {0: 128, 1: 32, 2: 32, 3: 64}])
def test_complete_model_constructor_and_loader_bind_layerwise_manifest(widths, monkeypatch):
    from drror_vllm_ascend.patches import model as model_patch
    sources = {dk: plan(dk) for dk in set(widths.values()) if dk != 128}
    bound = replace(LayerwiseDrrqrPlan.compose(sources, widths), manifest_digest="f" * 64)
    events = []
    class Attention:
        def __init__(self, config, vllm_config, prefix="", gqa_interleaved_layout=False):
            self.dk = config.linear_key_head_dim
    class Model:
        def __init__(self, *, vllm_config, prefix=""):
            self.layers = [Attention(vllm_config.model_config.hf_text_config,
                                    vllm_config, f"model.layers.{i}.linear_attn")
                           for i in range(4)]
        def load_weights(self, weights):
            self.loaded = dict(weights)
            return set(self.loaded)
    monkeypatch.setattr(model_patch, "load_runtime_plan", lambda *_: bound)
    monkeypatch.setattr(model_patch, "emit_evidence",
                        lambda *a, **k: events.append((a[1], k)))
    config = DrrqrConfig(enable=True, plan_path="/bound.json", plan_sha256="f" * 64)
    runtime = SimpleNamespace(model_config=SimpleNamespace(
        hf_text_config=SimpleNamespace(linear_key_head_dim=bound.max_head_k_dim)))
    model_patch.install_model_patch(Model, config, attention_cls=Attention)
    instance = Model(vllm_config=runtime)
    assert [layer.dk for layer in instance.layers] == list(widths.values())
    assert runtime.model_config.hf_text_config.linear_key_head_dim == bound.max_head_k_dim
    assert instance.load_weights(weights()) == set(dict(weights()))
    assert events[0][1]["old_head_k_dim"] == 128
    assert events[0][1]["layer_head_k_dims"] == widths
    assert events[-1][1]["target_tensor_count"] == 8
    for name, actual in instance.loaded.items():
        layer = int(name.split(".")[3])
        assert actual.shape[0] == bound.layer_shapes(layer)["global_qkv_rows"]


def test_shared_config_is_max_layer_envelope_not_implicit_dense(monkeypatch):
    bound = LayerwiseDrrqrPlan.compose({64: plan(64)}, {i: 64 for i in range(4)})
    monkeypatch.setattr(DrrqrPlan, "validate_source", lambda *_: None)
    ref = bound.reference
    text = SimpleNamespace(
        model_type=plan_module.TARGET_TEXT_MODEL_TYPE,
        linear_key_head_dim=64, linear_num_key_heads=ref.num_key_heads,
        linear_num_value_heads=ref.num_value_heads, linear_value_head_dim=ref.head_v_dim,
        linear_conv_kernel_dim=ref.conv_kernel_dim, hidden_size=ref.hidden_size,
        layer_types=ref.layer_types,
    )
    runtime = SimpleNamespace(
        model_config=SimpleNamespace(model="/model", hf_text_config=text,
                                     dtype=torch.bfloat16, quantization=None),
        parallel_config=SimpleNamespace(tensor_parallel_size=4, pipeline_parallel_size=1))
    bound.validate_runtime(runtime)
    text.linear_key_head_dim = 128
    with pytest.raises(ValueError, match="linear_key_head_dim"):
        bound.validate_runtime(runtime)


def test_incompatible_compact_states_cannot_share_hybrid_pool():
    from drror_vllm_ascend.patches.layerwise import validate_shared_mamba_layouts
    class Spec:
        def __init__(self, dk):
            self.shapes = ((3, 1536 + 8 * dk), (12, 128, dk))
            self.dtypes = (torch.bfloat16, torch.bfloat16)
    specs = {"a": Spec(64), "b": Spec(64), "c": Spec(32), "attention": object()}
    validate_shared_mamba_layouts(
        [SimpleNamespace(shared_by=["a", "b", "attention"])], specs, Spec)
    validate_shared_mamba_layouts(
        [SimpleNamespace(shared_by=["a"]), SimpleNamespace(shared_by=["c"])], specs, Spec)
    with pytest.raises(RuntimeError, match="incompatible raw cache"):
        validate_shared_mamba_layouts(
            [SimpleNamespace(shared_by=["a", "c", "attention"])], specs, Spec)
    with pytest.raises(RuntimeError, match="shares an attention pool"):
        validate_shared_mamba_layouts(
            [SimpleNamespace(shared_by=["c", "attention"])], specs, Spec, expected_max_dk=64)
    validate_shared_mamba_layouts(
        [SimpleNamespace(shared_by=["a", "attention"])], specs, Spec, expected_max_dk=64)
