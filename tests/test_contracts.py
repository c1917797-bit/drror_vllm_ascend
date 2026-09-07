from __future__ import annotations

import copy
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from drror_vllm_ascend import audit, envs, prepare, selection
from drror_vllm_ascend.envs import DrrqrConfig
from drror_vllm_ascend.patches import capture, gdn, model
from drror_vllm_ascend.plan import DrrqrPlan


class EnvironmentTests(unittest.TestCase):
    def test_disabled_is_inert_and_capture_requires_hashes(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(envs.get_config().enable)
            self.assertFalse(envs.get_config().capture_enable)
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_ASCEND_DRRQR_CAPTURE_ENABLE": "1",
                "VLLM_ASCEND_DRRQR_CAPTURE_DIR": "C:\\captures",
            },
            clear=True,
        ), self.assertRaisesRegex(ValueError, "SOURCE_CONFIG_SHA256"):
            envs.get_config()

    def test_capture_and_treatment_are_mutually_exclusive(self):
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_ASCEND_DRRQR_ENABLE": "1",
                "VLLM_ASCEND_DRRQR_CAPTURE_ENABLE": "1",
            },
            clear=True,
        ), self.assertRaisesRegex(ValueError, "mutually exclusive"):
            envs.get_config()


class CapturePatchTests(unittest.TestCase):
    class FakeGdn:
        def rearrange_mixed_qkv(self, value):
            self.original_calls += 1
            return value

    def setUp(self):
        original_rearrange = self.FakeGdn.rearrange_mixed_qkv
        self.addCleanup(
            setattr,
            self.FakeGdn,
            "rearrange_mixed_qkv",
            original_rearrange,
        )
        self.temp = tempfile.TemporaryDirectory(prefix="drror-capture-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / ".armed").touch()
        self.evidence = self.root / "evidence.jsonl"
        self.metadata = types.SimpleNamespace(
            num_prefills=1,
            num_decodes=0,
            spec_sequence_masks=None,
        )
        self.module = types.SimpleNamespace(
            torch=torch,
            get_forward_context=lambda: types.SimpleNamespace(
                attn_metadata={"model.language_model.layers.0.linear_attn": self.metadata}
            ),
        )
        self.config = DrrqrConfig(
            capture_enable=True,
            capture_dir=str(self.root),
            capture_max_tokens=3,
            capture_max_per_layer=1,
            source_config_sha256="a" * 64,
            calibration_sha256="b" * 64,
            evidence_file=str(self.evidence),
        )

    def instance(self):
        value = self.FakeGdn()
        value.original_calls = 0
        value.prefix = "model.language_model.layers.0.linear_attn"
        value.tp_rank = 0
        value.tp_size = 2
        value.key_dim = 8
        value.head_k_dim = 4
        return value

    def test_bounded_post_conv_capture_with_provenance(self):
        capture.install_capture_patch(self.FakeGdn, self.module, self.config)
        instance = self.instance()
        packed = torch.arange(60, dtype=torch.float32).reshape(5, 12)
        instance.rearrange_mixed_qkv(packed)
        instance.rearrange_mixed_qkv(packed)
        files = list(self.root.glob("*.pt"))
        self.assertEqual(len(files), 1)
        payload = torch.load(files[0], map_location="cpu", weights_only=True)
        self.assertEqual(payload["schema"], capture.CAPTURE_SCHEMA)
        self.assertEqual(payload["target_model_id"], "Qwen/Qwen3.8-27B")
        self.assertEqual(payload["source_config_sha256"], "a" * 64)
        self.assertEqual(payload["calibration_sha256"], "b" * 64)
        self.assertEqual(payload["q"].shape, (3, 4))
        self.assertEqual(payload["k"].shape, (3, 4))
        self.assertEqual(instance.original_calls, 2)
        records = [
            json.loads(line)
            for line in self.evidence.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual([row["event"] for row in records], ["post_conv_qk_capture_written"])

    def test_mixed_decode_capture_fails_closed(self):
        capture.install_capture_patch(self.FakeGdn, self.module, self.config)
        self.metadata.num_decodes = 1
        with self.assertRaisesRegex(RuntimeError, "pure non-speculative prefill"):
            self.instance().rearrange_mixed_qkv(torch.ones(2, 12))


class SelectionTests(unittest.TestCase):
    def test_strong_rrqr_is_deterministic_and_bounded(self):
        generator = torch.Generator().manual_seed(7)
        values = torch.randn(6000, 8, generator=generator)
        first = selection.strong_rrqr_indices(values, 4, seed=42)
        second = selection.strong_rrqr_indices(values, 4, seed=42)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(len(set(first.tolist())), 4)
        self.assertTrue(all(0 <= value < 8 for value in first))

    def test_nonfinite_activation_is_rejected(self):
        values = torch.eye(4)
        values[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            selection.strong_rrqr_indices(values, 2)


class PrepareTests(unittest.TestCase):
    def qwen38_config(self):
        types_ = [
            "full_attention" if (layer + 1) % 4 == 0 else "linear_attention"
            for layer in range(64)
        ]
        return {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "language_model_only": False,
            "text_config": {
                "model_type": "qwen3_5_text",
                "dtype": "bfloat16",
                "linear_key_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 48,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "hidden_size": 5120,
                "num_hidden_layers": 64,
                "layer_types": types_,
            },
        }

    def test_three_ratios_follow_official_floor_rule(self):
        self.assertEqual(
            [prepare._target_dim(128, ratio) for ratio in (0.2, 0.3, 0.5)],
            [102, 89, 64],
        )

    def test_exact_qwen38_hybrid_metadata_is_accepted(self):
        with tempfile.TemporaryDirectory(prefix="drror-source-") as directory:
            root = Path(directory)
            shard = root / "model.safetensors"
            shard.touch()
            config = self.qwen38_config()
            linear_layers = [
                index
                for index, kind in enumerate(config["text_config"]["layer_types"])
                if kind == "linear_attention"
            ]
            weight_map = {}
            for layer in linear_layers:
                base = f"model.language_model.layers.{layer}.linear_attn"
                weight_map[f"{base}.in_proj_qkv.weight"] = shard.name
                weight_map[f"{base}.conv1d.weight"] = shard.name
            index = {"weight_map": weight_map}

            class Slice:
                def __init__(self, shape):
                    self.shape = shape

                def get_shape(self):
                    return self.shape

                def get_dtype(self):
                    return "BF16"

            class Handle:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def keys(self):
                    return weight_map

                def get_slice(self, name):
                    shape = (10240, 1, 4) if ".conv1d." in name else (10240, 5120)
                    return Slice(shape)

            with mock.patch.object(prepare, "_safe_open", return_value=Handle()):
                text, observed = prepare.validate_source(
                    root,
                    config,
                    index,
                    102,
                    4,
                )
            self.assertEqual(text["model_type"], "qwen3_5_text")
            self.assertEqual(observed, linear_layers)
            self.assertEqual(len(observed), 48)

            wrong = copy.deepcopy(config)
            wrong["text_config"]["layer_types"][0] = "full_attention"
            with self.assertRaisesRegex(ValueError, "3:1 hybrid topology"):
                prepare.validate_source(root, wrong, index, 102, 4)


class WeightTransformTests(unittest.TestCase):
    def test_qk_and_conv_are_selected_while_v_is_preserved(self):
        plan = DrrqrPlan(
            digest="c" * 64,
            source_config_sha256="d" * 64,
            source_index_sha256="e" * 64,
            old_head_k_dim=4,
            target_head_k_dim=2,
            num_key_heads=2,
            num_value_heads=4,
            head_v_dim=3,
            conv_kernel_dim=2,
            hidden_size=5,
            layer_types=("linear_attention", "full_attention"),
            keep_indices=((0, (0, 2, 4, 7)),),
        )
        width = 2 * 2 * 4 + 4 * 3
        qkv = torch.arange(width * 5, dtype=torch.float32).reshape(width, 5)
        conv = torch.arange(width * 2, dtype=torch.float32).reshape(width, 1, 2)
        full = torch.ones(3, 3)
        transformed = dict(
            plan.transform_weights(
                iter(
                    [
                        ("layers.0.linear_attn.in_proj_qkv.weight", qkv),
                        ("layers.0.linear_attn.conv1d.weight", conv),
                        ("layers.1.self_attn.q_proj.weight", full),
                    ]
                )
            )
        )
        keep = torch.tensor([0, 2, 4, 7])
        expected = torch.cat(
            (
                qkv[:8].index_select(0, keep),
                qkv[8:16].index_select(0, keep),
                qkv[16:],
            )
        )
        self.assertTrue(
            torch.equal(
                transformed["layers.0.linear_attn.in_proj_qkv.weight"],
                expected,
            )
        )
        self.assertTrue(torch.equal(expected[8:], qkv[16:]))
        self.assertIs(transformed["layers.1.self_attn.q_proj.weight"], full)


class ModelPatchTests(unittest.TestCase):
    def test_model_constructor_binds_plan_and_loader_consumes_transform(self):
        class Plan:
            digest = "f" * 64
            old_head_k_dim = 128
            target_head_k_dim = 102
            keep_indices = tuple((layer, ()) for layer in range(48))

            def transform_weights(self, weights):
                yield from weights

        class Model:
            def __init__(self, *, vllm_config, prefix=""):
                self.config = vllm_config
                self.prefix = prefix

            def load_weights(self, weights):
                self.loaded = list(weights)
                return {"ok"}

        config = DrrqrConfig(
            enable=True,
            plan_path="/plan.json",
            plan_sha256="f" * 64,
        )
        with mock.patch.object(model, "load_runtime_plan", return_value=Plan()):
            model.install_model_patch(Model, config)
            instance = Model(vllm_config=object(), prefix="model")
            result = instance.load_weights(iter([("weight", torch.ones(1))]))
        self.assertEqual(result, {"ok"})
        self.assertEqual(len(instance.loaded), 1)
        self.assertEqual(instance._drror_drrqr_plan.target_head_k_dim, 102)

    def test_loader_that_does_not_consume_weights_fails_closed(self):
        class Plan:
            digest = "e" * 64
            old_head_k_dim = 128
            target_head_k_dim = 102
            keep_indices = tuple((layer, ()) for layer in range(48))

            def transform_weights(self, weights):
                yield from weights

        class Model:
            def __init__(self, *, vllm_config, prefix=""):
                self.config = vllm_config
                self.prefix = prefix

            def load_weights(self, weights):
                del weights
                return {"returned-early"}

        config = DrrqrConfig(
            enable=True,
            plan_path="/plan.json",
            plan_sha256="e" * 64,
        )
        with mock.patch.object(model, "load_runtime_plan", return_value=Plan()):
            model.install_model_patch(Model, config)
            instance = Model(vllm_config=object(), prefix="model")
            with self.assertRaisesRegex(RuntimeError, "did not consume all weights"):
                instance.load_weights(iter([("weight", torch.ones(1))]))


class GdnHotPathTests(unittest.TestCase):
    def test_reduced_shape_emits_hot_path_once(self):
        with tempfile.TemporaryDirectory(prefix="drror-gdn-") as directory:
            evidence = Path(directory) / "evidence.jsonl"

            class Target:
                @staticmethod
                def _chunk_gated_delta_rule_fused(*args):
                    return args[2], args[5]

            calls = []
            lengths = torch.tensor([0, 3])
            chunk = object()
            candidate = types.SimpleNamespace(
                prefill_query_start_loc=lengths,
                non_spec_prefill_metadata=types.SimpleNamespace(chunk=chunk),
            )

            def fallback(**kwargs):
                calls.append(kwargs)
                return kwargs["v"], kwargs["initial_state"] + 1

            module = types.SimpleNamespace(
                AscendGatedDeltaNetAttention=Target,
                get_forward_context=lambda: types.SimpleNamespace(
                    attn_metadata={"linear": candidate}
                ),
                chunk_gated_delta_rule=fallback,
            )
            config = DrrqrConfig(
                enable=True,
                plan_path="/plan.json",
                plan_sha256="1" * 64,
                evidence_file=str(evidence),
            )
            gdn.install_fused_shape_guard(module, config)
            q = torch.zeros(1, 3, 4, 102)
            v = torch.ones(1, 3, 12, 128)
            state = torch.zeros(1, 12, 128, 102)
            for _ in range(2):
                Target._chunk_gated_delta_rule_fused(
                    q,
                    q,
                    v,
                    torch.zeros(1, 3, 12),
                    torch.ones(1, 3, 12),
                    state,
                    lengths,
                    102**-0.5,
                )
            self.assertEqual(len(calls), 2)
            records = [
                json.loads(line)
                for line in evidence.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [row["event"] for row in records],
                ["reduced_gdn_hot_path"],
            )
            self.assertEqual(records[0]["key_head_dim"], 102)

    def test_two_65_token_requests_preserve_prebuilt_chunk_metadata(self):
        with tempfile.TemporaryDirectory(prefix="drror-gdn-multiseq-") as directory:
            evidence = Path(directory) / "evidence.jsonl"

            class Target:
                @staticmethod
                def _chunk_gated_delta_rule_fused(*args):
                    return args[2], args[5]

            starts = torch.tensor([0, 65, 130])
            chunk = object()
            candidate = types.SimpleNamespace(
                prefill_query_start_loc=starts,
                non_spec_prefill_metadata=types.SimpleNamespace(chunk=chunk),
            )
            calls = []

            def fallback(**kwargs):
                calls.append(kwargs)
                return kwargs["v"], kwargs["initial_state"]

            module = types.SimpleNamespace(
                AscendGatedDeltaNetAttention=Target,
                get_forward_context=lambda: types.SimpleNamespace(
                    attn_metadata={"linear": candidate}
                ),
                chunk_gated_delta_rule=fallback,
            )
            config = DrrqrConfig(
                enable=True,
                plan_path="/plan.json",
                plan_sha256="3" * 64,
                evidence_file=str(evidence),
            )
            gdn.install_fused_shape_guard(module, config)
            q = torch.zeros(1, 130, 4, 102)
            v = torch.ones(1, 130, 12, 128)
            state = torch.zeros(1, 12, 128, 102)
            Target._chunk_gated_delta_rule_fused(
                q,
                q,
                v,
                torch.zeros(1, 130, 12),
                torch.ones(1, 130, 12),
                state,
                starts,
                102**-0.5,
            )
            self.assertEqual(len(calls), 1)
            self.assertIs(calls[0]["prebuilt_meta"], chunk)
            self.assertIs(calls[0]["cu_seqlens"], starts)
            records = [
                json.loads(line)
                for line in evidence.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(records[0]["sequence_count"], 2)
            self.assertEqual(records[0]["token_count"], 130)


class ActivationAuditTests(unittest.TestCase):
    def test_tp4_complete_evidence_passes_and_missing_hot_path_fails(self):
        with tempfile.TemporaryDirectory(prefix="drror-audit-") as directory:
            path = Path(directory) / "evidence.jsonl"
            digest = "2" * 64
            rows = [
                {
                    "schema": "drror-vllm-ascend-evidence/v1",
                    "event": "plugin_build_active",
                    "component": "plugin",
                    "pid": 99,
                },
                {
                    "schema": "drror-vllm-ascend-evidence/v1",
                    "event": "runtime_patches_installed",
                    "component": "plugin",
                    "pid": 99,
                    "mode": "treatment",
                    "plan_sha256": digest,
                },
            ]
            for pid in range(1, 5):
                rows.extend(
                    [
                        {
                            "schema": "drror-vllm-ascend-evidence/v1",
                            "event": "model_configured",
                            "component": "model",
                            "pid": pid,
                            "plan_sha256": digest,
                            "old_head_k_dim": 128,
                            "target_head_k_dim": 102,
                            "linear_layer_count": 48,
                        },
                        {
                            "schema": "drror-vllm-ascend-evidence/v1",
                            "event": "weight_load_complete",
                            "component": "model",
                            "pid": pid,
                            "plan_sha256": digest,
                            "target_tensor_count": 96,
                        },
                        {
                            "schema": "drror-vllm-ascend-evidence/v1",
                            "event": "reduced_gdn_hot_path",
                            "component": "gdn",
                            "pid": pid,
                            "plan_sha256": digest,
                            "key_head_dim": 102,
                            "value_head_dim": 128,
                            "backend": "chunk_gated_delta_rule",
                            "token_count": 3,
                        },
                    ]
                )
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            result = audit.audit_activation(
                path,
                plan_sha256=digest,
                target_head_k_dim=102,
            )
            self.assertTrue(result["ok"], result)
            path.write_text(
                "".join(
                    json.dumps(row) + "\n"
                    for row in rows
                    if not (
                        row["event"] == "reduced_gdn_hot_path"
                        and row["pid"] == 4
                    )
                ),
                encoding="utf-8",
            )
            failed = audit.audit_activation(
                path,
                plan_sha256=digest,
                target_head_k_dim=102,
            )
            self.assertFalse(failed["ok"])
            self.assertTrue(any("worker 4" in error for error in failed["errors"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
