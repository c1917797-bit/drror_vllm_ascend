from __future__ import annotations

import contextvars
import copy
import hashlib
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from drror_vllm_ascend import audit, calibration, envs, prepare, selection
from drror_vllm_ascend.envs import DrrqrConfig
from drror_vllm_ascend.patches import bootstrap, capture, capture_binding, gdn, model
from drror_vllm_ascend.plan import (
    OFFICIAL_CONFIG_SHA256,
    OFFICIAL_INDEX_SHA256,
    DrrqrPlan,
    validate_official_checkpoint_hashes,
)


class EnvironmentTests(unittest.TestCase):
    def test_disabled_is_inert_and_capture_requires_hashes(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(envs.get_config().enable)
            self.assertFalse(envs.get_config().capture_enable)
        with (
            mock.patch.dict(
                os.environ,
                {
                    "VLLM_ASCEND_DRRQR_CAPTURE_ENABLE": "1",
                    "VLLM_ASCEND_DRRQR_CAPTURE_DIR": str(
                        Path.cwd().resolve() / "captures"
                    ),
                },
                clear=True,
            ),
            self.assertRaisesRegex(ValueError, "SOURCE_CONFIG_SHA256"),
        ):
            envs.get_config()

    def test_capture_and_treatment_are_mutually_exclusive(self):
        with (
            mock.patch.dict(
                os.environ,
                {
                    "VLLM_ASCEND_DRRQR_ENABLE": "1",
                    "VLLM_ASCEND_DRRQR_CAPTURE_ENABLE": "1",
                },
                clear=True,
            ),
            self.assertRaisesRegex(ValueError, "mutually exclusive"),
        ):
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
        self.binding = types.SimpleNamespace(
            verified=True,
            source_model_path=str(self.root),
            source_index_sha256="c" * 64,
            current_request=contextvars.ContextVar(
                "test_capture_request",
                default=None,
            ),
        )
        request_token = self.binding.current_request.set(
            {
                "row_index": 0,
                "input_ids": [1, 2, 3, 4, 5],
                "input_ids_sha256": calibration.token_sha256([1, 2, 3, 4, 5]),
            }
        )
        self.addCleanup(self.binding.current_request.reset, request_token)

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
        capture.install_capture_patch(self.FakeGdn, self.module, self.config, binding=self.binding)
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
        self.assertEqual(payload["source_index_sha256"], "c" * 64)
        self.assertEqual(payload["calibration_row_index"], 0)
        self.assertEqual(payload["input_ids_sha256"], calibration.token_sha256([1, 2, 3, 4, 5]))
        self.assertEqual(payload["q"].shape, (3, 4))
        self.assertEqual(payload["k"].shape, (3, 4))
        self.assertEqual(instance.original_calls, 2)
        records = [json.loads(line) for line in self.evidence.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["event"] for row in records], ["post_conv_qk_capture_written"])

    def test_mixed_decode_capture_fails_closed(self):
        capture.install_capture_patch(self.FakeGdn, self.module, self.config, binding=self.binding)
        self.metadata.num_decodes = 1
        with self.assertRaisesRegex(RuntimeError, "pure non-speculative prefill"):
            self.instance().rearrange_mixed_qkv(torch.ones(2, 12))

    def test_capture_requires_actual_model_and_request_binding(self):
        capture.install_capture_patch(self.FakeGdn, self.module, self.config, binding=self.binding)
        self.binding.verified = False
        with self.assertRaisesRegex(RuntimeError, "validated model"):
            self.instance().rearrange_mixed_qkv(torch.ones(5, 12))
        self.binding.verified = True
        token = self.binding.current_request.set(None)
        try:
            with self.assertRaisesRegex(RuntimeError, "validated model"):
                self.instance().rearrange_mixed_qkv(torch.ones(5, 12))
        finally:
            self.binding.current_request.reset(token)


class CaptureBindingTests(unittest.TestCase):
    def test_calibration_file_and_actual_tokens_are_bound(self):
        with tempfile.TemporaryDirectory(prefix="drror-binding-") as directory:
            path = Path(directory) / "calibration.jsonl"
            path.write_text('{"input_ids":[4,8,12]}\n', encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows = calibration.read_calibration(path, digest)
            binding = capture_binding.CaptureBinding(DrrqrConfig())
            binding.rows, binding.verified = rows, True
            self.assertEqual(binding.request(torch.tensor([4, 8, 12]))["row_index"], 0)
            for ids in (torch.tensor([4, 8]), torch.tensor([4, 9, 12]), None):
                with self.subTest(ids=ids), self.assertRaises(RuntimeError):
                    binding.request(ids)
            path.write_text('{"input_ids":[1,2,3]}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                calibration.read_calibration(path, digest)

    def test_model_forward_request_context_is_reset_after_failure(self):
        with tempfile.TemporaryDirectory(prefix="drror-forward-") as directory:
            config = DrrqrConfig(capture_enable=True, capture_dir=directory)
            binding = capture_binding.CaptureBinding(config)
            binding.verified = True
            binding.rows = [{"input_ids": [1, 2], "row_index": 0, "input_ids_sha256": calibration.token_sha256([1, 2])}]
            events = []

            class Model:
                def __init__(self, *, vllm_config, prefix=""):
                    events.append("constructed")

                def forward(self, input_ids, positions=None):
                    events.append(binding.current_request.get())
                    raise RuntimeError("synthetic kernel failure")

            with mock.patch.object(binding, "bind", side_effect=lambda config: events.append("bound")):
                capture_binding.install_capture_model_patch(Model, config, binding)
                instance = Model(vllm_config=object())
            self.assertEqual(events[:2], ["bound", "constructed"])
            (Path(directory) / ".armed").touch()
            with self.assertRaisesRegex(RuntimeError, "synthetic kernel failure"):
                instance.forward(torch.tensor([1, 2]))
            self.assertEqual(events[-1]["row_index"], 0)
            self.assertIsNone(binding.current_request.get())

    def test_capture_binding_checks_actual_disk_and_runtime_metadata(self):
        with tempfile.TemporaryDirectory(prefix="drror-source-bind-") as directory:
            root = Path(directory)
            source = PrepareTests().qwen38_config()
            (root / "config.json").write_text(json.dumps(source), encoding="utf-8")
            (root / "model.safetensors.index.json").write_text('{"weight_map":{}}', encoding="utf-8")
            calibration_path = root / "calibration.jsonl"
            calibration_path.write_text('{"input_ids":[1,2,3]}\n', encoding="utf-8")
            source_sha = hashlib.sha256((root / "config.json").read_bytes()).hexdigest()
            config = DrrqrConfig(
                capture_enable=True,
                capture_max_per_layer=1,
                source_config_sha256=source_sha,
                calibration_jsonl=str(calibration_path),
                calibration_sha256=hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
            )
            runtime = types.SimpleNamespace(
                model_config=types.SimpleNamespace(
                    model=str(root),
                    hf_text_config=types.SimpleNamespace(**source["text_config"]),
                    dtype=torch.bfloat16,
                    quantization=None,
                    enforce_eager=True,
                ),
                parallel_config=types.SimpleNamespace(tensor_parallel_size=4),
                lora_config=None,
                speculative_config=None,
            )
            # Synthetic small files cannot have the official model's hashes or
            # giant tensor headers. Isolate those independently tested gates.
            with (
                mock.patch.object(capture_binding, "validate_official_checkpoint_hashes") as check_hashes,
                mock.patch.object(
                    capture_binding, "validate_source", return_value=(source["text_config"], list(range(48)))
                ),
            ):
                binding = capture_binding.CaptureBinding(config)
                binding.bind(runtime)
                self.assertTrue(binding.verified)
                self.assertEqual(check_hashes.call_args.args[0], source_sha)
                runtime.model_config.hf_text_config.linear_key_head_dim = 64
                with self.assertRaisesRegex(ValueError, "runtime overrides source"):
                    capture_binding.CaptureBinding(config).bind(runtime)
                runtime.model_config.hf_text_config.linear_key_head_dim = 128
                (root / "config.json").write_text(json.dumps(source) + " ", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "loaded capture model differs"):
                    capture_binding.CaptureBinding(config).bind(runtime)


class V023PrefillTests(unittest.TestCase):
    def module(self, *, fail=False):
        calls = []

        def chunk(
            q,
            k,
            v,
            g,
            beta,
            scale=None,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=None,
            prebuilt_meta=None,
            head_first=False,
            use_qk_l2norm_in_kernel=False,
        ):
            calls.append(locals())
            if fail:
                raise RuntimeError("synthetic kernel failure")
            return v, initial_state

        # The real v0.23 module has no fused method. Do not add one to this fake.
        return types.SimpleNamespace(chunk_gated_delta_rule=chunk), calls

    def test_direct_v023_prefill_preserves_three_ratios_and_multisequence_metadata(self):
        for dk in (102, 89, 64):
            with self.subTest(dk=dk), tempfile.TemporaryDirectory(prefix="drror-v023-") as directory:
                evidence = Path(directory) / "evidence.jsonl"
                module, calls = self.module()
                config = DrrqrConfig(enable=True, plan_path="/plan", plan_sha256="a" * 64, evidence_file=str(evidence))
                gdn.install_prefill_patch(module, config)
                installed = module.chunk_gated_delta_rule
                gdn.install_prefill_patch(module, config)
                self.assertIs(module.chunk_gated_delta_rule, installed)
                q = torch.zeros(1, 130, 4, dk, dtype=torch.bfloat16)
                v = torch.ones(1, 130, 12, 128, dtype=torch.bfloat16)
                state = torch.zeros(2, 12, dk, 128)
                starts = torch.tensor([0, 65, 130])
                chunks = types.SimpleNamespace(chunk_indices_chunk64_host=(0, 0, 0, 1, 1, 0, 1, 1))
                for _ in range(2):
                    result = module.chunk_gated_delta_rule(
                        q,
                        q,
                        v,
                        None,
                        None,
                        initial_state=state,
                        cu_seqlens=starts,
                        prebuilt_meta=chunks,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=True,
                    )
                    self.assertIs(result[0], v)
                    self.assertIs(result[1], state)
                self.assertIs(calls[0]["prebuilt_meta"], chunks)
                self.assertIs(calls[0]["cu_seqlens"], starts)
                self.assertIs(calls[0]["initial_state"], state)
                self.assertIsNone(calls[0]["scale"])
                rows = [json.loads(line) for line in evidence.read_text().splitlines()]
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["sequence_count"], 2)
                self.assertEqual(rows[0]["key_head_dim"], dk)
                self.assertTrue(rows[0]["completed_python_call"])

    def test_failed_kernel_does_not_emit_success_evidence(self):
        with tempfile.TemporaryDirectory(prefix="drror-v023-fail-") as directory:
            evidence = Path(directory) / "evidence.jsonl"
            module, _ = self.module(fail=True)
            gdn.install_prefill_patch(module, DrrqrConfig(enable=True, evidence_file=str(evidence)))
            q = torch.zeros(1, 2, 4, 64)
            v = torch.zeros(1, 2, 12, 128)
            with self.assertRaisesRegex(RuntimeError, "synthetic kernel failure"):
                module.chunk_gated_delta_rule(
                    q,
                    q,
                    v,
                    None,
                    None,
                    initial_state=torch.zeros(1, 12, 64, 128),
                    cu_seqlens=torch.tensor([0, 2]),
                    prebuilt_meta=object(),
                    use_qk_l2norm_in_kernel=True,
                )
            self.assertFalse(evidence.exists())

    def test_missing_metadata_or_wrong_state_fails_before_kernel(self):
        module, calls = self.module()
        gdn.install_prefill_patch(module, DrrqrConfig(enable=True))
        q = torch.zeros(1, 2, 4, 89)
        v = torch.zeros(1, 2, 12, 128)
        for state, meta in ((torch.zeros(1, 12, 89, 128), None), (torch.zeros(1, 12, 128, 89), object())):
            with self.assertRaisesRegex(RuntimeError, "original state layout"):
                module.chunk_gated_delta_rule(
                    q,
                    q,
                    v,
                    None,
                    None,
                    initial_state=state,
                    cu_seqlens=torch.tensor([0, 2]),
                    prebuilt_meta=meta,
                    use_qk_l2norm_in_kernel=True,
                )
        self.assertEqual(calls, [])


class V023DecodeTests(unittest.TestCase):
    def test_decode_observer_preserves_core_and_records_actual_shapes(self):
        for dk in (64, 89, 102):
            with self.subTest(dk=dk), tempfile.TemporaryDirectory(prefix="drror-decode-") as directory:
                evidence = Path(directory) / "evidence.jsonl"
                metadata = types.SimpleNamespace(num_decodes=2, num_decode_tokens=2, spec_sequence_masks=None)
                module = types.SimpleNamespace(
                    get_forward_context=lambda meta=metadata: types.SimpleNamespace(attn_metadata={"layer": meta})
                )
                calls = []

                class Gdn:
                    prefix = "layer"

                    def __init__(self, dim=dk, call_log=calls):
                        self.dim, self.call_log = dim, call_log
                        self.kv_cache = [None, torch.zeros(4, 12, 128, dim)]

                    def rearrange_mixed_qkv(self, packed):
                        if packed is None:
                            return None, None, None
                        return (
                            torch.zeros(1, 2, 4, self.dim),
                            torch.zeros(1, 2, 4, self.dim),
                            torch.zeros(1, 2, 12, 128),
                        )

                    def _forward_core(self, mixed_qkv, b, a, core_attn_out):
                        self.call_log.append((mixed_qkv, core_attn_out))
                        self.rearrange_mixed_qkv(None)
                        self.rearrange_mixed_qkv(mixed_qkv)
                        return core_attn_out

                config = DrrqrConfig(enable=True, evidence_file=str(evidence))
                original_core = Gdn._forward_core
                gdn.install_decode_observer(Gdn, module, config)
                installed = Gdn._forward_core
                gdn.install_decode_observer(Gdn, module, config)
                self.assertIs(Gdn._forward_core, installed)
                self.assertIs(Gdn._forward_core._drror_core_original, original_core)
                packed, output = torch.zeros(2, 1024), torch.zeros(2, 12, 128)
                for _ in range(2):
                    self.assertIs(Gdn()._forward_core(packed, None, None, output), output)
                self.assertEqual(len(calls), 2)
                self.assertIs(calls[0][0], packed)
                row = json.loads(evidence.read_text().strip())
                self.assertEqual(row["event"], "reduced_gdn_decode_branch")
                self.assertEqual(row["query_shape"], [1, 2, 4, dk])
                self.assertEqual(row["state_shape"], [4, 12, 128, dk])
                self.assertEqual(row["backend"], "npu_recurrent_gated_delta_rule")


class SelectionTests(unittest.TestCase):
    def test_v2_capture_requires_matching_model_index_and_actual_token_row(self):
        with tempfile.TemporaryDirectory(prefix="drror-selection-binding-") as directory:
            root = Path(directory)
            digest = calibration.token_sha256(list(range(32)))
            payload = {
                "schema": capture.CAPTURE_SCHEMA,
                "target_model_id": "Qwen/Qwen3.8-27B",
                "source_config_sha256": "a" * 64,
                "source_index_sha256": OFFICIAL_INDEX_SHA256,
                "source_model_path": str(root),
                "calibration_sha256": "b" * 64,
                "stage": "post_conv_qk",
                "tp_size": 1,
                "old_head_k_dim": 4,
                "prefix": "model.layers.0.linear_attn",
                "tp_rank": 0,
                "capture_index": 0,
                "calibration_row_index": 0,
                "input_ids_sha256": digest,
                "local_key_dim": 8,
                "q": torch.randn(32, 8),
                "k": torch.randn(32, 8),
            }
            path = root / "capture.pt"
            arguments = {
                "expected_layer_ids": [0],
                "tp_size": 1,
                "num_heads": 2,
                "old_dim": 4,
                "new_dim": 2,
                "captures_per_rank": 1,
                "source_config_sha256": "a" * 64,
                "calibration_sha256": "b" * 64,
                "calibration_input_hashes": [digest],
            }
            torch.save(payload, path)
            rrqr_evidence = {
                "converged": True,
                "termination": "strong_rrqr_condition",
            }
            with mock.patch.object(
                selection,
                "strong_rrqr_indices",
                return_value=(np.array([0, 2]), rrqr_evidence),
            ):
                keeps, _ = selection.load_keep_maps(root, **arguments)
            self.assertEqual(keeps[0].tolist(), [0, 2, 4, 6])
            with mock.patch.object(
                selection,
                "strong_rrqr_indices",
                return_value=(
                    np.array([0, 2]),
                    {
                        "converged": False,
                        "termination": "max_swaps",
                    },
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "condition was not reached"):
                    selection.load_keep_maps(root, **arguments)
            for field, value in (
                ("schema", "drrqr-post-conv-qk/v1"),
                ("source_index_sha256", "c" * 64),
                ("calibration_row_index", 1),
                ("input_ids_sha256", "d" * 64),
            ):
                with self.subTest(field=field):
                    invalid = dict(payload)
                    invalid[field] = value
                    torch.save(invalid, path)
                    with self.assertRaises(ValueError):
                        selection.load_keep_maps(root, **arguments)

    def test_strong_rrqr_is_deterministic_and_bounded(self):
        generator = torch.Generator().manual_seed(7)
        values = torch.randn(6000, 8, generator=generator)
        first = selection.strong_rrqr_indices(values, 4, seed=42)
        second = selection.strong_rrqr_indices(values, 4, seed=42)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(len(set(first.tolist())), 4)
        self.assertTrue(all(0 <= value < 8 for value in first))

    def test_strong_rrqr_shared_rng_matches_official_legacy_sampling(self):
        generator = torch.Generator().manual_seed(17)
        first_head = torch.randn(6000, 8, generator=generator)
        second_head = torch.randn(6000, 8, generator=generator)
        shared = np.random.RandomState(42)
        first, first_evidence = selection.strong_rrqr_indices(
            first_head,
            4,
            rng=shared,
            return_evidence=True,
        )
        second, second_evidence = selection.strong_rrqr_indices(
            second_head,
            4,
            rng=shared,
            return_evidence=True,
        )
        reference = np.random.RandomState(42)
        first_rows = reference.choice(6000, 5000, replace=False)
        second_rows = reference.choice(6000, 5000, replace=False)
        first_reference = selection.strong_rrqr_indices(
            first_head[first_rows],
            4,
            seed=999,
        )
        second_reference = selection.strong_rrqr_indices(
            second_head[second_rows],
            4,
            seed=999,
        )
        self.assertTrue(np.array_equal(first, first_reference))
        self.assertTrue(np.array_equal(second, second_reference))
        self.assertTrue(first_evidence["subsampled"])
        self.assertTrue(second_evidence["subsampled"])

    def test_strong_rrqr_evidence_reports_convergence_state(self):
        values = torch.randn(32, 8)
        _, evidence = selection.strong_rrqr_indices(
            values,
            4,
            max_swaps=0,
            return_evidence=True,
        )
        self.assertIn(evidence["termination"], {"strong_rrqr_condition", "max_swaps"})
        self.assertEqual(
            evidence["converged"],
            evidence["termination"] == "strong_rrqr_condition",
        )

    def test_nonfinite_activation_is_rejected(self):
        values = torch.eye(4)
        values[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            selection.strong_rrqr_indices(values, 2)


class PrepareTests(unittest.TestCase):
    def qwen38_config(self):
        types_ = ["full_attention" if (layer + 1) % 4 == 0 else "linear_attention" for layer in range(64)]
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

    def test_official_qwen38_revision_hashes_are_pinned(self):
        validate_official_checkpoint_hashes(
            OFFICIAL_CONFIG_SHA256,
            OFFICIAL_INDEX_SHA256,
        )
        with self.assertRaisesRegex(ValueError, "official Qwen/Qwen3.8-27B"):
            validate_official_checkpoint_hashes("0" * 64, OFFICIAL_INDEX_SHA256)

    def test_exact_qwen38_hybrid_metadata_is_accepted(self):
        with tempfile.TemporaryDirectory(prefix="drror-source-") as directory:
            root = Path(directory)
            shard = root / "model.safetensors"
            shard.touch()
            config = self.qwen38_config()
            linear_layers = [
                index for index, kind in enumerate(config["text_config"]["layer_types"]) if kind == "linear_attention"
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
                            "event": "worker_dispatch_verified",
                            "component": "plugin",
                            "pid": pid,
                            "plan_sha256": digest,
                            "runtime_sources": {
                                "ascend_gdn": {"sha256": bootstrap.EXPECTED_GDN_SHA256},
                                "qwen_model": {"sha256": bootstrap.EXPECTED_QWEN_SHA256},
                            },
                        },
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
                            "query_shape": [1, 3, 4, 102],
                            "key_shape": [1, 3, 4, 102],
                            "initial_state_shape": [1, 12, 102, 128],
                            "prebuilt_metadata_forwarded": True,
                            "completed_python_call": True,
                        },
                        {
                            "schema": "drror-vllm-ascend-evidence/v1",
                            "event": "reduced_gdn_decode_branch",
                            "component": "gdn",
                            "pid": pid,
                            "plan_sha256": digest,
                            "key_head_dim": 102,
                            "value_head_dim": 128,
                            "backend": "npu_recurrent_gated_delta_rule",
                            "token_count": 1,
                            "query_shape": [1, 1, 4, 102],
                            "key_shape": [1, 1, 4, 102],
                            "state_shape": [8, 12, 128, 102],
                            "completed_python_call": True,
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
            for missing in ("worker_dispatch_verified", "reduced_gdn_hot_path", "reduced_gdn_decode_branch"):
                with self.subTest(missing=missing):
                    path.write_text(
                        "".join(
                            json.dumps(row) + "\n" for row in rows if not (row["event"] == missing and row["pid"] == 4)
                        ),
                        encoding="utf-8",
                    )
                    failed = audit.audit_activation(path, plan_sha256=digest, target_head_k_dim=102)
                    self.assertFalse(failed["ok"])
                    self.assertTrue(any("worker 4" in error for error in failed["errors"]))
            rows[2]["runtime_sources"]["ascend_gdn"]["sha256"] = "0" * 64
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            failed = audit.audit_activation(path, plan_sha256=digest, target_head_k_dim=102)
            self.assertFalse(failed["ok"])
            self.assertTrue(any("source hashes" in error for error in failed["errors"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
