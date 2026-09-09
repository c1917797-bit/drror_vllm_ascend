#!/usr/bin/env python3
"""Numerical parity gate for Qwen3.8 GDN reduced key dimensions on Ascend.

This is a diagnostic, not a performance benchmark.  It compares the v0.23
prefill and decode kernels against CPU float32 references at the TP4-local
Qwen3.8 shapes before any further quality or performance experiment.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F


SCHEMA = "drrqr-gdn-parity/v1"
H_QK = 4
H_V = 12
D_V = 128
PREFILL_RTOL = 1e-2
PREFILL_ATOL = 1e-2
DECODE_RTOL = 3e-3
DECODE_ATOL = 1e-2


def _csv_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _git_head(path: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _tensor_metrics(actual: torch.Tensor, reference: torch.Tensor, rtol: float, atol: float) -> dict:
    actual_cpu = actual.detach().to(torch.float32).cpu()
    reference_cpu = reference.detach().to(torch.float32).cpu()
    if actual_cpu.shape != reference_cpu.shape:
        return {
            "actual_shape": list(actual_cpu.shape),
            "reference_shape": list(reference_cpu.shape),
            "finite": False,
            "nonfinite": None,
            "max_abs": None,
            "p99_abs": None,
            "relative_l2": None,
            "allclose": False,
        }
    finite_mask = torch.isfinite(actual_cpu)
    reference_finite = torch.isfinite(reference_cpu)
    finite = bool(finite_mask.all() and reference_finite.all())
    nonfinite = int((~finite_mask).sum().item() + (~reference_finite).sum().item())
    if not finite:
        return {
            "actual_shape": list(actual_cpu.shape),
            "reference_shape": list(reference_cpu.shape),
            "finite": False,
            "nonfinite": nonfinite,
            "max_abs": None,
            "p99_abs": None,
            "relative_l2": None,
            "allclose": False,
        }
    diff = (actual_cpu - reference_cpu).abs()
    flat = diff.reshape(-1)
    max_abs = float(flat.max().item()) if flat.numel() else 0.0
    p99_abs = float(torch.quantile(flat, 0.99).item()) if flat.numel() else 0.0
    ref_norm = float(torch.linalg.vector_norm(reference_cpu).item())
    diff_norm = float(torch.linalg.vector_norm(actual_cpu - reference_cpu).item())
    relative_l2 = diff_norm / max(ref_norm, torch.finfo(torch.float32).eps)
    return {
        "actual_shape": list(actual_cpu.shape),
        "reference_shape": list(reference_cpu.shape),
        "finite": True,
        "nonfinite": 0,
        "max_abs": max_abs,
        "p99_abs": p99_abs,
        "relative_l2": relative_l2,
        "allclose": bool(torch.allclose(actual_cpu, reference_cpu, rtol=rtol, atol=atol)),
    }


def _cpu_l2norm(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    value_fp32 = value.to(torch.float32)
    normalized = value_fp32 * torch.rsqrt(value_fp32.square().sum(dim=-1, keepdim=True) + eps)
    return normalized.to(value.dtype)


def _decode_golden(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    state: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    actual_seq_lengths: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    g: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official v0.23 recurrent-gated-delta-rule reference, non-speculative."""
    q = query.to(torch.float32) * scale
    k = key.to(torch.float32)
    v = value.to(torch.float32)
    updated_state = state.clone().to(torch.float32)
    total_tokens, value_heads, _ = v.shape
    key_heads = q.shape[-2]
    gate = (
        torch.ones(total_tokens, value_heads, dtype=torch.float32)
        if g is None
        else g.to(torch.float32).exp()
    )
    beta_fp32 = (
        torch.ones(total_tokens, value_heads, dtype=torch.float32)
        if beta is None
        else beta.to(torch.float32)
    )
    output = torch.empty_like(v, dtype=torch.float32)
    seq_start = 0
    for seq_length in actual_seq_lengths.tolist():
        recurrent = updated_state[int(ssm_state_indices[seq_start])]
        for head_id in range(value_heads):
            head_state = recurrent[head_id]
            qk_head = head_id // (value_heads // key_heads)
            for token_id in range(seq_start, seq_start + int(seq_length)):
                q_i = q[token_id, qk_head]
                k_i = k[token_id, qk_head]
                v_i = v[token_id, head_id]
                head_state = head_state * gate[token_id, head_id]
                prediction = (head_state * k_i.unsqueeze(-2)).sum(dim=-1)
                residual = (v_i - prediction) * beta_fp32[token_id, head_id]
                head_state = head_state + residual[:, None] * k_i[None, :]
                updated_state[int(ssm_state_indices[token_id]), head_id] = head_state
                output[token_id, head_id] = (head_state * q_i.unsqueeze(-2)).sum(dim=-1)
        seq_start += int(seq_length)
    return output.to(query.dtype), updated_state


def _append_error(
    cases: list[dict],
    *,
    case_id: str,
    phase: str,
    dk: int,
    seed: int,
    metadata: dict,
    error: BaseException,
) -> None:
    cases.append(
        {
            "case_id": case_id,
            "phase": phase,
            "dk": dk,
            "seed": seed,
            **metadata,
            "output": None,
            "state": None,
            "passed": False,
            "error": "%s: %s" % (type(error).__name__, error),
            "traceback": traceback.format_exc(),
        }
    )


def _run_decode_sequences(
    *,
    device: torch.device,
    dims: list[int],
    seeds: list[int],
    batches: list[int],
    steps: int,
    cases: list[dict],
) -> None:
    op = torch.ops._C_ascend.npu_recurrent_gated_delta_rule
    for dk in dims:
        for seed in seeds:
            for batch in batches:
                generator = torch.Generator(device="cpu").manual_seed(seed * 100000 + dk * 100 + batch)
                state_reference = torch.randn(
                    (batch, H_V, D_V, dk),
                    generator=generator,
                    dtype=torch.float32,
                ) * 0.01
                state_npu = state_reference.to(device)
                for step in range(steps):
                    case_id = "decode-dk%d-seed%d-batch%d-step%d" % (dk, seed, batch, step + 1)
                    metadata = {
                        "input_shapes": {
                            "q_k": [batch, H_QK, dk],
                            "v": [batch, H_V, D_V],
                            "state": [batch, H_V, D_V, dk],
                        },
                        "state_dtype": "float32",
                        "state_layout_dut": "N,Nv,Dv,Dk",
                        "state_layout_reference": "N,Nv,Dv,Dk",
                        "tolerances": {"rtol": DECODE_RTOL, "atol": DECODE_ATOL},
                    }
                    try:
                        query = F.normalize(
                            torch.randn((batch, H_QK, dk), generator=generator, dtype=torch.float32),
                            p=2,
                            dim=-1,
                        ).to(torch.bfloat16)
                        key = F.normalize(
                            torch.randn((batch, H_QK, dk), generator=generator, dtype=torch.float32),
                            p=2,
                            dim=-1,
                        ).to(torch.bfloat16)
                        value = (
                            torch.randn((batch, H_V, D_V), generator=generator, dtype=torch.float32) * 0.1
                        ).to(torch.bfloat16)
                        g = F.logsigmoid(
                            torch.randn((batch, H_V), generator=generator, dtype=torch.float32) * 0.25 + 2.0
                        )
                        beta = torch.sigmoid(
                            torch.randn((batch, H_V), generator=generator, dtype=torch.float32)
                        ).to(torch.bfloat16)
                        seq_lengths = torch.ones(batch, dtype=torch.int32)
                        state_indices = torch.arange(batch, dtype=torch.int32)
                        scale = dk**-0.5
                        output_reference, next_state_reference = _decode_golden(
                            query,
                            key,
                            value,
                            state_reference,
                            beta,
                            scale,
                            seq_lengths,
                            state_indices,
                            g,
                        )
                        actual_seq_lengths_npu = torch.cat(
                            (torch.zeros(1, dtype=torch.int32), seq_lengths)
                        ).to(device)
                        output_npu = op(
                            query=query.to(device),
                            key=key.to(device),
                            value=value.to(device),
                            g=g.to(device),
                            beta=beta.to(device),
                            state=state_npu,
                            scale=scale,
                            actual_seq_lengths=actual_seq_lengths_npu,
                            ssm_state_indices=state_indices.to(device),
                        )
                        torch.npu.synchronize()
                        output_metrics = _tensor_metrics(
                            output_npu,
                            output_reference,
                            DECODE_RTOL,
                            DECODE_ATOL,
                        )
                        state_metrics = _tensor_metrics(
                            state_npu,
                            next_state_reference,
                            DECODE_RTOL,
                            DECODE_ATOL,
                        )
                        passed = bool(output_metrics["allclose"] and state_metrics["allclose"])
                        cases.append(
                            {
                                "case_id": case_id,
                                "phase": "decode",
                                "dk": dk,
                                "seed": seed,
                                **metadata,
                                "output": output_metrics,
                                "state": state_metrics,
                                "passed": passed,
                                "error": None,
                            }
                        )
                        state_reference = next_state_reference
                    except BaseException as error:
                        _append_error(
                            cases,
                            case_id=case_id,
                            phase="decode",
                            dk=dk,
                            seed=seed,
                            metadata=metadata,
                            error=error,
                        )
                        break
                del state_npu
                gc.collect()
                torch.npu.empty_cache()


def _build_prefill_metadata(device: torch.device, cu_seqlens_cpu: torch.Tensor):
    from vllm_ascend.ops.gdn_attn_builder import _build_non_spec_chunked_prefill_metadata

    builder = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(linear_num_value_heads=48)
            ),
            parallel_config=SimpleNamespace(tensor_parallel_size=4),
        )
    )
    return _build_non_spec_chunked_prefill_metadata(builder, cu_seqlens_cpu, device)


def _run_prefill_cases(
    *,
    device: torch.device,
    dims: list[int],
    seeds: list[int],
    lengths: list[int],
    cases: list[dict],
) -> None:
    from vllm_ascend._310p.ops.fla.chunk_gated_delta_rule import (
        chunk_gated_delta_rule_pytorch,
    )
    from vllm_ascend.ops.triton.fla import chunk as chunk_module

    cu_values = [0]
    for length in lengths:
        cu_values.append(cu_values[-1] + length)
    cu_cpu = torch.tensor(cu_values, dtype=torch.int64)
    total_tokens = cu_values[-1]
    for dk in dims:
        for seed in seeds:
            case_id = "prefill-dk%d-seed%d-lengths-%s" % (
                dk,
                seed,
                "-".join(str(item) for item in lengths),
            )
            metadata = {
                "input_shapes": {
                    "q_k": [1, total_tokens, H_QK, dk],
                    "v": [1, total_tokens, H_V, D_V],
                    "state": [len(lengths), H_V, dk, D_V],
                },
                "cu_seqlens": cu_values,
                "state_dtype": "float32",
                "l2norm_mode": "DUT runtime; CPU explicit sum-square normalization",
                "state_layout_dut": "N,Nv,Dk,Dv",
                "state_layout_reference": "N,Nv,Dv,Dk",
                "tolerances": {"rtol": PREFILL_RTOL, "atol": PREFILL_ATOL},
            }
            try:
                generator = torch.Generator(device="cpu").manual_seed(seed * 100000 + dk)
                query = torch.randn(
                    (1, total_tokens, H_QK, dk),
                    generator=generator,
                    dtype=torch.float32,
                ).to(torch.bfloat16)
                key = torch.randn(
                    (1, total_tokens, H_QK, dk),
                    generator=generator,
                    dtype=torch.float32,
                ).to(torch.bfloat16)
                value = (
                    torch.randn(
                        (1, total_tokens, H_V, D_V),
                        generator=generator,
                        dtype=torch.float32,
                    )
                    * 0.1
                ).to(torch.bfloat16)
                g = F.logsigmoid(
                    torch.randn(
                        (1, total_tokens, H_V),
                        generator=generator,
                        dtype=torch.float32,
                    )
                    * 0.25
                    + 2.0
                )
                beta = torch.sigmoid(
                    torch.randn(
                        (1, total_tokens, H_V),
                        generator=generator,
                        dtype=torch.float32,
                    )
                ).to(torch.bfloat16)
                state_dut = torch.randn(
                    (len(lengths), H_V, dk, D_V),
                    generator=generator,
                    dtype=torch.float32,
                ) * 0.01
                state_reference = state_dut.transpose(-1, -2).contiguous()
                query_reference = _cpu_l2norm(query)
                key_reference = _cpu_l2norm(key)
                scale = dk**-0.5
                output_reference, final_state_reference = chunk_gated_delta_rule_pytorch(
                    q=query_reference,
                    k=key_reference,
                    v=value,
                    g=g,
                    beta=beta,
                    scale=scale,
                    initial_state=state_reference,
                    output_final_state=True,
                    cu_seqlens=cu_cpu,
                    head_first=False,
                    use_qk_l2norm_in_kernel=False,
                )
                prebuilt_meta = _build_prefill_metadata(device, cu_cpu)
                with (
                    patch.object(
                        chunk_module,
                        "get_forward_context",
                        return_value=SimpleNamespace(attn_metadata=None),
                    ),
                    patch.object(
                        chunk_module,
                        "get_pcp_group",
                        return_value=SimpleNamespace(world_size=1),
                    ),
                ):
                    output_npu, final_state_npu = chunk_module.chunk_gated_delta_rule(
                        q=query.to(device),
                        k=key.to(device),
                        v=value.to(device),
                        g=g.to(device),
                        beta=beta.to(device),
                        scale=scale,
                        initial_state=state_dut.to(device),
                        output_final_state=True,
                        cu_seqlens=cu_cpu.to(device),
                        prebuilt_meta=prebuilt_meta,
                        head_first=False,
                        use_qk_l2norm_in_kernel=True,
                    )
                torch.npu.synchronize()
                output_metrics = _tensor_metrics(
                    output_npu,
                    output_reference,
                    PREFILL_RTOL,
                    PREFILL_ATOL,
                )
                expected_state = final_state_reference.transpose(-1, -2).contiguous()
                state_metrics = _tensor_metrics(
                    final_state_npu,
                    expected_state,
                    PREFILL_RTOL,
                    PREFILL_ATOL,
                )
                passed = bool(output_metrics["allclose"] and state_metrics["allclose"])
                cases.append(
                    {
                        "case_id": case_id,
                        "phase": "prefill",
                        "dk": dk,
                        "seed": seed,
                        **metadata,
                        "output": output_metrics,
                        "state": state_metrics,
                        "passed": passed,
                        "error": None,
                    }
                )
            except BaseException as error:
                _append_error(
                    cases,
                    case_id=case_id,
                    phase="prefill",
                    dk=dk,
                    seed=seed,
                    metadata=metadata,
                    error=error,
                )
            gc.collect()
            torch.npu.empty_cache()


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _summarize(report: dict) -> int:
    cases = report["cases"]
    failed = [case["case_id"] for case in cases if not case["passed"]]
    dk128 = [case for case in cases if case["dk"] == 128]
    if not dk128 or any(not case["passed"] for case in dk128):
        verdict = "HARNESS_INVALID"
        exit_code = 2
    elif failed:
        verdict = "FAIL"
        exit_code = 1
    else:
        verdict = "PASS"
        exit_code = 0
    report["verdict"] = verdict
    report["complete"] = True
    report["summary"] = {
        "total": len(cases),
        "passed": len(cases) - len(failed),
        "failed": len(failed),
        "failed_case_ids": failed,
    }
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dks", type=_csv_ints, default=[128, 104, 88, 64])
    parser.add_argument("--seeds", type=_csv_ints, default=[42, 111])
    parser.add_argument("--prefill-lengths", type=_csv_ints, default=[63, 64, 65])
    parser.add_argument("--decode-batches", type=_csv_ints, default=[1, 16])
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--phases", choices=("both", "prefill", "decode"), default="both")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-id", default=os.environ.get("DRRQR_IMAGE_ID", "unknown"))
    parser.add_argument("--physical-npu", type=int, default=4)
    args = parser.parse_args()

    import torch_npu
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
    from vllm_ascend.utils import enable_custom_op

    torch_npu.npu.set_compile_mode(jit_compile=False)
    if not enable_custom_op():
        raise RuntimeError("vLLM-Ascend custom operators were not loaded")
    device = torch.device(args.device)
    torch.npu.set_device(device)
    init_device_properties_triton()
    if not hasattr(torch.ops._C_ascend, "npu_recurrent_gated_delta_rule"):
        raise RuntimeError("npu_recurrent_gated_delta_rule is unavailable")

    report = {
        "schema": SCHEMA,
        "complete": False,
        "verdict": "RUNNING",
        "created_unix_ns": time.time_ns(),
        "protocol": {
            "dims": args.dks,
            "seeds": args.seeds,
            "prefill_lengths": args.prefill_lengths,
            "decode_batches": args.decode_batches,
            "decode_steps": args.decode_steps,
            "phases": args.phases,
            "tp4_local_shape": {"query_key_heads": H_QK, "value_heads": H_V, "value_dim": D_V},
            "interpretation": {
                "dk128_failure": "HARNESS_INVALID",
                "dk128_and_dk64_pass_104_or_88_fail": "direct evidence of reduced-shape numerical defect",
                "all_dims_pass": "escalate to packed projection, causal conv, rearrange and cache parity",
            },
        },
        "provenance": {
            "hostname": platform.node(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_npu": getattr(torch_npu, "__version__", "unknown"),
            "vllm": _package_version("vllm"),
            "vllm_ascend": _package_version("vllm-ascend"),
            "plugin": _package_version("drror-vllm-ascend-plugin"),
            "plugin_commit": _git_head("/cache/cch/drror_vllm_ascend"),
            "vllm_ascend_source_commit": _git_head("/cache/austinov/src/vllm-ascend"),
            "image_id": args.image_id,
            "physical_npu": args.physical_npu,
            "container_device": args.device,
            "device_name": torch.npu.get_device_name(device),
            "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        },
        "cases": [],
        "summary": None,
    }
    _write_report(args.output, report)

    if args.phases in ("both", "decode"):
        _run_decode_sequences(
            device=device,
            dims=args.dks,
            seeds=args.seeds,
            batches=args.decode_batches,
            steps=args.decode_steps,
            cases=report["cases"],
        )
        _write_report(args.output, report)
    if args.phases in ("both", "prefill"):
        _run_prefill_cases(
            device=device,
            dims=args.dks,
            seeds=args.seeds,
            lengths=args.prefill_lengths,
            cases=report["cases"],
        )
        _write_report(args.output, report)

    exit_code = _summarize(report)
    report["completed_unix_ns"] = time.time_ns()
    _write_report(args.output, report)
    print(json.dumps({"output": str(args.output), "summary": report["summary"], "verdict": report["verdict"]}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
