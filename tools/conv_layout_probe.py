#!/usr/bin/env python3
"""Compare unchanged Ascend convolution with original versus load-time packed weights.

This validates the layout optimization, not DRRQR accuracy or model-level performance.
Run only in the audited container mapping physical NPU4 to logical NPU0.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch_npu

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from drror_vllm_ascend.patches import conv_layout
from vllm_ascend.utils import enable_custom_op


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensors(channels, batch, length, seed, has_bias):
    generator = torch.Generator().manual_seed(seed)
    def rand(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.bfloat16).to("npu:0")
    original = rand(channels, 1, 4)
    module = SimpleNamespace(weight=torch.nn.Parameter(original.clone(), requires_grad=False))
    layout = conv_layout.prepare_conv_weight_layout(module)
    w0 = original.view(channels, 4).T
    w1 = module.weight.view(channels, 4).T
    assert torch.equal(w0.cpu(), w1.cpu()) and w1.is_contiguous() and not w0.is_contiguous()
    state = rand(batch + 3, 3, channels)
    indices = torch.arange(batch, 0, -1, dtype=torch.int32, device="npu:0")
    starts = torch.arange(0, (batch + 1) * length, length, dtype=torch.int32, device="npu:0")
    return dict(x=rand(batch * length, channels), weights=(w0, w1),
                states=(state.clone(), state.clone()), bias=rand(channels) if has_bias else None,
                indices=indices, starts=starts, layout=layout)


def operation(data, variant, mode, initial):
    output = torch.empty_like(data["x"])
    torch.ops._C_ascend.npu_causal_conv1d_custom(
        output, data["x"], data["weights"][variant],
        conv_state=data["states"][variant], bias_opt=data["bias"],
        query_start_loc_opt=data["starts"], cache_indices_opt=data["indices"],
        initial_state_mode_opt=initial, num_accepted_tokens_opt=None,
        activation_mode=1, pad_slot_id=-1, run_mode=mode)
    return output


def exact(a, b):
    a, b = a.cpu(), b.cpu()
    return bool(torch.isfinite(a).all() and torch.isfinite(b).all() and torch.equal(a, b))


def check_case(channels, batch, length, phase, has_bias, initial_state, seed):
    data = tensors(channels, batch, length, seed, has_bias)
    initial = torch.full((batch,), initial_state, dtype=torch.bool, device="npu:0") if phase == "prefill" else None
    steps = 1 if phase == "prefill" else 32
    passed = []
    for step in range(steps):
        outputs = [operation(data, i, int(phase == "decode"), initial) for i in (0, 1)]
        torch.npu.synchronize()
        output_exact = exact(*outputs)
        state_exact = exact(*data["states"])
        passed.append({"step": step, "output_exact_finite": output_exact, "state_exact_finite": state_exact})
    return {"channels": channels, "batch": batch, "length": length, "phase": phase,
            "bias": has_bias, "initial_state": initial_state, "layout": data["layout"],
            "steps": passed, "passed": all(s["output_exact_finite"] and s["state_exact_finite"] for s in passed)}


def profile_layout(output_dir, variant):
    data = tensors(2048, 16, 1, 20260908, False)
    for _ in range(10):
        operation(data, variant, 1, None)
    torch.npu.synchronize()
    experimental = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization)
    profile = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU, torch_npu.profiler.ProfilerActivity.CPU],
        with_stack=False, record_shapes=False, profile_memory=False,
        experimental_config=experimental,
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=30, repeat=1, skip_first=0),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(output_dir)))
    with profile:
        for _ in range(30):
            operation(data, variant, 1, None)
            torch.npu.synchronize()
            profile.step()
    return str(output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if not str(args.output_dir).startswith("/drrqr-results/") or args.output_dir.exists():
        parser.error("choose a fresh /drrqr-results/ experiment directory")
    args.output_dir.mkdir(parents=True)
    torch.npu.set_device(0)
    enable_custom_op()
    records = []
    for dk in (128, 112, 96, 64):
        channels = 2 * 4 * dk + 12 * 128
        for batch, length in ((1, 128), (1, 32768), (16, 16)):
            for initial in (False, True):
                for bias in (False, True):
                    records.append(check_case(channels, batch, length, "prefill", bias, initial, dk + batch + length))
        for batch in (1, 16):
            for bias in (False, True):
                records.append(check_case(channels, batch, 1, "decode", bias, True, dk + batch))
        print(json.dumps({"dk": dk, "complete_cases": len(records), "passed": all(r["passed"] for r in records)}), flush=True)
    report = {"schema": "drrqr-conv-layout-probe/v1", "status": "parity_pass" if all(r["passed"] for r in records) else "parity_fail",
              "script_sha256": checksum(__file__), "module_path": conv_layout.__file__,
              "module_sha256": checksum(conv_layout.__file__), "device": "container npu:0 mapped to physical NPU4",
              "claim": "weight-layout operator equivalence only; model graph and end-to-end gains are unproven",
              "cases": records}
    (args.output_dir / "parity.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "parity_pass":
        raise RuntimeError("layout parity failed")
    profiles = {name: profile_layout(args.output_dir / name, index)
                for index, name in enumerate(("original", "packed"))}
    print(json.dumps({"status": report["status"], "profiles": profiles}), flush=True)


if __name__ == "__main__":
    main()
