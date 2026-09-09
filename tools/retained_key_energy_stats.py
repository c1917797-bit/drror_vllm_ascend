#!/usr/bin/env python3
"""Describe real captured retained Q/K energy; not accuracy or recurrence replay."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import torch

from analyze_calibration_stability import index_captures
from drror_vllm_ascend.calibration import read_calibration
from drror_vllm_ascend.plan import DrrqrPlan


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def fractions(value, keep):
    if value.ndim != 3 or keep.ndim != 2 or value.shape[1] != keep.shape[0]:
        raise ValueError("expected token/head/coordinate tensors")
    if value.shape[-1] != 128 or not value.is_floating_point():
        raise ValueError("expected floating original Dk128")
    if not torch.isfinite(value).all():
        raise ValueError("nonfinite capture")
    if keep.dtype != torch.int64 or (keep < 0).any() or (keep >= 128).any():
        raise ValueError("invalid local coordinates")
    if any(len(set(row.tolist())) != row.numel() for row in keep):
        raise ValueError("duplicate local coordinates")
    square = value.double().square()
    full = square.sum(-1)
    if (full <= 0).any():
        raise ValueError("zero-energy row makes retained fraction undefined")
    selected = square.gather(-1, keep.unsqueeze(0).expand(value.shape[0], -1, -1)).sum(-1)
    result = selected / full
    if not torch.isfinite(result).all() or (result < 0).any() or (result > 1 + 1e-12).any():
        raise ValueError("invalid retained fraction")
    return result


def describe(values):
    values = values.double().reshape(-1)
    if not values.numel() or not torch.isfinite(values).all():
        raise ValueError("expected nonempty finite observations")
    quantiles = torch.quantile(values, torch.tensor([0., .05, .5, .95, .99, 1.], dtype=torch.float64))
    mean, std = values.mean().item(), values.std(correction=0).item()
    return dict(count=values.numel(), mean=mean, std=std,
                cv=std / mean if mean > 0 else None,
                **dict(zip(("min", "p05", "median", "p95", "p99", "max"), quantiles.tolist())))


def audit(capture_dir, plan_path, plan_sha256, calibration_path, layers):
    raw_plan = plan_path.read_bytes()
    if digest(raw_plan) != plan_sha256:
        raise ValueError("plan hash mismatch")
    plan = DrrqrPlan.load(str(plan_path), plan_sha256)
    if plan.target_head_k_dim != 64 or plan.num_key_heads != 16:
        raise ValueError("this bounded diagnostic requires Dk64 and 16 global key heads")
    document = json.loads(raw_plan)
    calibration_sha = document["provenance"]["calibration_sha256"]
    rows = read_calibration(calibration_path, calibration_sha)
    if len(rows) != 16:
        raise ValueError("expected the frozen 16-row calibration")
    index = index_captures(capture_dir)
    maps = dict(plan.keep_indices)
    if len(layers) != len(set(layers)) or not set(layers).issubset(maps):
        raise ValueError("invalid requested layer coverage")
    records, files = [], []
    for layer in layers:
        for rank in range(4):
            paths = index.get((layer, rank), {})
            if set(paths) != set(range(16)):
                raise ValueError(f"incomplete calibration: {layer=} {rank=}")
            all_keep = torch.tensor(maps[layer], dtype=torch.int64).reshape(16, 64)
            keep = all_keep[rank * 4:(rank + 1) * 4] - torch.arange(rank * 4, (rank + 1) * 4)[:, None] * 128
            qr, kr = [], []
            for row_id in range(16):
                raw = paths[row_id].read_bytes()
                data = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
                required = {
                    "schema": "drrqr-post-conv-qk/v2",
                    "target_model_id": "Qwen/Qwen3.8-27B",
                    "source_config_sha256": plan.source_config_sha256,
                    "source_index_sha256": plan.source_index_sha256,
                    "calibration_sha256": calibration_sha,
                    "stage": "post_conv_qk", "tp_size": 4, "tp_rank": rank,
                    "old_head_k_dim": 128, "local_key_dim": 512,
                    "capture_index": row_id, "calibration_row_index": row_id,
                    "input_ids_sha256": rows[row_id]["input_ids_sha256"],
                }
                if any(data.get(k) != v for k, v in required.items()):
                    raise ValueError(f"capture binding mismatch: {paths[row_id]}")
                if not str(data.get("prefix", "")).endswith(f".layers.{layer}.linear_attn"):
                    raise ValueError("layer prefix mismatch")
                q, k = data["q"], data["k"]
                if q.shape != k.shape or q.ndim != 2 or q.shape != (len(rows[row_id]["input_ids"]), 512):
                    raise ValueError("captured token/head shape differs")
                qr.append(fractions(q.reshape(-1, 4, 128), keep))
                kr.append(fractions(k.reshape(-1, 4, 128), keep))
                files.append({"name": paths[row_id].name, "sha256": digest(raw), "tokens": q.shape[0]})
            for head in range(4):
                q_all = torch.cat([x[:, head] for x in qr])
                k_all = torch.cat([x[:, head] for x in kr])
                k_train = torch.cat([x[:, head] for x in kr[:12]])
                k_hold = torch.cat([x[:, head] for x in kr[12:]])
                if (k_all <= 0).any():
                    raise ValueError("zero retained K energy is singular for state rescaling")
                mean_train = k_train.mean()
                records.append({
                    "layer": layer, "rank": rank, "local_head": head,
                    "global_head": rank * 4 + head,
                    "q_retained": describe(q_all), "k_retained": describe(k_all),
                    "qk_output_scale_ratio": describe(torch.sqrt(q_all / k_all)),
                    "train_k_mean": mean_train.item(),
                    "heldout_k_mean": k_hold.mean().item(),
                    "heldout_minus_train_k_mean": (k_hold.mean() - mean_train).item(),
                    "heldout_relative_to_train_deviation": describe((k_hold / mean_train - 1).abs()),
                })
    return {
        "schema": "drrqr-retained-key-energy-stats/v1",
        "claim_boundary": (
            "Descriptive calibration statistics only. Frozen coordinate selection used all 16 rows, "
            "so the 12/4 scalar split is not held-out selector validation or task generalization. "
            "Captures lack V/g/beta/state/teacher outputs: not actual recurrent replay. "
            "No NPU, accuracy or throughput claim."
        ),
        "definition": "sum(selected raw post-conv coordinates squared) / sum(all 128 coordinates squared); not a replacement for runtime epsilon/quantization semantics",
        "plan": {"path": str(plan_path), "sha256": plan_sha256, "objective": document.get("selection_objective")},
        "calibration": {"path": str(calibration_path), "sha256": calibration_sha,
                        "scalar_train_rows": list(range(12)), "scalar_holdout_rows": list(range(12, 16))},
        "capture_dir": str(capture_dir), "audited_layers": layers,
        "capture_files": files, "head_records": records,
        "summaries": {
            "head_mean_k_retention": describe(torch.tensor([x["k_retained"]["mean"] for x in records], dtype=torch.float64)),
            "head_k_retention_cv": describe(torch.tensor([x["k_retained"]["cv"] for x in records], dtype=torch.float64)),
            "head_holdout_relative_deviation_p95": describe(torch.tensor([x["heldout_relative_to_train_deviation"]["p95"] for x in records], dtype=torch.float64)),
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture-dir", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--plan-sha256", required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if any(not x.is_absolute() for x in (args.capture_dir, args.plan, args.calibration, args.output_dir)):
        p.error("absolute paths required")
    args.output_dir.mkdir(parents=False, exist_ok=False)
    report = {"schema": "drrqr-retained-key-energy-stats/v1", "status": "failed"}
    try:
        report = audit(args.capture_dir, args.plan, args.plan_sha256, args.calibration, [0, 12, 25, 38, 50, 62])
        report["status"] = "completed_descriptive_only"
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        report["script_sha256"] = digest(Path(__file__).read_bytes())
        with (args.output_dir / "report.json").open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"status": report["status"], "summaries": report.get("summaries")}, indent=2))


if __name__ == "__main__":
    main()
