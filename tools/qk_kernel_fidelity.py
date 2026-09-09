#!/usr/bin/env python3
"""Screen DRRQR plans against the normalized Q/K geometry used by GDN.

This is an in-calibration proxy, not a task-accuracy result.  It compares the
full-Dk and reduced-Dk normalized QK score matrices on the frozen post-conv
captures.  A candidate must improve this proxy before consuming NPU time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
CAPTURE_INDEX_RE = re.compile(r"_capture(\d+)_")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_capture_indices(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(sorted({int(item) for item in value.split(",")}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("capture indexes must be comma-separated integers") from error
    if not parsed or parsed[0] < 0:
        raise argparse.ArgumentTypeError("capture indexes must be non-negative")
    return parsed


def sample_rows(length: int, maximum: int) -> torch.Tensor:
    if length < 1 or maximum < 1:
        raise ValueError("length and maximum must be positive")
    if length <= maximum:
        return torch.arange(length)
    # Deterministic coverage of the complete prompt, including both endpoints.
    return torch.linspace(0, length - 1, steps=maximum).round().long().unique()


@dataclass
class Accumulator:
    count: int = 0
    abs_sum: float = 0.0
    square_sum: float = 0.0
    max_abs: float = 0.0
    q_energy_sum: float = 0.0
    k_energy_sum: float = 0.0
    energy_count: int = 0

    def update(
        self,
        reference: torch.Tensor,
        candidate: torch.Tensor,
        q_energy: torch.Tensor,
        k_energy: torch.Tensor,
    ) -> None:
        delta = (candidate - reference).double()
        self.count += delta.numel()
        self.abs_sum += delta.abs().sum().item()
        self.square_sum += delta.square().sum().item()
        self.max_abs = max(self.max_abs, delta.abs().max().item())
        self.q_energy_sum += q_energy.double().sum().item()
        self.k_energy_sum += k_energy.double().sum().item()
        self.energy_count += q_energy.numel()

    def result(self) -> dict[str, float | int]:
        if self.count == 0 or self.energy_count == 0:
            raise RuntimeError("no samples accumulated")
        return {
            "score_count": self.count,
            "qk_score_mae": self.abs_sum / self.count,
            "qk_score_rmse": math.sqrt(self.square_sum / self.count),
            "qk_score_max_abs": self.max_abs,
            "mean_q_energy_fraction": self.q_energy_sum / self.energy_count,
            "mean_k_energy_fraction": self.k_energy_sum / self.energy_count,
        }


def load_plan(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = (
        "old_head_k_dim",
        "target_head_k_dim",
        "num_key_heads",
        "keep_indices",
    )
    if not isinstance(data, dict) or any(name not in data for name in required):
        raise ValueError(f"invalid DRRQR plan: {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--tokens-per-capture", type=int, default=64)
    parser.add_argument(
        "--capture-indices",
        type=parse_capture_indices,
        default=parse_capture_indices("0,4,8,12"),
    )
    parser.add_argument("--l2norm-eps", type=float, default=1e-6)
    args = parser.parse_args()

    if args.tp_size < 1 or args.tokens_per_capture < 1 or args.l2norm_eps <= 0:
        parser.error("tp-size, tokens-per-capture, and l2norm-eps must be positive")
    plan_paths = [path.resolve() for path in args.plan]
    if len(plan_paths) != len(set(plan_paths)):
        parser.error("plan paths must be unique")
    plans = {path.name: load_plan(path) for path in plan_paths}
    first = next(iter(plans.values()))
    shared = {
        key: first[key]
        for key in ("old_head_k_dim", "target_head_k_dim", "num_key_heads")
    }
    for name, plan in plans.items():
        if any(plan[key] != value for key, value in shared.items()):
            raise ValueError(f"plan geometry mismatch: {name}")

    old_dim = int(shared["old_head_k_dim"])
    new_dim = int(shared["target_head_k_dim"])
    num_heads = int(shared["num_key_heads"])
    if num_heads % args.tp_size:
        raise ValueError("num_key_heads must divide tp_size")
    local_heads = num_heads // args.tp_size
    selected_capture_indexes = set(args.capture_indices)
    files = []
    coverage: set[tuple[int, int, int]] = set()
    for path in sorted(args.capture_dir.glob("*.pt")):
        filename_match = CAPTURE_INDEX_RE.search(path.name)
        if filename_match is None:
            raise ValueError(f"cannot identify capture index in {path}")
        if int(filename_match.group(1)) not in selected_capture_indexes:
            continue
        payload = torch.load(path, map_location="cpu", weights_only=True)
        capture_index = int(payload.get("capture_index", -1))
        if capture_index not in selected_capture_indexes:
            continue
        match = LAYER_RE.search(str(payload.get("prefix", "")))
        if match is None:
            raise ValueError(f"cannot identify layer in {path}")
        layer = int(match.group(1))
        rank = int(payload.get("tp_rank", -1))
        key = (layer, rank, capture_index)
        if rank not in range(args.tp_size) or key in coverage:
            raise ValueError(f"invalid or duplicate capture metadata: {path}")
        q, k = payload.get("q"), payload.get("k")
        if (
            not isinstance(q, torch.Tensor)
            or not isinstance(k, torch.Tensor)
            or q.shape != k.shape
            or q.ndim != 2
            or q.shape[1] != local_heads * old_dim
            or not torch.isfinite(q).all()
            or not torch.isfinite(k).all()
        ):
            raise ValueError(f"invalid Q/K capture: {path}")
        files.append((path, payload, layer, rank, capture_index))
        coverage.add(key)

    if not files:
        raise RuntimeError("no matching captures")
    layers = sorted({layer for _, _, layer, _, _ in files})
    expected = {
        (layer, rank, capture_index)
        for layer in layers
        for rank in range(args.tp_size)
        for capture_index in args.capture_indices
    }
    if coverage != expected:
        raise RuntimeError(
            f"capture coverage mismatch missing={sorted(expected - coverage)} "
            f"extra={sorted(coverage - expected)}"
        )
    expected_layers = set(map(int, first["keep_indices"]))
    if set(layers) != expected_layers:
        raise RuntimeError("capture layers do not match plan linear-attention layers")

    totals = {name: Accumulator() for name in plans}
    by_layer = {
        name: {layer: Accumulator() for layer in layers}
        for name in plans
    }
    overlap: dict[str, dict[str, float | int]] = {}
    reference_name = next(iter(plans))
    reference_plan = plans[reference_name]
    for name, plan in plans.items():
        intersections = 0
        denominator = 0
        identical_heads = 0
        for layer in layers:
            left = reference_plan["keep_indices"][str(layer)]
            right = plan["keep_indices"][str(layer)]
            for head in range(num_heads):
                left_head = set(left[head * new_dim : (head + 1) * new_dim])
                right_head = set(right[head * new_dim : (head + 1) * new_dim])
                intersections += len(left_head & right_head)
                denominator += new_dim
                identical_heads += left_head == right_head
        overlap[name] = {
            "reference_plan": reference_name,
            "keep_overlap_fraction": intersections / denominator,
            "identical_head_count": identical_heads,
            "head_count": len(layers) * num_heads,
        }

    for _, payload, layer, rank, _ in files:
        q = payload["q"].float().view(-1, local_heads, old_dim)
        k = payload["k"].float().view(-1, local_heads, old_dim)
        rows = sample_rows(q.shape[0], args.tokens_per_capture)
        q = q.index_select(0, rows)
        k = k.index_select(0, rows)
        q_norm = F.normalize(q, p=2.0, dim=-1, eps=args.l2norm_eps)
        k_norm = F.normalize(k, p=2.0, dim=-1, eps=args.l2norm_eps)
        reference_scores = torch.einsum("thd,shd->hts", q_norm, k_norm)
        for name, plan in plans.items():
            keep = plan["keep_indices"][str(layer)]
            scale_spec = plan.get("coordinate_scales")
            if scale_spec is None:
                q_scales = [1.0] * (num_heads * new_dim)
                k_scales = [1.0] * (num_heads * new_dim)
            else:
                q_scales = scale_spec["q"][str(layer)]
                k_scales = scale_spec["k"][str(layer)]
            local_keep = []
            local_q_scales = []
            local_k_scales = []
            for local_head in range(local_heads):
                global_head = rank * local_heads + local_head
                selected = keep[global_head * new_dim : (global_head + 1) * new_dim]
                local_keep.append(
                    torch.tensor(
                        [index - global_head * old_dim for index in selected],
                        dtype=torch.long,
                    )
                )
                scale_slice = slice(global_head * new_dim, (global_head + 1) * new_dim)
                local_q_scales.append(torch.tensor(q_scales[scale_slice], dtype=torch.float32))
                local_k_scales.append(torch.tensor(k_scales[scale_slice], dtype=torch.float32))
            q_selected = torch.stack(
                [q[:, head].index_select(-1, local_keep[head]) for head in range(local_heads)],
                dim=1,
            )
            k_selected = torch.stack(
                [k[:, head].index_select(-1, local_keep[head]) for head in range(local_heads)],
                dim=1,
            )
            q_selected = q_selected * torch.stack(local_q_scales, dim=0).unsqueeze(0)
            k_selected = k_selected * torch.stack(local_k_scales, dim=0).unsqueeze(0)
            q_energy = q_selected.square().sum(-1) / q.square().sum(-1).clamp_min(1e-24)
            k_energy = k_selected.square().sum(-1) / k.square().sum(-1).clamp_min(1e-24)
            candidate_scores = torch.einsum(
                "thd,shd->hts",
                F.normalize(q_selected, p=2.0, dim=-1, eps=args.l2norm_eps),
                F.normalize(k_selected, p=2.0, dim=-1, eps=args.l2norm_eps),
            )
            totals[name].update(reference_scores, candidate_scores, q_energy, k_energy)
            by_layer[name][layer].update(reference_scores, candidate_scores, q_energy, k_energy)

    report = {
        "schema": "drrqr-qk-kernel-fidelity/v1",
        "scope": (
            "In-calibration screening proxy over frozen post-conv Q/K captures. "
            "It is not a LongBench/GSM8K accuracy result and cannot establish the user target."
        ),
        "capture_dir": str(args.capture_dir.resolve()),
        "capture_indices": list(args.capture_indices),
        "tokens_per_capture": args.tokens_per_capture,
        "capture_file_count": len(files),
        "layer_count": len(layers),
        "tp_size": args.tp_size,
        "l2norm_eps": args.l2norm_eps,
        "metric": (
            "All-pairs per-head QK cosine score error after applying the runtime "
            "L2 normalization separately at full and reduced Dk."
        ),
        "plans": {
            name: {
                "path": str(path),
                "sha256": file_sha256(path),
                "selection_objective": plan.get("provenance", {})
                .get("selection", {})
                .get("selection_objective"),
                "coordinate_scaling": plan.get("coordinate_scales", {}).get("method"),
                "overlap": overlap[name],
                "aggregate": totals[name].result(),
                "by_layer": {
                    str(layer): accumulator.result()
                    for layer, accumulator in by_layer[name].items()
                },
            }
            for (name, plan), path in zip(plans.items(), plan_paths, strict=True)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "plans": {name: item["aggregate"] for name, item in report["plans"].items()}}, indent=2))


if __name__ == "__main__":
    main()
