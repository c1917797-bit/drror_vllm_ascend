#!/usr/bin/env python3
"""CPU-only stability audit for energy-kernel DRRQR coordinate selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

import torch


CAPTURE_RE = re.compile(
    r"^rank(?P<rank>\d+)_.*_layers_(?P<layer>\d+)_linear_attn_"
    r"capture(?P<capture>\d+)_.*\.pt$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def stable_topk(energy: torch.Tensor, count: int) -> torch.Tensor:
    # stable=True makes coordinate index the deterministic tie breaker.
    return torch.argsort(energy, descending=True, stable=True)[:count]


def overlap(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float]:
    common = len(set(left.tolist()) & set(right.tolist()))
    overlap_fraction = common / len(left)
    jaccard = common / (2 * len(left) - common)
    return overlap_fraction, jaccard


def index_captures(root: Path) -> dict[tuple[int, int], dict[int, Path]]:
    result: dict[tuple[int, int], dict[int, Path]] = {}
    for path in root.glob("*.pt"):
        match = CAPTURE_RE.match(path.name)
        if match is None:
            continue
        key = (int(match.group("layer")), int(match.group("rank")))
        capture = int(match.group("capture"))
        if capture in result.setdefault(key, {}):
            raise RuntimeError(f"duplicate capture {key=} {capture=}")
        result[key][capture] = path
    return result


def load_row_energy(paths: dict[int, Path]) -> tuple[torch.Tensor, dict]:
    indexes = sorted(paths)
    if indexes != list(range(len(indexes))):
        raise RuntimeError(f"non-contiguous capture indexes: {indexes}")
    energies = []
    binding = []
    old_dim = None
    local_heads = None
    for capture_index in indexes:
        payload = torch.load(paths[capture_index], map_location="cpu", weights_only=True)
        if int(payload["capture_index"]) != capture_index:
            raise RuntimeError(f"capture index mismatch: {paths[capture_index]}")
        q = payload["q"].float()
        k = payload["k"].float()
        candidate_dim = int(payload["old_head_k_dim"])
        candidate_heads = int(payload["local_key_dim"]) // candidate_dim
        if old_dim is None:
            old_dim, local_heads = candidate_dim, candidate_heads
        if candidate_dim != old_dim or candidate_heads != local_heads:
            raise RuntimeError(f"inconsistent capture shape: {paths[capture_index]}")
        q = q.view(-1, local_heads, old_dim)
        k = k.view(-1, local_heads, old_dim)
        q = torch.nn.functional.normalize(q, p=2.0, dim=-1, eps=1e-6)
        k = torch.nn.functional.normalize(k, p=2.0, dim=-1, eps=1e-6)
        energies.append(q.square().sum(dim=0) + k.square().sum(dim=0))
        binding.append(
            {
                "capture_index": capture_index,
                "input_ids_sha256": payload["input_ids_sha256"],
                "calibration_row_index": int(payload["calibration_row_index"]),
            }
        )
    return torch.stack(energies), {
        "captures": len(indexes),
        "local_heads": local_heads,
        "old_head_k_dim": old_dim,
        "binding": binding,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="0,12,25,38,50,62")
    parser.add_argument("--keep-dims", default="64,96")
    parser.add_argument("--folds", type=int, default=4)
    args = parser.parse_args()

    requested_layers = [int(value) for value in args.layers.split(",")]
    keep_dims = [int(value) for value in args.keep_dims.split(",")]
    index = index_captures(args.capture_dir)
    available_layers = sorted({layer for layer, _ in index})
    missing = sorted(set(requested_layers) - set(available_layers))
    if missing:
        raise RuntimeError(
            f"requested layers are absent: {missing}; available={available_layers}"
        )

    records = []
    binding_hashes: dict[int, str] = {}
    for layer in requested_layers:
        ranks = sorted(rank for candidate_layer, rank in index if candidate_layer == layer)
        for rank in ranks:
            row_energy, metadata = load_row_energy(index[(layer, rank)])
            rows, heads, old_dim = row_energy.shape
            if rows % args.folds:
                raise RuntimeError(f"{rows=} is not divisible by folds={args.folds}")
            current_binding = json.dumps(metadata["binding"], sort_keys=True)
            binding_hash = hashlib.sha256(current_binding.encode()).hexdigest()
            if layer in binding_hashes and binding_hashes[layer] != binding_hash:
                raise RuntimeError(f"calibration binding differs across ranks for layer {layer}")
            binding_hashes[layer] = binding_hash

            all_rows = list(range(rows))
            full_energy = row_energy.sum(dim=0)
            for keep_dim in keep_dims:
                if not 0 < keep_dim < old_dim:
                    raise ValueError(f"invalid keep dimension {keep_dim} for width {old_dim}")
                for head in range(heads):
                    full_keep = stable_topk(full_energy[head], keep_dim)
                    full_retained = float(
                        full_energy[head, full_keep].sum() / full_energy[head].sum()
                    )
                    for fold in range(args.folds):
                        heldout_rows = [row for row in all_rows if row % args.folds == fold]
                        train_rows = [row for row in all_rows if row % args.folds != fold]
                        train_energy = row_energy[train_rows, head].sum(dim=0)
                        heldout_energy = row_energy[heldout_rows, head].sum(dim=0)
                        train_keep = stable_topk(train_energy, keep_dim)
                        oracle_keep = stable_topk(heldout_energy, keep_dim)
                        train_full_overlap, train_full_jaccard = overlap(
                            train_keep, full_keep
                        )
                        train_hold_overlap, train_hold_jaccard = overlap(
                            train_keep, oracle_keep
                        )
                        heldout_retained = float(
                            heldout_energy[train_keep].sum() / heldout_energy.sum()
                        )
                        heldout_oracle = float(
                            heldout_energy[oracle_keep].sum() / heldout_energy.sum()
                        )
                        records.append(
                            {
                                "layer": layer,
                                "rank": rank,
                                "local_head": head,
                                "keep_dim": keep_dim,
                                "fold": fold,
                                "train_rows": train_rows,
                                "heldout_rows": heldout_rows,
                                "full_retained_energy_fraction": full_retained,
                                "train_vs_full_overlap_fraction": train_full_overlap,
                                "train_vs_full_jaccard": train_full_jaccard,
                                "train_vs_heldout_overlap_fraction": train_hold_overlap,
                                "train_vs_heldout_jaccard": train_hold_jaccard,
                                "heldout_retained_energy_fraction": heldout_retained,
                                "heldout_oracle_energy_fraction": heldout_oracle,
                                "heldout_oracle_regret_pp": 100.0
                                * (heldout_oracle - heldout_retained),
                            }
                        )

    summaries = {}
    for keep_dim in keep_dims:
        selected = [record for record in records if record["keep_dim"] == keep_dim]
        summaries[str(keep_dim)] = {
            name: summarize([float(record[name]) for record in selected])
            for name in (
                "full_retained_energy_fraction",
                "train_vs_full_overlap_fraction",
                "train_vs_full_jaccard",
                "train_vs_heldout_overlap_fraction",
                "train_vs_heldout_jaccard",
                "heldout_retained_energy_fraction",
                "heldout_oracle_energy_fraction",
                "heldout_oracle_regret_pp",
            )
        }

    primary = summaries.get("64")
    gates = {
        "primary_keep_dim": 64,
        "median_heldout_oracle_regret_pp_max": 1.0,
        "p95_heldout_oracle_regret_pp_max": 2.0,
    }
    if primary is None:
        decision = "diagnostic_only_no_dk64"
    elif (
        primary["heldout_oracle_regret_pp"]["median"] <= 1.0
        and primary["heldout_oracle_regret_pp"]["p95"] <= 2.0
    ):
        decision = "calibration_energy_selection_stable"
    else:
        decision = "calibration_energy_selection_instability_supported"

    report = {
        "schema": "drrqr-calibration-stability/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "four-fold heldout audit of joint normalized Q/K coordinate energy",
        "claim_boundary": (
            "This is an offline selector-stability diagnostic. It neither measures "
            "model accuracy nor supplies end-to-end performance evidence."
        ),
        "capture_dir": str(args.capture_dir),
        "capture_files_indexed": sum(len(paths) for paths in index.values()),
        "available_layers": available_layers,
        "audited_layers": requested_layers,
        "keep_dims": keep_dims,
        "folds": args.folds,
        "calibration_binding_sha256_by_layer": binding_hashes,
        "gates": gates,
        "decision": decision,
        "summaries": summaries,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "decision": decision,
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "summaries": summaries,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
