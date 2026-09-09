#!/usr/bin/env python3
"""Deterministically export the eight raw torch-NPU trace directories."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import torch_npu
from torch_npu.profiler.profiler import analyse


REQUIRED = ("kernel_details.csv", "op_statistic.csv", "api_statistic.csv", "trace_view.json")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.service_dir.resolve()
    manifest_path = directory / "service-manifest.json"
    client_path = directory / "profile-client.json"
    manifest = json.loads(manifest_path.read_text())
    client = json.loads(client_path.read_text())
    if (client.get("status") != "requests_complete_activation_validated"
            or client.get("service_manifest_sha256") != digest(manifest_path)
            or manifest.get("profiler_config", {}).get("max_iterations") != 30):
        raise RuntimeError("profile artifacts are not bound to the frozen service")
    traces = []
    for phase in ("prefill", "decode"):
        values = client["results"][phase]["trace_dirs"]
        if len(values) != 4:
            raise RuntimeError("exactly four trace directories required for " + phase)
        for value in values:
            path = (directory / value).resolve()
            if path.parent != (directory / "traces").resolve():
                raise RuntimeError("trace escaped the service trace directory")
            match = re.fullmatch(r"rank([0-3])_\d+_\d+_ascend_pt", path.name)
            if not match:
                raise RuntimeError("unexpected trace directory name: " + path.name)
            traces.append((phase, int(match[1]), path))
    if len({path for _, _, path in traces}) != 8:
        raise RuntimeError("prefill/decode trace directories overlap")
    if {rank for phase, rank, _ in traces if phase == "prefill"} != {0, 1, 2, 3}:
        raise RuntimeError("prefill rank coverage differs")
    if {rank for phase, rank, _ in traces if phase == "decode"} != {0, 1, 2, 3}:
        raise RuntimeError("decode rank coverage differs")

    records = []
    for phase, rank, path in traces:
        output = path / "ASCEND_PROFILER_OUTPUT"
        before = all((output / name).is_file() for name in REQUIRED)
        if not before:
            analyse(str(path), max_process_number=1)
        missing = [name for name in REQUIRED if not (output / name).is_file()]
        if missing:
            raise RuntimeError(f"{path.name}: export missing {missing}")
        records.append({
            "phase": phase,
            "rank": rank,
            "trace_dir": str(path.relative_to(directory)),
            "already_complete": before,
            "sha256": {name: digest(output / name) for name in REQUIRED},
        })
    receipt = directory / "profile-export.json"
    if receipt.exists():
        raise RuntimeError("preserve previous export receipt")
    receipt.write_text(json.dumps({
        "schema": "drrqr-profile-export/v1",
        "status": "eight_traces_exported",
        "service_manifest_sha256": digest(manifest_path),
        "profile_client_sha256": digest(client_path),
        "script_sha256": digest(Path(__file__)),
        "records": records,
    }, indent=2) + "\n")
    print(json.dumps({"status": "eight_traces_exported", "records": len(records)}), flush=True)


if __name__ == "__main__":
    main()
