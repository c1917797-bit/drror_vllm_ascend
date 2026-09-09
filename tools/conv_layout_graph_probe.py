#!/usr/bin/env python3
"""Bounded conv-layout graph replay gate; not full-model accuracy/performance."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import sysconfig
from pathlib import Path

from conv_layout_probe import checksum, conv_layout, enable_custom_op, exact, operation, tensors, torch, torch_npu

DKS = (64, 128)
BATCHES = (1, 2, 4, 8, 16, 24, 32)
STEPS = 32


def validate(records):
    expected = {(dk, batch) for dk in DKS for batch in BATCHES}
    keys = [(r["dk"], r["batch"]) for r in records]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("missing or duplicate graph cases")
    for record in records:
        if len(record["steps"]) != STEPS:
            raise ValueError("incomplete graph replay")
        for index, step in enumerate(record["steps"]):
            if step["step"] != index or not all(step[name] is True for name in (
                "graph_original_output", "graph_packed_output", "eager_packed_output",
                "graph_original_state", "graph_packed_state", "eager_packed_state",
                "storage_pointers_unchanged",
            )):
                raise ValueError("graph/eager exact-finite comparison failed")
    return "graph_operator_parity_pass"


def case(dk, batch):
    data = tensors(8 * dk + 1536, batch, 1, 20260908 + dk + batch, False)
    initial_state = data["states"][0].clone()
    # Four independent state chains: eager original/packed; graphed original/packed.
    eager = dict(data, states=(initial_state.clone(), initial_state.clone()))
    graphs = [torch.npu.NPUGraph(), torch.npu.NPUGraph()]
    graph_outputs = []
    for variant in (0, 1):
        for _ in range(3):
            operation(data, variant, 1, None)
        torch.npu.synchronize()
        with torch.npu.graph(graphs[variant]):
            result = operation(data, variant, 1, None)
        graph_outputs.append(result)
        torch.npu.synchronize()
    # Capture/warmup may mutate state. Restore values without replacing captured storage.
    for state in (*data["states"], *eager["states"]):
        state.copy_(initial_state)
    torch.npu.synchronize()
    tracked = [data["x"], data["indices"], *data["weights"], *data["states"]]
    pointers = [value.data_ptr() for value in tracked]
    generator = torch.Generator().manual_seed(917 + dk + batch)
    steps = []
    for step in range(STEPS):
        next_x = torch.randn(data["x"].shape, dtype=torch.bfloat16, generator=generator)
        data["x"].copy_(next_x)
        # Change metadata in place, including different active slots; never alias states.
        indices = ((torch.arange(batch, dtype=torch.int32) + step) % (batch + 3)).to("npu:0")
        data["indices"].copy_(indices)
        eager_outputs = [operation(eager, variant, 1, None) for variant in (0, 1)]
        for graph in graphs:
            graph.replay()
        torch.npu.synchronize()
        checks = {
            "step": step,
            "graph_original_output": exact(eager_outputs[0], graph_outputs[0]),
            "graph_packed_output": exact(eager_outputs[0], graph_outputs[1]),
            "eager_packed_output": exact(eager_outputs[0], eager_outputs[1]),
            "graph_original_state": exact(eager["states"][0], data["states"][0]),
            "graph_packed_state": exact(eager["states"][0], data["states"][1]),
            "eager_packed_state": exact(eager["states"][0], eager["states"][1]),
            "storage_pointers_unchanged": pointers == [value.data_ptr() for value in tracked],
        }
        steps.append(checks)
        if not all(value is True for key, value in checks.items() if key != "step"):
            break
    for graph in graphs:
        graph.reset()
    return {"dk": dk, "batch": batch, "channels": 8 * dk + 1536,
            "bias": False, "layout": data["layout"], "steps": steps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--build-manifest", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if Path("/drrqr-results") not in output.parents or output.exists():
        parser.error("fresh directory below /drrqr-results required")
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "0,1,2,3":
        parser.error("requires audited container physical4-7 to logical0-3 mapping")
    build = json.loads(args.build_manifest.read_text())
    if checksum(build["wheel"]) != build["wheel_sha256"]:
        raise RuntimeError("wheel digest mismatch")
    # The reused helper adds the source tree to sys.path; its stale egg-info is
    # not the installed wheel. Resolve metadata only from the runtime purelib.
    installed = [d for d in importlib.metadata.distributions(path=[sysconfig.get_path("purelib")])
                 if d.metadata.get("Name") == "drror-vllm-ascend-plugin"]
    if len(installed) != 1:
        raise RuntimeError("expected one installed plugin distribution")
    distribution = installed[0]
    if distribution.version != "0.1.7":
        raise RuntimeError("wrong installed plugin")
    for name, expected in build["source_sha256"].items():
        if name.startswith("drror_vllm_ascend/"):
            if checksum(Path(build["source_root"]) / name) != expected or checksum(distribution.locate_file(name)) != expected:
                raise RuntimeError("installed/source plugin mismatch: " + name)
    output.mkdir(parents=True)
    protocol = {"dks": DKS, "batches": BATCHES, "steps": STEPS, "dtype": "bfloat16",
                "bias": False, "changing_inputs_and_cache_indices": True,
                "compare": "each graph and eager packed vs eager original; output and entire state",
                "tolerance": "finite and bitwise equal", "graph": "torch.npu.NPUGraph",
                "budget": "one frozen matrix; no automatic rerun or parameter search"}
    report = {"schema": "drrqr-conv-layout-graph/v1", "status": "running",
              "protocol": protocol, "script_sha256": checksum(__file__),
              "helper_sha256": checksum(Path(__file__).with_name("conv_layout_probe.py")),
              "module_sha256": checksum(conv_layout.__file__),
              "wheel_sha256": build["wheel_sha256"],
              "torch_version": torch.__version__, "torch_npu_version": torch_npu.__version__,
              "graph_api_sha256": checksum(Path(torch_npu.__file__).parent / "npu/graphs.py"),
              "device": "logical npu:0, audited physical NPU4",
              "limits": "isolated convolution graph, synthetic inputs/weights; not TP/full-model logit parity or task accuracy",
              "cases": []}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    try:
        torch.npu.set_device(0)
        enable_custom_op()
        for dk in DKS:
            for batch in BATCHES:
                record = case(dk, batch)
                report["cases"].append(record)
                (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({"dk": dk, "batch": batch, "steps": len(record["steps"]),
                                  "exact": all(all(v is True for k, v in s.items() if k != "step") for s in record["steps"])}), flush=True)
                if len(record["steps"]) != STEPS or not all(
                    all(v is True for k, v in s.items() if k != "step") for s in record["steps"]
                ):
                    raise RuntimeError("numerical difference; preserve first failure and stop")
        report["status"] = validate(report["cases"])
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "report": str(output / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
