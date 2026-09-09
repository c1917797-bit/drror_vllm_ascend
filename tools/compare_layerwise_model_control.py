#!/usr/bin/env python3
"""Compare legacy64 and layerwise-all64 model controls, not performance/accuracy."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time

RESULT_ROOT = Path("/drrqr-results")
LAYERS = tuple(i for i in range(64) if i % 4 != 3)
DIMS = {str(i): 64 for i in LAYERS}
HASH = re.compile(r"^[0-9a-f]{64}$")
MAX_LOGPROB_ABS = 0.02
MEAN_LOGPROB_ABS = 0.005
COMMON = {"conv_layout": "packed", "prefill_mc2": True, "prefill_mc2_mixed": True}
PROTOCOL = {"tensor_parallel_size": 4, "pipeline_parallel_size": 1,
            "enable_prefix_caching": False, "speculative_config": None,
            "kv_transfer_config": None, "weight_transfer_config": None,
            "dtype": "torch.bfloat16"}
CRITERIA = {"prompt_hashes_exact": True, "token_ids_exact": True,
            "selected_logprob_max_abs": MAX_LOGPROB_ABS,
            "selected_logprob_mean_abs": MEAN_LOGPROB_ABS}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def checksum(value):
    return isinstance(value, str) and HASH.fullmatch(value) is not None


def hash_map(value):
    return isinstance(value, dict) and bool(value) and all(
        isinstance(k, str) and checksum(v) for k, v in value.items())


def read_report(path, inputs):
    path = Path(path).resolve()
    raw = path.read_bytes()
    inputs[str(path)] = hashlib.sha256(raw).hexdigest()
    document = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError("nonfinite JSON constant: " + value)))
    require(isinstance(document, dict), "JSON report is not an object: " + str(path))
    return document


def fields(document, names, context):
    missing = set(names) - set(document)
    require(not missing, context + ": missing fields " + repr(sorted(missing)))
    return {name: document[name] for name in names}


def tensor_contract(tensor, rank, context, parameter=False):
    names = ("shape", "stride", "dtype", "device", "numel", "element_size",
             "storage_offset", "storage_nbytes", "contiguous", "data_ptr", "storage_ptr")
    value = fields(tensor, names, context)
    shape, stride = value["shape"], value["stride"]
    require(isinstance(shape, list) and bool(shape) and all(integer(n, 1) for n in shape),
            context + ": invalid shape")
    require(isinstance(stride, list) and len(shape) == len(stride)
            and all(integer(n) for n in stride), context + ": invalid stride")
    require(value["device"] == "npu:" + str(rank), context + ": wrong device")
    require(type(value["contiguous"]) is bool, context + ": missing contiguity")
    require(integer(value["element_size"], 1) and integer(value["numel"], 1)
            and value["numel"] == math.prod(shape), context + ": invalid tensor size")
    require(integer(value["storage_offset"]) and integer(value["storage_nbytes"], 1)
            and integer(value["data_ptr"], 1) and integer(value["storage_ptr"], 1),
            context + ": invalid storage metadata")
    require(value["data_ptr"] == value["storage_ptr"] +
            value["storage_offset"] * value["element_size"], context + ": bad pointer offset")
    extent = value["storage_offset"] + 1 + sum((n - 1) * s for n, s in zip(shape, stride))
    require(extent * value["element_size"] <= value["storage_nbytes"],
            context + ": tensor exceeds storage")
    if parameter:
        require(isinstance(tensor.get("name"), str) and tensor["name"]
                and checksum(tensor.get("sha256")), context + ": invalid parameter identity")
        value.update(name=tensor["name"], sha256=tensor["sha256"])
    return {k: v for k, v in value.items() if k not in ("data_ptr", "storage_ptr")}


def aliases(tensors):
    """Canonical storage/data pointer equivalence classes, ignoring addresses."""
    storage, data, rows = {}, {}, []
    for label, tensor in tensors:
        sp, dp = tensor["storage_ptr"], tensor["data_ptr"]
        storage.setdefault(sp, len(storage))
        data.setdefault(dp, len(data))
        rows.append({"tensor": label, "storage_group": storage[sp], "data_group": data[dp]})
    return rows


def validate_manifest(manifest, kind):
    require(manifest.get("schema") == "drrqr-layerwise-service/v1", "wrong service schema")
    require(manifest.get("plan_kind") == kind, "wrong plan kind for " + kind)
    require(manifest.get("shared_hf_head_k_dim") == 64
            and manifest.get("layer_head_k_dims") == DIMS
            and manifest.get("num_linear_layers") == 48
            and manifest.get("sum_layer_head_k_dims") == 3072, "not all48 Dk64")
    require(manifest.get("common_optimizations") == COMMON, "common optimization mismatch")
    require("profiler_config" in manifest and manifest["profiler_config"] is None,
            "profiling must be disabled")
    require(manifest.get("physical_npu_contract") == [4, 5, 6, 7]
            and manifest.get("container_npu_namespace") == [0, 1, 2, 3],
            "wrong device contract")
    for name in ("wheel_sha256", "build_manifest_sha256", "reference_manifest_sha256",
                 "wrapper_sha256", "plan_sha256"):
        require(checksum(manifest.get(name)), "invalid " + name)
    for name in ("module_sha256", "build_source_sha256", "source_sha256"):
        require(hash_map(manifest.get(name)), "missing or invalid " + name)
    require(isinstance(manifest.get("plugin_version"), str), "missing plugin version")
    require(isinstance(manifest.get("command"), list) and manifest["command"],
            "missing serving command")
    command = manifest["command"]
    expected = {"--model": "/cache/austinov/Qwen3.8-27B", "--tensor-parallel-size": "4",
                "--dtype": "bfloat16", "--max-model-len": "40960",
                "--max-num-batched-tokens": "40960", "--max-num-seqs": "32",
                "--worker-cls": "layerwise_probe_worker.LayerwiseProbeWorker"}
    for flag, value in expected.items():
        require(command.count(flag) == 1 and command[command.index(flag) + 1] == value,
                "wrong serving command " + flag)
    require("--no-enable-prefix-caching" in command and "--async-scheduling" in command
            and "--profiler-config" not in command, "wrong serving mode")
    require(command.count("--hf-overrides") == 1 and json.loads(
        command[command.index("--hf-overrides") + 1]) ==
        {"text_config": {"linear_key_head_dim": 64}}, "wrong HF shared envelope")
    diagnostic = manifest.get("diagnostic_worker")
    require(isinstance(diagnostic, dict)
            and diagnostic.get("class") == "LayerwiseProbeWorker"
            and checksum(diagnostic.get("worker_sha256")), "missing bound diagnostic worker")
    flags = manifest.get("plugin_flags")
    require(isinstance(flags, dict), "missing plugin flags")
    expected_flags = {
        "VLLM_ASCEND_DRRQR_ENABLE": "1", "VLLM_ASCEND_DRRQR_CAPTURE_ENABLE": "0",
        "VLLM_ASCEND_DRRQR_STRICT": "1", "VLLM_ASCEND_DRRQR_REQUIRE_RUNTIME_HOOKS": "1",
        "VLLM_ASCEND_DRRQR_CONV_LAYOUT": "1", "VLLM_ASCEND_DRRQR_PREFILL_MC2": "1",
        "VLLM_ASCEND_DRRQR_PREFILL_MC2_MIXED": "1",
        "VLLM_ASCEND_DRRQR_PLAN_PATH": manifest.get("plan_path"),
        "VLLM_ASCEND_DRRQR_PLAN_SHA256": manifest["plan_sha256"],
    }
    require(all(flags.get(k) == v for k, v in expected_flags.items()), "plugin flags differ from plan/protocol")
    require(isinstance(manifest.get("inherited_runtime_environment"), dict), "missing inherited runtime settings")


def read_run(directory, kind, inputs):
    manifest = read_report(directory / "service-manifest.json", inputs)
    validate_manifest(manifest, kind)
    exit_report = read_report(directory / "service-exit.json", inputs)
    require(exit_report.get("schema") == "drrqr-layerwise-service-exit/v1"
            and exit_report.get("stop_reason") == "stop_request"
            and exit_report.get("group_remaining") is False
            and type(exit_report.get("exit_code")) is int
            and exit_report["exit_code"] in (0, -15), "service did not stop cleanly through stop.request")
    require(finite(exit_report.get("started_epoch_s")) and finite(exit_report.get("ended_epoch_s"))
            and exit_report["ended_epoch_s"] >= exit_report["started_epoch_s"],
            "invalid service exit timestamps")
    process = read_report(directory / "service-process.json", inputs)
    require(integer(process.get("pid"), 1) and process.get("pid") == process.get("pgid")
            and process["pid"] == exit_report.get("service_pid") == exit_report.get("process_group"),
            "service process/exit identity mismatch")
    phases = {}
    worker_pids = set()
    for rank in range(4):
        phases[rank] = {}
        for phase in ("load", "cache"):
            path = directory / "worker-probe" / f"rank{rank}.{phase}.json"
            report = read_report(path, inputs)
            require(report.get("schema") == "ascend-drrqr-layerwise-worker-probe/v1"
                    and report.get("status") == "passed" and report.get("phase") == phase
                    and type(report.get("rank")) is int and report["rank"] == rank
                    and type(report.get("local_rank")) is int and report["local_rank"] == rank,
                    f"{kind} rank{rank} {phase}: incomplete or wrong report")
            require(report.get("plan_sha256") == manifest["plan_sha256"]
                    and report.get("plan_class") ==
                    ("DrrqrPlan" if kind == "uniform" else "LayerwiseDrrqrPlan"),
                    "worker plan binding mismatch")
            require(report.get("protocol") == PROTOCOL, "worker protocol mismatch")
            require(report.get("worker_source", {}).get("sha256") ==
                    manifest["diagnostic_worker"]["worker_sha256"], "worker source hash mismatch")
            require(integer(report.get("pid"), 1), "missing worker pid")
            require(integer(report.get("started_unix_ns"), 1)
                    and integer(report.get("finished_unix_ns"), 1)
                    and report["finished_unix_ns"] >= report["started_unix_ns"],
                    "unfinished worker phase")
            runtime = report.get("runtime_sources")
            require(isinstance(runtime, dict) and len(runtime) == 3
                    and all(isinstance(v, dict) and checksum(v.get("sha256"))
                            and isinstance(v.get("path"), str) for v in runtime.values()),
                    "worker runtime source coverage missing")
            layers = report.get("layers")
            require(report.get("layer_count") == 48 and isinstance(layers, list) and len(layers) == 48,
                    "incomplete worker layer count")
            indexed = {}
            for layer in layers:
                require(isinstance(layer, dict) and integer(layer.get("layer"))
                        and layer["layer"] not in indexed, "invalid/duplicate worker layer")
                i = layer["layer"]
                attributes = layer.get("expected_attributes", {})
                require(attributes.get("head_k_dim") == 64 and attributes.get("layer_idx") == i
                        and attributes.get("tp_size") == 4 and attributes.get("tp_rank") == rank,
                        "wrong local GDN dimensions")
                require(layer.get("state_shapes", [None, None])[1] == [12, 128, 64],
                        "wrong recurrent state shape")
                indexed[i] = layer
            require(set(indexed) == set(LAYERS), "wrong worker layer coverage")
            if phase == "cache":
                require(integer(report.get("num_blocks"), 1) and report.get("cache_values_read") is False,
                        "missing cache allocation qualification")
                require(report["pid"] == phases[rank]["load"]["pid"], "worker replaced between load and cache")
            else:
                worker_pids.add(report["pid"])
                require(integer(report.get("parameter_bytes_hashed"), 1), "no local parameters hashed")
            report["_layers"] = indexed
            phases[rank][phase] = report
    require(len(worker_pids) == 4, "expected four distinct worker processes")
    sequence = read_report(directory / "integration-sequence.json", inputs)
    require(sequence.get("schema") == "drrqr-layerwise-integration-sequence/v1"
            and sequence.get("status") == "requests_complete", "request sequence incomplete")
    require(sequence.get("service_manifest_sha256") == inputs[str((directory / "service-manifest.json").resolve())],
            "collector bound to a different service manifest")
    require(checksum(sequence.get("script_sha256")), "missing collector source hash")
    protocol = fields(sequence.get("protocol", {}), ("thinking", "temperature", "seed", "requests",
                       "batches", "output_tokens", "ignore_eos", "logprobs", "input_tokens"), "sequence protocol")
    require(protocol["thinking"] is False and protocol["temperature"] == 0 and protocol["seed"] == 0
            and protocol["requests"] == 4 and protocol["batches"] == [1, 3]
            and protocol["output_tokens"] == 32 and protocol["ignore_eos"] is True
            and protocol["logprobs"] == 5, "wrong representative request protocol")
    require(all(sequence.get("comparison_criteria", {}).get(k) == v for k, v in CRITERIA.items()),
            "collector criteria differ from fixed comparator gate")
    results = sequence.get("results")
    require(isinstance(results, list) and len(results) == 4, "missing request results")
    indexed = {}
    for result in results:
        index = result.get("index")
        require(integer(index) and index in range(4) and index not in indexed, "duplicate/wrong request index")
        require(integer(result.get("input_tokens"), 1) and checksum(result.get("prompt_sha256")),
                "missing prompt identity")
        require(isinstance(result.get("token_ids"), list) and len(result["token_ids"]) == 32
                and all(integer(v) for v in result["token_ids"]), "incomplete token IDs")
        require(isinstance(result.get("selected_logprobs"), list)
                and len(result["selected_logprobs"]) == 32
                and all(finite(v) for v in result["selected_logprobs"]), "invalid logprob coverage")
        indexed[index] = result
    require(protocol["input_tokens"] == [indexed[i]["input_tokens"] for i in range(4)],
            "sequence input length coverage mismatch")
    sequence["_results"] = indexed
    return {"manifest": manifest, "phases": phases, "sequence": sequence}


def validate_plan_pair(legacy, layerwise, inputs):
    a, b = legacy["manifest"], layerwise["manifest"]
    require(a["plan_sha256"] != b["plan_sha256"], "uniform/layerwise bindings must be distinct")
    paths = [Path(a["plan_path"]), Path(b["plan_path"])]
    require(all(p.is_absolute() for p in paths), "plan paths must be absolute")
    old, new = [read_report(path, inputs) for path in paths]
    require(inputs[str(paths[0].resolve())] == a["plan_sha256"]
            and inputs[str(paths[1].resolve())] == b["plan_sha256"], "plan document hash changed")
    require(old.get("schema") == "ascend-drrqr-plan/v1"
            and old.get("target_head_k_dim") == 64 and old.get("old_head_k_dim") == 128
            and old.get("num_key_heads") == 16 and old.get("num_value_heads") == 48
            and old.get("head_v_dim") == 128, "not the legacy64 plan")
    require(old.get("target_model_id") == "Qwen/Qwen3.8-27B"
            and old.get("model_type") == "qwen3_5_text", "wrong plan model")
    layer_types = old.get("layer_types")
    require(isinstance(layer_types, list) and len(layer_types) == 64
            and [i for i, kind in enumerate(layer_types) if kind == "linear_attention"] == list(LAYERS),
            "wrong legacy topology")
    keeps = old.get("keep_indices")
    require(isinstance(keeps, dict) and set(keeps) == set(DIMS), "incomplete legacy keep indices")
    for layer, indices in keeps.items():
        require(isinstance(indices, list) and len(indices) == 1024
                and all(integer(v) for v in indices) and len(set(indices)) == 1024,
                "invalid legacy selection at layer " + layer)
        require(all(head * 128 <= value < (head + 1) * 128
                    for head in range(16) for value in indices[head * 64:(head + 1) * 64]),
                "legacy selection crosses heads")
    ref = {"64": {"path": a["plan_path"], "sha256": a["plan_sha256"]}}
    require(new.get("schema") == "ascend-drrqr-layerwise-manifest/v1"
            and new.get("layer_head_k_dims") == DIMS
            and new.get("source_plans") == ref
            and b.get("source_plans") == ref and a.get("source_plans") is None,
            "layerwise all64 is not composed from the same legacy plan")


def compare_dirs(legacy_dir, layerwise_dir):
    inputs, failures = {}, []
    report = {"schema": "drrqr-layerwise-model-control-comparison/v1",
              "status": "failed", "criteria": CRITERIA.copy(), "input_sha256": inputs,
              "failures": failures, "created_epoch_s": time.time(),
              "claim": "legacy64 versus layerwise-all64 model/control and representative trajectory qualification",
              "limits": ["No performance or task accuracy measurement.",
                         "Cache metadata/alias comparisons do not compare multi-step cache values.",
                         "Physical device mapping relies on the parent preflight."]}
    def equal(label, first, second):
        if first != second:
            failures.append(label)
    try:
        legacy_dir, layerwise_dir = Path(legacy_dir).resolve(), Path(layerwise_dir).resolve()
        require(legacy_dir != layerwise_dir, "two distinct service directories required")
        a = read_run(legacy_dir, "uniform", inputs)
        b = read_run(layerwise_dir, "layerwise", inputs)
        validate_plan_pair(a, b, inputs)
        ma, mb = a["manifest"], b["manifest"]
        for name in ("wheel_sha256", "build_manifest_sha256", "module_sha256", "build_source_sha256",
                     "source_sha256", "reference_manifest_sha256", "plugin_version", "wrapper_sha256",
                     "command", "common_optimizations", "inherited_runtime_environment"):
            equal("service." + name, ma[name], mb[name])
        equal("diagnostic_worker_sha256", ma["diagnostic_worker"]["worker_sha256"],
              mb["diagnostic_worker"]["worker_sha256"])
        variable_flags = {"VLLM_ASCEND_DRRQR_PLAN_PATH", "VLLM_ASCEND_DRRQR_PLAN_SHA256",
                          "VLLM_ASCEND_DRRQR_EVIDENCE_FILE"}
        equal("plugin_flags", {k: v for k, v in ma["plugin_flags"].items() if k not in variable_flags},
              {k: v for k, v in mb["plugin_flags"].items() if k not in variable_flags})
        rank_results = []
        common_layer_keys = ("module_name", "prefix", "expected_attributes", "local_qkv_rows",
                             "projection_layout", "state_shapes", "state_dtypes")
        for rank in range(4):
            tensors = {"legacy": [], "layerwise": []}
            parameter_count = 0
            for phase in ("load", "cache"):
                pa, pb = a["phases"][rank][phase], b["phases"][rank][phase]
                equal(f"rank{rank}.{phase}.runtime_sources", pa["runtime_sources"], pb["runtime_sources"])
                if phase == "cache":
                    equal(f"rank{rank}.num_blocks", pa["num_blocks"], pb["num_blocks"])
                byte_counts = [0, 0]
                for layer in LAYERS:
                    left, right = pa["_layers"][layer], pb["_layers"][layer]
                    label = f"rank{rank}.layer{layer}.{phase}"
                    equal(label + ".geometry", fields(left, common_layer_keys, label),
                          fields(right, common_layer_keys, label))
                    if phase == "load":
                        normalized = []
                        for side, record in enumerate((left, right)):
                            params = record.get("parameters")
                            require(isinstance(params, list) and params, label + ": no parameters")
                            names = [p.get("name") for p in params]
                            require(all(isinstance(n, str) for n in names) and len(set(names)) == len(names),
                                    label + ": parameter names are missing/duplicated")
                            identities = [tensor_contract(p, rank, label + "." + p["name"], True)
                                          for p in sorted(params, key=lambda p: p["name"])]
                            normalized.append(identities)
                            byte_counts[side] += sum(p["numel"] * p["element_size"] for p in params)
                        equal(label + ".parameters_exact", normalized[0], normalized[1])
                        parameter_count += len(normalized[0])
                    else:
                        cache_keys = ("cache_group_index", "runner_cache_index", "mamba_spec")
                        equal(label + ".cache_assignment", fields(left, cache_keys, label),
                              fields(right, cache_keys, label))
                        for side, record in (("legacy", left), ("layerwise", right)):
                            require(record.get("binding_identity_verified") is True,
                                    label + ": binding not verified")
                            spec = record["mamba_spec"]
                            fields(spec, ("class", "shapes", "dtypes", "block_size", "page_size_padded",
                                         "page_size_bytes", "mamba_type", "mamba_cache_mode",
                                         "num_speculative_blocks"), label + ".spec")
                            require(spec["shapes"] == record["state_shapes"]
                                    and spec["dtypes"] == record["state_dtypes"], label + ": spec/geometry mismatch")
                            states = record.get("states")
                            require(isinstance(states, list) and len(states) == 2
                                    and [s.get("index") for s in states] == [0, 1], label + ": wrong state tuple")
                            for state in states:
                                i = state["index"]
                                require(state["shape"][1:] == spec["shapes"][i]
                                        and state["dtype"] == spec["dtypes"][i]
                                        and state["shape"][0] >= (pa if side == "legacy" else pb)["num_blocks"],
                                        label + ": actual state/spec mismatch")
                                tensors[side].append((f"layer{layer}.state{i}", state))
                        equal(label + ".states", [tensor_contract(s, rank, label) for s in left["states"]],
                              [tensor_contract(s, rank, label) for s in right["states"]])
                if phase == "load":
                    require(byte_counts == [pa["parameter_bytes_hashed"], pb["parameter_bytes_hashed"]],
                            f"rank{rank}: parameter byte coverage mismatch")
            alias_a, alias_b = aliases(tensors["legacy"]), aliases(tensors["layerwise"])
            equal(f"rank{rank}.cache_alias_pattern", alias_a, alias_b)
            rank_results.append({"rank": rank, "layers_compared": 48,
                                 "parameters_compared": parameter_count,
                                 "cache_tensors_compared": 96,
                                 "cache_alias_pattern_exact": alias_a == alias_b})
        sa, sb = a["sequence"], b["sequence"]
        equal("collector.script_sha256", sa["script_sha256"], sb["script_sha256"])
        equal("collector.protocol", sa["protocol"], sb["protocol"])
        deltas, request_results = [], []
        for index in range(4):
            ra, rb = sa["_results"][index], sb["_results"][index]
            equal(f"request{index}.input_tokens", ra["input_tokens"], rb["input_tokens"])
            equal(f"request{index}.prompt_sha256", ra["prompt_sha256"], rb["prompt_sha256"])
            equal(f"request{index}.token_ids", ra["token_ids"], rb["token_ids"])
            diff = [abs(x - y) for x, y in zip(ra["selected_logprobs"], rb["selected_logprobs"])]
            deltas.extend(diff)
            request_results.append({"index": index, "input_tokens": ra["input_tokens"],
                                    "prompt_hash_exact": ra["prompt_sha256"] == rb["prompt_sha256"],
                                    "token_ids_exact": ra["token_ids"] == rb["token_ids"],
                                    "logprobs_exact": ra["selected_logprobs"] == rb["selected_logprobs"],
                                    "logprob_max_abs": max(diff), "logprob_mean_abs": sum(diff) / len(diff)})
        maximum, mean = max(deltas), sum(deltas) / len(deltas)
        if maximum > MAX_LOGPROB_ABS:
            failures.append("selected_logprob_max_abs")
        if mean > MEAN_LOGPROB_ABS:
            failures.append("selected_logprob_mean_abs")
        report.update(ranks=rank_results, requests=request_results,
                      selected_logprob_count=len(deltas), selected_logprobs_exact=all(x == 0 for x in deltas),
                      selected_logprob_max_abs=maximum, selected_logprob_mean_abs=mean,
                      status="passed" if not failures else "failed")
    except (ValueError, KeyError, TypeError, IndexError, AttributeError, OverflowError, OSError) as error:
        failures.append("incomplete_or_invalid_input")
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-dir", required=True, type=Path)
    parser.add_argument("--layerwise-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not args.output.is_absolute() or RESULT_ROOT.resolve() not in output.parents:
        parser.error("output must be an absolute new JSON path below /drrqr-results")
    if output.exists() or not output.parent.is_dir():
        parser.error("output parent must exist and output must be new")
    # Claim the exact target before reading. Never overwrite an old result.
    with output.open("x", encoding="utf-8") as stream:
        report = compare_dirs(args.legacy_dir, args.layerwise_dir)
        report["comparator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "output": str(output),
                      "failures": report["failures"]}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
