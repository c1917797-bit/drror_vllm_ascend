#!/usr/bin/env python3
"""Compare one uniform64 service restart with the original uniform64 control.

A/A evidence diagnoses repeatability only. It never changes the failed A/B
gate, certifies a layerwise candidate, or measures performance/task accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import types

VALIDATOR_SHA256 = "9bb27324c083a52e65df23309900227761efb0bd3b36f64e59f880119d735b26"
VALIDATOR_PATH = Path(__file__).resolve().with_name("compare_layerwise_model_control.py")
RESULT_ROOT = Path("/drrqr-results")
PARAMETER_NAMES = frozenset(("dt_bias", "A_log", "conv1d.weight", "in_proj_qkvz.weight",
                             "in_proj_ba.weight", "norm.weight", "out_proj.weight"))


def load_validator(path=VALIDATOR_PATH):
    # Compile the exact bytes that were hashed; do not reread through an importer.
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != VALIDATOR_SHA256:
        raise ValueError("the admitted model-control validator bytes changed")
    module = types.ModuleType("uniform_repeatability_pinned_validator")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def validate_same_uniform_plan(first, repeat, inputs, validator):
    a, b = first["manifest"], repeat["manifest"]
    require = validator.require
    require(a["plan_sha256"] == b["plan_sha256"] and a["plan_path"] == b["plan_path"],
            "A/A requires the exact same uniform plan binding")
    require(a.get("source_plans") is None and b.get("source_plans") is None,
            "A/A may not contain a layerwise source manifest")
    path = Path(a["plan_path"])
    require(path.is_absolute(), "uniform plan path must be absolute")
    plan = validator.read_report(path, inputs)
    require(inputs[str(path.resolve())] == a["plan_sha256"], "uniform plan bytes changed")
    require(plan.get("schema") == "ascend-drrqr-plan/v1"
            and plan.get("target_model_id") == "Qwen/Qwen3.8-27B"
            and plan.get("model_type") == "qwen3_5_text"
            and plan.get("target_head_k_dim") == 64 and plan.get("old_head_k_dim") == 128
            and plan.get("num_key_heads") == 16 and plan.get("num_value_heads") == 48
            and plan.get("head_v_dim") == 128, "not the admitted uniform64 plan")
    require(plan.get("layer_types") == [
        "linear_attention" if i in validator.LAYERS else "full_attention" for i in range(64)],
        "uniform plan topology differs")
    keeps = plan.get("keep_indices")
    require(isinstance(keeps, dict) and set(keeps) == set(validator.DIMS),
            "uniform plan selection coverage incomplete")
    for layer, indices in keeps.items():
        require(isinstance(indices, list) and len(indices) == 1024
                and all(validator.integer(i) for i in indices) and len(set(indices)) == 1024,
                "invalid uniform indices at layer " + layer)
        require(all(head * 128 <= i < (head + 1) * 128
                    for head in range(16) for i in indices[head * 64:(head + 1) * 64]),
                "uniform selection crosses head boundaries")
    return a["plan_sha256"]


def compare_runs(first_dir, repeat_dir):
    inputs, failures = {}, []
    report = {
        "schema": "drrqr-uniform-repeatability-comparison/v1", "status": "failed",
        "created_epoch_s": time.time(), "input_sha256": inputs, "failures": failures,
        "validator_sha256": VALIDATOR_SHA256,
        "criteria": {"prompt_hashes_exact": True, "token_ids_exact": True,
                     "selected_logprob_max_abs": 0.02, "selected_logprob_mean_abs": 0.005,
                     "aggregation": "max and arithmetic mean over all 128 selected-token logprobs"},
        "claim": "one independent restart of the same uniform64 model-control protocol",
        "ab_status_unchanged": "failed; A/A does not pass or reclassify the prior layerwise A/B result",
        "limits": [
            "One A/A pair is not a distribution or proof of run-to-run numerical noise.",
            "A/A failure would show that layerwise construction is not necessary for this discrepancy; it does not prove layerwise correctness.",
            "Cache metadata and alias identity do not compare recurrent state values.",
            "No performance or benchmark accuracy measurement.",
        ],
    }
    def equal(label, a, b):
        if a != b:
            failures.append(label)
    try:
        v = load_validator()
        first_dir, repeat_dir = Path(first_dir).resolve(), Path(repeat_dir).resolve()
        v.require(first_dir != repeat_dir, "A/A requires two distinct service directories")
        first = v.read_run(first_dir, "uniform", inputs)
        repeat = v.read_run(repeat_dir, "uniform", inputs)
        report["plan_sha256"] = validate_same_uniform_plan(first, repeat, inputs, v)
        a, b = first["manifest"], repeat["manifest"]
        for name in ("wheel_sha256", "build_manifest_sha256", "module_sha256", "build_source_sha256",
                     "source_sha256", "reference_manifest_sha256", "plugin_version", "wrapper_sha256",
                     "command", "common_optimizations", "inherited_runtime_environment", "shared_hf_head_k_dim",
                     "layer_head_k_dims", "plan_sha256", "plan_path"):
            equal("service." + name, a[name], b[name])
        equal("diagnostic_worker", {k: value for k, value in a["diagnostic_worker"].items() if k != "output_dir"},
              {k: value for k, value in b["diagnostic_worker"].items() if k != "output_dir"})
        equal("plugin_flags", {k: value for k, value in a["plugin_flags"].items()
                               if k != "VLLM_ASCEND_DRRQR_EVIDENCE_FILE"},
              {k: value for k, value in b["plugin_flags"].items()
               if k != "VLLM_ASCEND_DRRQR_EVIDENCE_FILE"})
        common_fields = ("module_name", "prefix", "expected_attributes", "local_qkv_rows",
                         "projection_layout", "state_shapes", "state_dtypes")
        ranks = []
        for rank in range(4):
            rank_start = len(failures)
            cache_tensors = [[], []]
            parameter_counts = [0, 0]
            blocks = []
            for phase in ("load", "cache"):
                pair = [run["phases"][rank][phase] for run in (first, repeat)]
                pa, pb = pair
                dispatch_fields = ("parent_worker", "worker_source", "runtime_sources")
                if phase == "load":
                    dispatch_fields += ("model_class",)
                equal(f"rank{rank}.{phase}.dispatch",
                      v.fields(pa, dispatch_fields, phase), v.fields(pb, dispatch_fields, phase))
                byte_counts = [0, 0]
                if phase == "cache":
                    blocks = [pa["num_blocks"], pb["num_blocks"]]
                    equal(f"rank{rank}.num_blocks", *blocks)
                for layer in v.LAYERS:
                    records = [item["_layers"][layer] for item in pair]
                    label = f"rank{rank}.layer{layer}.{phase}"
                    equal(label + ".geometry", *(v.fields(item, common_fields, label) for item in records))
                    if phase == "load":
                        identities = []
                        for side, item in enumerate(records):
                            v.fields(item, ("module_class", "method_sources"), label + ".dispatch")
                            parameters = item.get("parameters")
                            v.require(isinstance(parameters, list) and len(parameters) == 7
                                      and {p.get("name") for p in parameters} == PARAMETER_NAMES,
                                      label + ": expected all seven frozen local GDN parameters")
                            identities.append([v.tensor_contract(p, rank, label + "." + p["name"], True)
                                               for p in sorted(parameters, key=lambda p: p["name"])])
                            parameter_counts[side] += len(parameters)
                            byte_counts[side] += sum(p["numel"] * p["element_size"] for p in parameters)
                        equal(label + ".parameters_exact", *identities)
                        for key in ("module_class", "method_sources"):
                            equal(label + "." + key, records[0][key], records[1][key])
                    else:
                        for key in ("cache_group_index", "runner_cache_index", "mamba_spec"):
                            v.require(all(key in item for item in records), label + ": missing " + key)
                            equal(label + "." + key, records[0][key], records[1][key])
                        normalized_states = []
                        for side, item in enumerate(records):
                            v.require(item.get("binding_identity_verified") is True, label + ": binding not qualified")
                            spec = item["mamba_spec"]
                            v.fields(spec, ("class", "shapes", "dtypes", "block_size", "page_size_padded",
                                            "page_size_bytes", "mamba_type", "mamba_cache_mode",
                                            "num_speculative_blocks"), label + ".spec")
                            v.require(spec["shapes"] == item["state_shapes"]
                                      and spec["dtypes"] == item["state_dtypes"], label + ": cache spec geometry mismatch")
                            states = item.get("states")
                            v.require(isinstance(states, list) and len(states) == 2
                                      and [s.get("index") for s in states] == [0, 1], label + ": wrong cache tuple")
                            normalized_states.append([v.tensor_contract(s, rank, label) for s in states])
                            for state in states:
                                index = state["index"]
                                v.require(state["shape"][1:] == spec["shapes"][index]
                                          and state["dtype"] == spec["dtypes"][index]
                                          and state["shape"][0] >= pair[side]["num_blocks"],
                                          label + ": actual cache differs from allocation spec")
                                cache_tensors[side].append((f"layer{layer}.state{index}", state))
                        equal(label + ".states_exact", *normalized_states)
                if phase == "load":
                    v.require(byte_counts == [p["parameter_bytes_hashed"] for p in pair],
                              f"rank{rank}: incomplete parameter byte coverage")
            v.require(parameter_counts == [336, 336], f"rank{rank}: expected 336 local GDN parameters")
            alias_pair = [v.aliases(states) for states in cache_tensors]
            equal(f"rank{rank}.cache_alias_pattern", *alias_pair)
            ranks.append({"rank": rank, "layers_compared": 48, "parameter_counts": parameter_counts,
                          "cache_tensors_compared": [len(x) for x in cache_tensors],
                          "expected_num_blocks": 925, "observed_num_blocks": blocks,
                          "num_blocks_match_expected": blocks == [925, 925],
                          "cache_alias_pattern_exact": alias_pair[0] == alias_pair[1],
                          "structural_exact": len(failures) == rank_start})
        report["ranks"] = ranks
        sa, sb = first["sequence"], repeat["sequence"]
        for key in ("script_sha256", "protocol", "comparison_criteria"):
            equal("collector." + key, sa[key], sb[key])
        requests, deltas = [], []
        for index in range(4):
            qa, qb = sa["_results"][index], sb["_results"][index]
            for key in ("prompt_sha256", "input_tokens", "token_ids"):
                equal(f"request{index}." + key, qa[key], qb[key])
            tokens = [{"position": j, "first_token_id": qa["token_ids"][j],
                       "repeat_token_id": qb["token_ids"][j],
                       "first_logprob": x, "repeat_logprob": y, "delta": y - x, "abs_delta": abs(y - x)}
                      for j, (x, y) in enumerate(zip(qa["selected_logprobs"], qb["selected_logprobs"]))]
            difference = [item["abs_delta"] for item in tokens]
            deltas.extend(difference)
            maximum, mean = max(difference), sum(difference) / len(difference)
            requests.append({"index": index, "isolated_first_request": index == 0,
                             "input_tokens": qa["input_tokens"], "prompt_hash_exact": qa["prompt_sha256"] == qb["prompt_sha256"],
                             "token_ids_exact": qa["token_ids"] == qb["token_ids"],
                             "logprobs_exact": all(d == 0 for d in difference),
                             "logprob_max_abs": maximum, "logprob_mean_abs": mean,
                             "request_meets_same_thresholds": maximum <= 0.02 and mean <= 0.005,
                             "tokens": tokens})
        maximum, mean = max(deltas), sum(deltas) / len(deltas)
        if maximum > 0.02:
            failures.append("selected_logprob_max_abs")
        if mean > 0.005:
            failures.append("selected_logprob_mean_abs")
        report.update(requests=requests, first_isolated_request=requests[0],
                      selected_logprob_count=len(deltas),
                      selected_logprobs_exact=all(x == 0 for x in deltas),
                      selected_logprob_max_abs=maximum, selected_logprob_mean_abs=mean,
                      status="passed" if not failures else "failed")
    except (ValueError, KeyError, TypeError, IndexError, AttributeError, OverflowError, OSError) as error:
        failures.append("incomplete_or_invalid_input")
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-dir", required=True, type=Path)
    parser.add_argument("--repeat-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not args.output.is_absolute() or RESULT_ROOT.resolve() not in output.parents:
        parser.error("output must be an absolute new path below /drrqr-results")
    if output.exists() or not output.parent.is_dir():
        parser.error("output parent must exist and output must be new")
    with output.open("x", encoding="utf-8") as stream:
        report = compare_runs(args.first_dir, args.repeat_dir)
        report["comparator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "output": str(output),
                      "failures": report["failures"]}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

