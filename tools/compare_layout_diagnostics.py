#!/usr/bin/env python3
"""Compare provenance-bound layout probes without treating unstable outputs as parity."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command_key(command):
    command = list(command)
    if "--profiler-config" in command:
        i = command.index("--profiler-config") + 1
        config = json.loads(command[i])
        config["torch_profiler_dir"] = "<per-service-artifact-directory>"
        command[i] = json.dumps(config, sort_keys=True)
    return command


def validate_pair(original_dir, packed_dir):
    manifests = [json.loads((p / "service-manifest.json").read_text()) for p in (original_dir, packed_dir)]
    probes = [json.loads((p / "output-probe.json").read_text()) for p in (original_dir, packed_dir)]
    for root, manifest, probe, layout in zip((original_dir, packed_dir), manifests, probes, ("original", "packed")):
        if manifest["layout"] != layout or probe["layout"] != layout:
            raise ValueError("layout labels differ")
        if probe["service_manifest_sha256"] != sha(root / "service-manifest.json"):
            raise ValueError("probe is bound to another manifest")
        if probe["variant"] != manifest["variant"]:
            raise ValueError("probe variant differs")
        if manifest["plugin_flags"]["VLLM_ASCEND_DRRQR_CONV_LAYOUT"] != ("1" if layout == "packed" else "0"):
            raise ValueError("layout flag differs")
    for field in ("variant", "module_sha256", "source_sha256", "wheel_sha256", "plan_sha256",
                  "physical_npu_contract", "container_npu_namespace"):
        if manifests[0][field] != manifests[1][field]:
            raise ValueError(f"unmatched manifest field: {field}")
    if command_key(manifests[0]["command"]) != command_key(manifests[1]["command"]):
        raise ValueError("unmatched service command")
    flags = [{k: v for k, v in m["plugin_flags"].items()
              if k not in ("VLLM_ASCEND_DRRQR_CONV_LAYOUT", "VLLM_ASCEND_DRRQR_EVIDENCE_FILE")}
             for m in manifests]
    if flags[0] != flags[1]:
        raise ValueError("unmatched non-layout plugin flags")
    for field in ("cases", "driver_sha256", "script_sha256"):
        if probes[0][field] != probes[1][field]:
            raise ValueError(f"unmatched probe field: {field}")
    return manifests, probes


def index_rounds(probe):
    expected = {c["id"] for c in probe["cases"]}
    coverage = sorted((c["input_tokens"], c["output_tokens"], c["concurrency"]) for c in probe["cases"])
    if (coverage != [(2048, 128, 16)] * 16 + [(32768, 128, 1)] * 2
            or len(expected) != len(probe["cases"]) or len(probe["rounds"]) != 2):
        raise ValueError("invalid case/repeat coverage")
    indexed = []
    for rows in probe["rounds"]:
        values = {r["id"]: r["token_ids"] for r in rows}
        if set(values) != expected or len(values) != len(rows):
            raise ValueError("missing or duplicate result case")
        for case in probe["cases"]:
            tokens = values[case["id"]]
            if len(tokens) != case["output_tokens"] or not all(type(t) is int for t in tokens):
                raise ValueError("incomplete/invalid token output")
        indexed.append(values)
    return indexed


def compare(original_dir, packed_dir):
    manifests, probes = validate_pair(original_dir, packed_dir)
    rounds = [index_rounds(p) for p in probes]
    cases = []
    for case in probes[0]["cases"]:
        name = case["id"]
        a = [r[name] for r in rounds[0]]
        b = [r[name] for r in rounds[1]]
        cases.append({"id": name, "input_tokens": case["input_tokens"],
                      "original_repeat_stable": a[0] == a[1],
                      "packed_repeat_stable": b[0] == b[1],
                      "all_four_equal": all(t == a[0] for t in a + b),
                      "cross_layout_equal_round_pairs": sum(x == y for x in a for y in b)})
    stable = all(c["original_repeat_stable"] and c["packed_repeat_stable"] for c in cases)
    all_equal = all(c["all_four_equal"] for c in cases)
    return {"schema": "drrqr-layout-diagnostic-comparison/v1", "variant": manifests[0]["variant"],
            "provenance_matched": True,
            "status": "matched_output_probe" if all_equal else ("layout_output_difference" if stable else "inconclusive_repeatability"),
            "cases": cases,
            "summary_by_input_length": {str(n): {
                "cases": sum(c["input_tokens"] == n for c in cases),
                "original_stable": sum(c["input_tokens"] == n and c["original_repeat_stable"] for c in cases),
                "packed_stable": sum(c["input_tokens"] == n and c["packed_repeat_stable"] for c in cases),
                "all_four_equal": sum(c["input_tokens"] == n and c["all_four_equal"] for c in cases)}
                for n in sorted({c["input_tokens"] for c in cases})},
            "sources": {str(root / name): sha(root / name) for root in (original_dir, packed_dir)
                        for name in ("service-manifest.json", "output-probe.json")},
            "limits": ["Not a quality or performance evaluation.",
                       "Matching sampled token outputs is not proof of every internal tensor/state.",
                       "An unstable original control does not absolve the candidate of numerical validation.",
                       "No tolerance was relaxed and no failed case was excluded."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--packed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.original, args.packed)
    report["script_sha256"] = sha(Path(__file__))
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "summary_by_input_length")}, indent=2))


if __name__ == "__main__":
    main()
