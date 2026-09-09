"""CPU-only observation of the installed MC2 route; no tensor/math mutations."""
from collections import Counter
import functools
import inspect
import json


def snapshot_context(context, rows, eligible_rows):
    metadata = getattr(context, "attn_metadata", None)
    signatures = Counter()
    if isinstance(metadata, dict):
        for item in metadata.values():
            record = {"type": type(item).__name__}
            for key in ("num_prefills", "num_decodes", "num_decode_tokens", "num_actual_tokens"):
                value = getattr(item, key, None)
                record[key] = value if type(value) is int else None
            signatures[json.dumps(record, sort_keys=True)] += 1
    values = [dict(json.loads(key), entries=count) for key, count in sorted(signatures.items())]
    valid = bool(values) and all(all(v[key] is not None for key in (
        "num_prefills", "num_decodes", "num_decode_tokens", "num_actual_tokens")) for v in values)
    actual = {v["num_actual_tokens"] for v in values}
    decodes = {v["num_decode_tokens"] for v in values}
    actual_rows = next(iter(actual)) if valid and len(actual) == 1 else None
    decode_rows = next(iter(decodes)) if valid and len(decodes) == 1 else None
    phase = "unknown"
    if valid:
        if all(v["num_prefills"] > 0 and v["num_decodes"] == v["num_decode_tokens"] == 0 for v in values):
            phase = "prefill"
        elif all(v["num_prefills"] == 0 and v["num_decodes"] > 0 for v in values):
            phase = "decode"
        elif all(v["num_prefills"] > 0 and v["num_decodes"] > 0 for v in values):
            phase = "mixed"
        else:
            phase = "heterogeneous"
    return {"phase": phase, "padded_rows": rows if type(rows) is int else None,
            "eligible_rows": eligible_rows, "actual_rows": actual_rows,
            "decode_rows": decode_rows,
            "prefill_rows": actual_rows - decode_rows if actual_rows is not None and decode_rows is not None else None,
            "metadata": values, "graph_mode": getattr(getattr(context, "cudagraph_runtime_mode", None), "name", None),
            "in_profile_run": bool(getattr(context, "in_profile_run", False)),
            "capturing": bool(getattr(context, "capturing", False))}


def install_observer(runner, adapter, bindings, get_context, active, emit, max_steps=8192):
    if getattr(runner, "_drrqr_coverage_installed", False):
        raise RuntimeError("coverage already installed")
    original_forward = runner._model_forward
    signature = inspect.signature(original_forward)
    if "num_tokens_padded" not in signature.parameters:
        raise RuntimeError("runner ABI changed")
    original_emit = adapter.emit_evidence
    current = None
    sequence = 0

    def observed_emit(config, event, **fields):
        if event != "prefill_mc2_called" or current is None:
            return original_emit(config, event, **fields)
        name = fields["name"]
        if name not in bindings or name in current["calls"]:
            raise RuntimeError("unexpected or duplicate projection execution")
        current["calls"][name] = (fields["k"], fields["rows"])
        # Preserve the ordinary once-per-target evidence semantics.
        if not current["prior"][name]:
            return original_emit(config, event, **fields)

    @functools.wraps(original_forward)
    def observed_forward(*args, **kwargs):
        nonlocal current, sequence
        if not active():
            return original_forward(*args, **kwargs)
        if current is not None:
            raise RuntimeError("nested model forward is not supported by coverage")
        sequence += 1
        if sequence > max_steps:
            raise RuntimeError("coverage step budget exceeded")
        rows = signature.bind(*args, **kwargs).arguments["num_tokens_padded"]
        context = get_context()
        record = snapshot_context(context, rows, adapter.prefill_rows(context, rows))
        prior = {name: binding["observed"] for name, (_, binding) in bindings.items()}
        current = {"calls": {}, "prior": prior}
        for _, binding in bindings.values():
            binding["observed"] = False
        status = "complete"
        try:
            return original_forward(*args, **kwargs)
        except BaseException:
            status = "failed"
            raise
        finally:
            calls = current["calls"]
            counts, row_calls = Counter(), Counter()
            for k, nrows in calls.values():
                counts[str(k)] += 1
                row_calls[str(k)] += nrows
            for name, (_, binding) in bindings.items():
                binding["observed"] = prior[name] or binding["observed"]
            current = None
            emit({"event": "model_forward", "sequence": sequence, "status": status,
                  **record, "fused_calls_by_k": dict(counts), "fused_row_calls_by_k": dict(row_calls)})

    adapter.emit_evidence = observed_emit
    runner._model_forward = observed_forward
    runner._drrqr_coverage_installed = True
