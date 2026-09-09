"""Narrow deterministic diagnostic settings; no operator substitution."""
def apply_settings(torch_module, environ):
    if environ.get("DRRQR_DIAGNOSTIC_DETERMINISM") != "1":
        raise RuntimeError("diagnostic worker requires explicit opt-in")
    if environ.get("VLLM_BATCH_INVARIANT", "0") != "0":
        raise RuntimeError("keep the existing GDN custom operators enabled")
    if environ.get("HCCL_DETERMINISTIC") != "strict" or environ.get("LCCL_DETERMINISTIC") != "1":
        raise RuntimeError("collective settings must be supplied before worker import")
    if torch_module.distributed.is_initialized():
        raise RuntimeError("deterministic settings must precede communicator initialization")
    torch_module.use_deterministic_algorithms(True, warn_only=True)
    if not torch_module.are_deterministic_algorithms_enabled():
        raise RuntimeError("torch deterministic option did not activate")
    return {"torch_deterministic_algorithms": True, "torch_warn_only": True,
            "HCCL_DETERMINISTIC": "strict", "LCCL_DETERMINISTIC": "1",
            "VLLM_BATCH_INVARIANT": "0", "operator_substitutions": False}

