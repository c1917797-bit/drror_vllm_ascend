"""Bounded post-convolution Q/K capture through a runtime monkeypatch."""

from __future__ import annotations

import functools
import os
import threading
import time
from pathlib import Path

from ..diagnostics import emit_evidence
from ..envs import DrrqrConfig

CAPTURE_SCHEMA = "drrqr-post-conv-qk/v1"


def install_capture_patch(gdn_cls, gdn_module, config: DrrqrConfig) -> None:
    """Capture pure-prefill tensors passed to post-conv QKV rearrangement."""

    original = gdn_cls.rearrange_mixed_qkv
    identity = (
        config.capture_dir,
        config.source_config_sha256,
        config.calibration_sha256,
    )
    installed = getattr(original, "_drror_capture_config", None)
    if installed is not None:
        if installed != identity:
            raise RuntimeError("DRRQR: capture hook already installed for another identity")
        return

    lock = threading.Lock()
    counts: dict[tuple[int, str], int] = {}

    @functools.wraps(original)
    def rearrange(self, mixed_qkv):
        if mixed_qkv is not None and mixed_qkv.numel():
            context = gdn_module.get_forward_context()
            metadata = getattr(context, "attn_metadata", None)
            if isinstance(metadata, dict):
                metadata = metadata.get(self.prefix)
            capture_root = Path(config.capture_dir)
            armed = capture_root / ".armed"
            if metadata is not None and getattr(metadata, "num_prefills", 0) > 0 and armed.is_file():
                if (
                    getattr(metadata, "num_decodes", 0) != 0
                    or getattr(metadata, "spec_sequence_masks", None) is not None
                ):
                    raise RuntimeError("DRRQR: capture requires a pure non-speculative prefill")
                tp_rank = int(self.tp_rank)
                prefix = str(self.prefix)
                key = (tp_rank, prefix)
                with lock:
                    capture_index = counts.get(key, 0)
                    if capture_index >= config.capture_max_per_layer:
                        return original(self, mixed_qkv)
                    counts[key] = capture_index + 1

                local_key_dim = int(self.key_dim // self.tp_size)
                if mixed_qkv.ndim != 2 or mixed_qkv.shape[1] < 2 * local_key_dim:
                    raise RuntimeError(
                        f"DRRQR: unexpected post-conv packed shape {tuple(mixed_qkv.shape)}"
                    )
                token_count = min(int(mixed_qkv.shape[0]), config.capture_max_tokens)
                if token_count < int(mixed_qkv.shape[0]):
                    indices = gdn_module.torch.linspace(
                        0,
                        int(mixed_qkv.shape[0]) - 1,
                        steps=token_count,
                        device=mixed_qkv.device,
                    ).long()
                    sampled = mixed_qkv.index_select(0, indices)
                else:
                    sampled = mixed_qkv
                q = sampled[:, :local_key_dim].detach().to(device="cpu").contiguous()
                k = (
                    sampled[:, local_key_dim : 2 * local_key_dim]
                    .detach()
                    .to(device="cpu")
                    .contiguous()
                )
                capture_root.mkdir(parents=True, exist_ok=True)
                safe_prefix = prefix.replace("/", "_").replace(".", "_")
                final_path = capture_root / (
                    f"rank{tp_rank}_{safe_prefix}_capture{capture_index}_"
                    f"{os.getpid()}_{time.time_ns()}.pt"
                )
                temporary = final_path.with_suffix(".pt.tmp")
                payload = {
                    "schema": CAPTURE_SCHEMA,
                    "target_model_id": "Qwen/Qwen3.8-27B",
                    "source_config_sha256": config.source_config_sha256,
                    "calibration_sha256": config.calibration_sha256,
                    "stage": "post_conv_qk",
                    "prefix": prefix,
                    "tp_rank": tp_rank,
                    "tp_size": int(self.tp_size),
                    "capture_index": capture_index,
                    "token_count": token_count,
                    "local_key_dim": local_key_dim,
                    "old_head_k_dim": int(self.head_k_dim),
                    "q": q,
                    "k": k,
                }
                gdn_module.torch.save(payload, temporary)
                os.replace(temporary, final_path)
                emit_evidence(
                    config,
                    "post_conv_qk_capture_written",
                    component="capture",
                    prefix=prefix,
                    tp_rank=tp_rank,
                    capture_index=capture_index,
                    token_count=token_count,
                    source_config_sha256=config.source_config_sha256,
                    calibration_sha256=config.calibration_sha256,
                    file=final_path.name,
                )
        return original(self, mixed_qkv)

    rearrange._drror_capture_config = identity
    gdn_cls.rearrange_mixed_qkv = rearrange
