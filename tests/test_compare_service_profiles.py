import json

import pytest

from tools.compare_service_profiles import normalized_command, target_head_dim


def _command(head_dim=None):
    result = [
        "python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--profiler-config",
        json.dumps({"profiler": "torch"}),
    ]
    if head_dim is not None:
        result.extend([
            "--hf-overrides",
            json.dumps({"text_config": {"linear_key_head_dim": head_dim}}),
        ])
    return result


def test_normalized_command_matches_dense_and_reduced_dimensions():
    assert normalized_command(_command(), 128) == normalized_command(
        _command(64), 64)
    assert normalized_command(_command(32), 32) == ["python3", "-m",
                                                    "vllm.entrypoints.openai.api_server"]


def test_normalized_command_rejects_wrong_dimension():
    with pytest.raises(ValueError, match="override differs"):
        normalized_command(_command(64), 32)


def test_normalized_command_rejects_dense_override():
    with pytest.raises(ValueError, match="dense baseline"):
        normalized_command(_command(128), 128)


def test_target_head_dim_supports_legacy_manifest():
    assert target_head_dim({"command": _command()}) == 128
    assert target_head_dim({"command": _command(64)}) == 64
    assert target_head_dim({"command": _command(), "target_head_k_dim": 32}) == 32
