"""Fake-tokenizer/HTTP tests; no model, network, or device execution."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def setup_collector(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "tools/collect_layerwise_sequence.py"
    spec = importlib.util.spec_from_file_location("isolated_sequence_collector", script)
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    root = tmp_path / "results"
    service = root / "service"
    service.mkdir(parents=True)
    (service / "service-manifest.json").write_text("{}")
    # Model the /drrqr-results boundary with a private test directory.
    monkeypatch.setattr(collector, "Path", lambda value: root if str(value) == "/drrqr-results" else Path(value))
    monkeypatch.setattr(sys, "argv", [str(script), "--service-dir", str(service)])
    state = SimpleNamespace(
        service=service, payloads=[], templates=[], health_checks=0,
        response_mutator=None, bad_prompt=False,
    )

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            state.templates.append((messages, kwargs))
            if state.bad_prompt:
                return {"input_ids": [101]}
            index = len(state.templates) - 1
            return [101 + index] * (index + 4)

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert path == "/cache/austinov/Qwen3.8-27B"
            assert kwargs == {"trust_remote_code": True, "local_files_only": True}
            return FakeTokenizer()

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    def fake_urlopen(request, timeout):
        if isinstance(request, str):
            assert request == "http://127.0.0.1:6666/health" and timeout == 3
            state.health_checks += 1
            response = io.StringIO("")
            response.status = 200
            return response
        assert request.full_url == "http://127.0.0.1:6666/v1/completions"
        assert request.method == "POST" and timeout == 600
        payload = json.loads(request.data)
        state.payloads.append(payload)
        response = {
            "choices": [{"token_ids": [19] * 32, "finish_reason": "length",
                         "logprobs": {"token_logprobs": [-0.1] * 32}}],
            "usage": {"prompt_tokens": len(payload["prompt"]), "completion_tokens": 32},
        }
        if state.response_mutator:
            state.response_mutator(response)
        return io.StringIO(json.dumps(response))

    monkeypatch.setattr(collector, "urlopen", fake_urlopen)
    return collector, state


def read_receipt(state):
    return json.loads((state.service / "integration-sequence.json").read_text())


def test_exact_four_by_32_protocol_and_fixed_comparison_criteria(setup_collector):
    collector, state = setup_collector
    collector.main()
    receipt = read_receipt(state)
    assert receipt["status"] == "requests_complete"
    assert receipt["claim"] == "model integration and representative output trajectory only"
    assert receipt["protocol"] == {
        "thinking": False, "temperature": 0, "seed": 0, "requests": 4,
        "batches": [1, 3], "output_tokens": 32, "ignore_eos": True,
        "logprobs": 5, "input_tokens": [4, 5, 6, 7],
    }
    criteria = receipt["comparison_criteria"]
    assert criteria["token_ids_exact"] is True
    assert criteria["prompt_hashes_exact"] is True
    assert criteria["selected_logprob_max_abs"] == 0.02
    assert criteria["selected_logprob_mean_abs"] == 0.005
    assert "bitwise gates remain separate" in criteria["note"]
    assert len(state.payloads) == 4
    assert [r["index"] for r in receipt["results"]] == [0, 1, 2, 3]
    for payload in state.payloads:
        assert set(payload) == {
            "model", "prompt", "temperature", "seed", "max_tokens", "ignore_eos",
            "return_token_ids", "logprobs", "stream",
        }
        assert payload["model"] == "qwen3.8"
        assert payload["max_tokens"] == 32 and payload["ignore_eos"] is True
        assert payload["return_token_ids"] is True and payload["logprobs"] == 5
        assert payload["temperature"] == payload["seed"] == 0
        assert payload["stream"] is False
    for messages, kwargs in state.templates:
        assert len(messages) == 1 and messages[0]["role"] == "user"
        assert kwargs == {
            "tokenize": True, "return_dict": False, "add_generation_prompt": True,
            "enable_thinking": False,
        }
    for index, result in enumerate(receipt["results"]):
        prompt = [101 + index] * (index + 4)
        assert result["prompt_sha256"] == hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
        assert result["token_ids"] == [19] * 32
    assert (state.service / "integration-sequence.claim.json").is_file()
    assert (state.service / "stop.request").is_file()
    assert not (state.service / "integration-sequence.json.partial").exists()
    assert "throughput" not in receipt and "accuracy" not in receipt


def test_nonflat_tokenizer_output_sends_no_request(setup_collector):
    collector, state = setup_collector
    state.bad_prompt = True
    with pytest.raises(RuntimeError, match="flat integer token IDs"):
        collector.main()
    assert state.payloads == []
    assert read_receipt(state)["status"] == "failed"
    assert (state.service / "stop.request").is_file()


@pytest.mark.parametrize("kind", ["bool_token", "wrong_count", "nan_score", "missing_score", "early_eos"])
def test_invalid_response_is_failed_receipt_without_retry(setup_collector, kind):
    collector, state = setup_collector
    def corrupt(response):
        choice = response["choices"][0]
        if kind == "bool_token":
            choice["token_ids"][0] = True
        elif kind == "wrong_count":
            response["usage"]["completion_tokens"] = 31
        elif kind == "nan_score":
            choice["logprobs"]["token_logprobs"][0] = float("nan")
        elif kind == "missing_score":
            choice["logprobs"]["token_logprobs"].pop()
        else:
            choice["finish_reason"] = "stop"
    state.response_mutator = corrupt
    with pytest.raises(RuntimeError, match="coverage mismatch"):
        collector.main()
    assert len(state.payloads) == 1
    assert read_receipt(state)["status"] == "failed"
    assert (state.service / "stop.request").is_file()


def test_startup_exit_is_saved_before_any_health_or_completion(setup_collector):
    collector, state = setup_collector
    (state.service / "service-exit.json").write_text("{}")
    with pytest.raises(RuntimeError, match="service exited"):
        collector.main()
    assert state.health_checks == 0 and state.payloads == []
    receipt = read_receipt(state)
    assert receipt["status"] == "failed" and receipt["service_manifest_sha256"] is None
    assert (state.service / "stop.request").is_file()


def test_startup_timeout_is_saved_and_stops_service(setup_collector, monkeypatch):
    collector, state = setup_collector
    (state.service / "service-manifest.json").unlink()
    clock = iter([0, 901])
    monkeypatch.setattr(collector, "time", SimpleNamespace(
        monotonic=lambda: next(clock), time_ns=lambda: 123, sleep=lambda _: None,
    ))
    with pytest.raises(TimeoutError, match="900-second"):
        collector.main()
    assert state.health_checks == 0 and state.payloads == []
    assert read_receipt(state)["status"] == "failed"
    assert (state.service / "stop.request").is_file()


def test_existing_claim_never_sends_or_stops_other_collector(setup_collector):
    collector, state = setup_collector
    claim = state.service / "integration-sequence.claim.json"
    claim.write_text('{"owner":"other collector"}')
    original = claim.read_bytes()
    with pytest.raises(SystemExit) as error:
        collector.main()
    assert error.value.code == 2
    assert state.health_checks == 0 and state.payloads == []
    assert claim.read_bytes() == original
    assert not (state.service / "integration-sequence.json").exists()
    assert not (state.service / "stop.request").exists()


def test_receipt_publish_does_not_overwrite_and_is_complete_at_visibility(setup_collector, monkeypatch):
    collector, state = setup_collector
    path = state.service / "test-receipt.json"
    real_link = collector.os.link
    def checked_link(source, destination):
        assert json.loads(source.read_text()) == {"status": "complete"}
        assert not destination.exists()
        real_link(source, destination)
    monkeypatch.setattr(collector.os, "link", checked_link)
    collector.publish_receipt(path, {"status": "complete"})
    assert json.loads(path.read_text()) == {"status": "complete"}
    monkeypatch.setattr(collector.os, "link", real_link)
    with pytest.raises(FileExistsError):
        collector.publish_receipt(path, {"status": "must not replace"})
    assert json.loads(path.read_text()) == {"status": "complete"}
