import http.client
import io
import json
import urllib.error

import pytest

from src.llm.backends.openai_compatible import OpenAICompatibleBackend
from src.llm.output_parser import parse_llm_json


def test_output_parser_strips_json_fence():
    parsed = parse_llm_json('```json\n{"ok": true}\n```')

    assert parsed == {"ok": True}


def test_output_parser_extracts_first_json_object():
    parsed = parse_llm_json(
        'Here is the JSON:\n{"robot_trees": {"robot_0": "<root attr=\\"{goal}\\"/>"}}\nDone.'
    )

    assert parsed["robot_trees"]["robot_0"] == '<root attr="{goal}"/>'


def test_openai_profile_opencode_disables_response_format(monkeypatch):
    monkeypatch.setenv("OPENAI_PROVIDER_PROFILE", "opencode-zen")
    backend = OpenAICompatibleBackend(
        base_url="https://opencode.ai/zen/v1",
        api_key="secret",
        model="minimax-m2.5-free",
    )

    payload = backend.build_payload("prompt")
    headers = backend.build_headers()

    assert payload == {
        "model": "minimax-m2.5-free",
        "messages": [{"role": "user", "content": "prompt"}],
        "temperature": 0,
        "max_tokens": 4096,
        "stream": False,
    }
    assert "response_format" not in payload
    assert headers["Accept"] == "application/json"
    assert headers["Accept-Encoding"] == "identity"
    assert headers["Connection"] == "close"


def test_openai_disable_response_format_env(monkeypatch):
    monkeypatch.setenv("OPENAI_DISABLE_RESPONSE_FORMAT", "1")
    backend = OpenAICompatibleBackend(
        base_url="https://api.example.com",
        api_key="secret",
        model="model-a",
        provider_profile="default",
    )

    payload = backend.build_payload("prompt")

    assert "response_format" not in payload


def test_openai_backend_retries_incomplete_read(monkeypatch):
    calls = {"count": 0}

    def fake_urlopen(request, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            raise http.client.IncompleteRead(b'{"choices":')
        return DummyResponse(
            {
                "choices": [
                    {"message": {"content": '{"robot_trees": {}, "assignment": []}'}}
                ]
            }
        )

    monkeypatch.setattr(
        "src.llm.backends.openai_compatible.urllib.request.urlopen", fake_urlopen
    )
    monkeypatch.setattr("src.llm.backends.openai_compatible.time.sleep", lambda _: None)
    backend = OpenAICompatibleBackend(
        base_url="https://api.example.com",
        api_key="secret",
        model="model-a",
        max_retries=1,
    )

    assert backend.generate("prompt") == '{"robot_trees": {}, "assignment": []}'
    assert calls["count"] == 2


def test_openai_backend_does_not_print_api_key(monkeypatch, tmp_path):
    secret = "sk-secret-value"

    def fake_urlopen(request, timeout):
        raise http_error(403, '{"error":"forbidden"}')

    monkeypatch.setenv("OPENAI_DEBUG_REQUEST", "1")
    monkeypatch.setattr(
        "src.llm.backends.openai_compatible.urllib.request.urlopen", fake_urlopen
    )
    backend = OpenAICompatibleBackend(
        base_url="https://api.example.com",
        api_key=secret,
        model="model-a",
        max_retries=0,
        debug_dir=tmp_path,
    )

    with pytest.raises(RuntimeError) as exc_info:
        backend.generate("prompt")

    assert secret not in str(exc_info.value)
    for path in tmp_path.glob("openai_*_last.*"):
        assert secret not in path.read_text(encoding="utf-8")


def test_model_fallbacks_are_attempted(monkeypatch):
    attempted = []

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        attempted.append(payload["model"])
        if payload["model"] == "bad-model":
            raise http_error(403, '{"error":"model denied"}')
        return DummyResponse({"choices": [{"message": {"content": '{"ok": true}'}}]})

    monkeypatch.setenv("OPENAI_MODEL_FALLBACKS", "fallback-a,fallback-b")
    monkeypatch.setattr(
        "src.llm.backends.openai_compatible.urllib.request.urlopen", fake_urlopen
    )
    backend = OpenAICompatibleBackend(
        base_url="https://api.example.com",
        api_key="secret",
        model="bad-model",
        max_retries=0,
    )

    assert backend.generate("prompt") == '{"ok": true}'
    assert attempted == ["bad-model", "fallback-a"]


class DummyResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def http_error(status, body):
    return urllib.error.HTTPError(
        url="https://api.example.com/chat/completions",
        code=status,
        msg="error",
        hdrs={},
        fp=io.BytesIO(body.encode("utf-8")),
    )
