import httpx
import pytest

from audit_copilot.config import LLMConfig, ModelConfig
from audit_copilot.llm import FakeLLM, FallbackClient, LLMError, LLMResponse, OpenAICompatibleClient

OK_BODY = {"model": "m", "usage": {"prompt_tokens": 10, "completion_tokens": 5},
           "choices": [{"message": {"content": "", "tool_calls": [
               {"id": "1", "function": {"name": "weekend_entries", "arguments": "{\"include_saturday\": false}"}}]}}]}


def make_client(handler, retries=3):
    cfg = LLMConfig(primary=ModelConfig(base_url="http://test/v1", model="m"), max_retries=retries, backoff_s=0.01)
    return OpenAICompatibleClient(cfg.primary, cfg, transport=httpx.MockTransport(handler), sleep=lambda s: None)


def test_retries_transient_errors_then_succeeds():
    statuses = iter([503, 429, 200])

    def handler(request):
        status = next(statuses)
        return httpx.Response(status, json=OK_BODY if status == 200 else {"error": "busy"})

    resp = make_client(handler).chat([{"role": "user", "content": "hi"}])
    assert resp.tool_calls[0].arguments == {"include_saturday": False}
    assert resp.prompt_tokens == 10


def test_gives_up_after_max_retries():
    with pytest.raises(LLMError):
        make_client(lambda r: httpx.Response(503, json={}), retries=2).chat([])


def test_client_error_is_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, json={"error": "bad request"})

    with pytest.raises(LLMError):
        make_client(handler).chat([])
    assert len(calls) == 1


def test_invalid_tool_arguments_are_reported():
    body = {"choices": [{"message": {"tool_calls": [{"id": "1", "function": {"name": "x", "arguments": "{bad"}}]}}]}
    resp = make_client(lambda r: httpx.Response(200, json=body)).chat([])
    assert resp.tool_calls[0].parse_error


class Broken:
    def chat(self, *a, **k):
        raise LLMError("down")


def test_fallback_uses_next_client():
    backup = FakeLLM([LLMResponse(content="from backup")])
    assert FallbackClient([Broken(), backup]).chat([]).content == "from backup"


def test_fallback_raises_when_all_fail():
    with pytest.raises(LLMError):
        FallbackClient([Broken(), Broken()]).chat([])
