import json

from fastapi.testclient import TestClient

from audit_copilot.api import create_app
from audit_copilot.config import ROOT, load_settings
from audit_copilot.llm import FakeLLM, LLMResponse, ToolCall


def scripted():
    report = {"answer": "Overview done.", "findings": [], "tools_used": ["ledger_summary"]}
    return FakeLLM([LLMResponse(tool_calls=[ToolCall(id="1", name="ledger_summary")]),
                    LLMResponse(content="Done."), LLMResponse(content=json.dumps(report))])


def test_analyze_uses_bundled_sample_when_no_file():
    client = TestClient(create_app(llm=scripted()))
    resp = client.post("/analyze", data={"question": "Give me an overview"})
    assert resp.status_code == 200
    assert resp.json()["report_source"] == "llm"


def test_rate_limit(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_HOUR", "2")
    client = TestClient(create_app(llm=FakeLLM([])))
    codes = [client.post("/analyze", data={"question": "x"}).status_code for _ in range(3)]
    assert codes[-1] == 429 and 429 not in codes[:2]
    assert client.get("/health").status_code == 200        # GET is never limited


def test_cloud_config_uses_hosted_model_and_secret_env():
    cfg = load_settings(ROOT / "config" / "settings.cloud.yaml")
    assert cfg.llm.primary.base_url.startswith("https://")
    assert cfg.llm.primary.api_key_env == "LLM_API_KEY"
    assert cfg.llm.fallback is not None and cfg.llm.fallback.api_key_env == "LLM_API_KEY"


def test_gemini_thought_signature_is_echoed_back():
    from audit_copilot.agent import _assistant_message
    from audit_copilot.llm import LLMResponse, _parse_tool_calls

    message = {"tool_calls": [{"id": "c1", "type": "function",
                               "function": {"name": "benford_test", "arguments": "{}"},
                               "extra_content": {"google": {"thought_signature": "sig"}}}]}
    out = _assistant_message(LLMResponse(tool_calls=_parse_tool_calls(message)))
    assert out["tool_calls"][0]["extra_content"] == {"google": {"thought_signature": "sig"}}
    plain = _assistant_message(LLMResponse(tool_calls=_parse_tool_calls({"tool_calls": [
        {"id": "c2", "function": {"name": "benford_test", "arguments": "{}"}}]})))
    assert "extra_content" not in plain["tool_calls"][0]


def test_llm_outage_returns_503():
    from fastapi.testclient import TestClient

    from audit_copilot.api import create_app
    from audit_copilot.llm import LLMError, LLMResponse

    class DownLLM:
        def chat(self, messages, tools=None, json_mode=False) -> LLMResponse:
            raise LLMError("All LLM clients failed: HTTP 503")

    client = TestClient(create_app(llm=DownLLM()))
    resp = client.post("/analyze", data={"question": "Run a full audit"})
    assert resp.status_code == 503


def test_cloud_config_has_fallback_model():
    from audit_copilot.config import load_settings

    cfg = load_settings(ROOT / "config" / "settings.cloud.yaml")
    assert cfg.llm.fallback is not None and cfg.llm.fallback.model != cfg.llm.primary.model
