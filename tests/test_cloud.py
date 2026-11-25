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
    assert cfg.llm.primary.api_key_env == "MISTRAL_API_KEY"
    assert cfg.llm.fallback is None
