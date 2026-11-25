import json

from fastapi.testclient import TestClient

from audit_copilot.api import create_app
from audit_copilot.fec import write_fec
from audit_copilot.llm import FakeLLM, LLMResponse, ToolCall


def test_health():
    client = TestClient(create_app(llm=FakeLLM([])))
    assert client.get("/health").json()["status"] == "ok"


def test_analyze_end_to_end(tmp_path, ledger_and_truth):
    ledger, truth = ledger_and_truth
    path = tmp_path / "fec.txt"
    write_fec(ledger, path)
    report = {"answer": "Duplicates found.", "tools_used": ["duplicate_entries"], "findings": [
        {"title": "Duplicates", "risk": "high", "tool": "duplicate_entries",
         "entries": truth["duplicate_entries"], "explanation": "Same piece, date and amount."}]}
    llm = FakeLLM([LLMResponse(tool_calls=[ToolCall(id="1", name="duplicate_entries")]),
                   LLMResponse(content="Done."), LLMResponse(content=json.dumps(report))])
    client = TestClient(create_app(llm=llm))
    with path.open("rb") as fh:
        resp = client.post("/analyze", data={"question": "Any duplicate invoices?"},
                           files={"fec": ("fec.txt", fh, "text/plain")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["report_source"] == "llm"
    assert set(body["report"]["findings"][0]["entries"]) == set(truth["duplicate_entries"])


def test_invalid_fec_returns_422():
    client = TestClient(create_app(llm=FakeLLM([])))
    resp = client.post("/analyze", data={"question": "check"},
                       files={"fec": ("x.txt", b"a\tb\n1\t2\n", "text/plain")})
    assert resp.status_code == 422
