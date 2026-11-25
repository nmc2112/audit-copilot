import json

from audit_copilot.agent import AuditAgent
from audit_copilot.llm import FakeLLM, LLMResponse, ToolCall


def call(name, args=None, cid="c1", raw=None, error=None):
    return LLMResponse(tool_calls=[ToolCall(id=cid, name=name, arguments=args or {},
                                            raw_arguments=raw or json.dumps(args or {}),
                                            parse_error=error)])


def report(entries, tool="weekend_entries"):
    return LLMResponse(content=json.dumps({
        "answer": "Weekend entries found.",
        "findings": [{"title": "Weekend postings", "risk": "medium", "tool": tool,
                      "entries": entries, "explanation": "Posted on a weekend."}],
        "tools_used": [tool], "limitations": ""}))


DONE = LLMResponse(content="Done.")


def test_happy_path(ledger_and_truth, settings):
    ledger, truth = ledger_and_truth
    llm = FakeLLM([call("weekend_entries"), DONE, report(truth["weekend_entries"])])
    result = AuditAgent(ledger, settings, llm).run("Weekend entries?")
    assert result.report_source == "llm"
    assert set(result.report.findings[0].entries) == set(truth["weekend_entries"])
    assert result.iterations == 2
    assert llm.calls[0]["tools"] and llm.calls[-1]["json_mode"]


def test_hallucinated_entry_is_rejected_then_fixed(ledger_and_truth, settings):
    ledger, truth = ledger_and_truth
    llm = FakeLLM([call("weekend_entries"), DONE,
                   report(["999999"]), report(truth["weekend_entries"])])
    result = AuditAgent(ledger, settings, llm).run("Weekend entries?")
    assert result.report_source == "llm"
    rejected = [t for t in result.trace if t["node"] == "report" and not t.get("valid")]
    assert rejected and "not returned by any tool" in rejected[0]["error"]


def test_invalid_json_falls_back_to_deterministic_report(ledger_and_truth, settings):
    ledger, truth = ledger_and_truth
    bad = LLMResponse(content="not json")
    llm = FakeLLM([call("weekend_entries"), DONE] + [bad] * settings.agent.report_max_attempts)
    result = AuditAgent(ledger, settings, llm).run("Weekend entries?")
    assert result.report_source == "deterministic_fallback"
    assert set(result.report.findings[0].entries) == set(truth["weekend_entries"])


def test_bad_tool_arguments_are_returned_to_the_model(ledger_and_truth, settings):
    ledger, truth = ledger_and_truth
    llm = FakeLLM([call("round_amounts", {"multiple": -1}), call("round_amounts", cid="c2"),
                   DONE, report(truth["round_amounts"], tool="round_amounts")])
    result = AuditAgent(ledger, settings, llm).run("Round amounts?")
    tool_steps = [t for t in result.trace if t["node"] == "tool"]
    assert [t["ok"] for t in tool_steps] == [False, True]
    assert result.report_source == "llm"


def test_unknown_tool_and_unparseable_arguments(ledger_and_truth, settings):
    ledger, _ = ledger_and_truth
    llm = FakeLLM([call("delete_ledger"), call("weekend_entries", raw="{oops", error="Invalid JSON"),
                   DONE, LLMResponse(content='{"answer": "nothing", "findings": [], "tools_used": []}')])
    result = AuditAgent(ledger, settings, llm).run("Do something")
    assert [t["ok"] for t in result.trace if t["node"] == "tool"] == [False, False]
    tool_messages = [m for m in llm.calls[-1]["messages"] if m["role"] == "tool"]
    assert "Unknown tool" in tool_messages[0]["content"]


def test_iteration_cap(ledger_and_truth, settings):
    ledger, _ = ledger_and_truth
    max_it = settings.agent.max_iterations
    script = [call("ledger_summary", cid=f"c{i}") for i in range(max_it + 1)]
    script.append(LLMResponse(content='{"answer": "summary", "findings": [], "tools_used": ["ledger_summary"]}'))
    result = AuditAgent(ledger, settings, FakeLLM(script)).run("Loop forever")
    assert result.iterations == max_it + 1
    assert len([t for t in result.trace if t["node"] == "tool"]) == max_it
