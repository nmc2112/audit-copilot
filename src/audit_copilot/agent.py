"""LangGraph orchestration: plan -> call tools -> (repeat) -> structured report.

Guards that make the agent reliable:
- a cap on tool-calling rounds;
- tool arguments validated with Pydantic, errors returned to the model;
- the final report must match a Pydantic schema (retried with the error);
- grounding check: every cited entry must come from a tool result;
- if the model still fails, a deterministic report is built from tool results.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Literal, TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError

from .config import Settings
from .llm import LLMClient, LLMError, LLMResponse
from .tools import Tool, ToolResult, get_tools

log = logging.getLogger("audit_copilot.agent")


class Finding(BaseModel):
    title: str
    risk: Literal["low", "medium", "high"]
    tool: str = Field(description="Name of the tool that produced the evidence")
    entries: list[str] = Field(default_factory=list, description="EcritureNum values cited as evidence")
    explanation: str


class AuditReport(BaseModel):
    answer: str = Field(description="Direct answer to the auditor's question")
    findings: list[Finding] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    limitations: str = ""


class AgentResult(BaseModel):
    question: str
    report: AuditReport
    report_source: Literal["llm", "deterministic_fallback"]
    run_id: str
    prompt_version: str
    iterations: int
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    trace: list[dict]


class State(TypedDict, total=False):
    messages: list[dict]
    iterations: int
    pending_calls: list
    tool_results: list[dict]
    report: dict | None
    report_source: str
    trace: list[dict]


class GroundingError(ValueError):
    pass


def check_grounding(report: AuditReport, tool_results: list[ToolResult]) -> None:
    """Reject reports that cite entries or tools the agent never saw."""
    seen_entries = {e for r in tool_results for e in r.flagged_entries}
    seen_tools = {r.tool for r in tool_results}
    for f in report.findings:
        if f.tool not in seen_tools:
            raise GroundingError(f"Finding '{f.title}' cites tool '{f.tool}' that was not called")
        unknown = set(f.entries) - seen_entries
        if unknown:
            raise GroundingError(f"Finding '{f.title}' cites entries not returned by any tool: {sorted(unknown)[:5]}")


def deterministic_report(tool_results: list[ToolResult]) -> AuditReport:
    findings = [
        Finding(title=r.summary, risk="medium", tool=r.tool, entries=r.flagged_entries,
                explanation="Generated from tool output because the model did not return a valid report.")
        for r in tool_results if r.flagged_count > 0
    ]
    return AuditReport(answer="; ".join(r.summary for r in tool_results) or "No tool was run.",
                       findings=findings, tools_used=sorted({r.tool for r in tool_results}),
                       limitations="Automatic summary of tool results; no LLM interpretation.")


def _assistant_message(resp: LLMResponse) -> dict:
    msg: dict = {"role": "assistant", "content": resp.content or ""}
    if resp.tool_calls:
        calls = []
        for tc in resp.tool_calls:
            call: dict = {"id": tc.id, "type": "function",
                          "function": {"name": tc.name, "arguments": tc.raw_arguments or json.dumps(tc.arguments)}}
            if tc.extra_content:  # Gemini requires its thought signature to be sent back
                call["extra_content"] = tc.extra_content
            calls.append(call)
        msg["tool_calls"] = calls
    return msg


class AuditAgent:
    def __init__(self, ledger: pd.DataFrame, settings: Settings, llm: LLMClient):
        self.ledger, self.settings, self.llm = ledger, settings, llm
        self.tools: dict[str, Tool] = get_tools(settings.tools.enabled)
        self.tool_schemas = [t.openai_schema() for t in self.tools.values()]
        self.graph = self._build_graph()

    # ----- nodes -----
    def _log(self, state: State, node: str, resp: LLMResponse | None = None, **extra) -> None:
        entry = {"node": node, **extra}
        if resp is not None:
            entry.update(model=resp.model, prompt_tokens=resp.prompt_tokens,
                         completion_tokens=resp.completion_tokens, latency_s=resp.latency_s)
        state["trace"].append(entry)
        log.info(json.dumps(entry, default=str))

    def plan(self, state: State) -> State:
        resp = self.llm.chat(state["messages"], tools=self.tool_schemas)
        state["messages"].append(_assistant_message(resp))
        state["pending_calls"] = resp.tool_calls
        state["iterations"] = state.get("iterations", 0) + 1
        self._log(state, "plan", resp, tool_calls=[tc.name for tc in resp.tool_calls])
        return state

    def route(self, state: State) -> str:
        if state.get("pending_calls") and state["iterations"] <= self.settings.agent.max_iterations:
            return "tools"
        return "report"

    def run_tools(self, state: State) -> State:
        for tc in state.get("pending_calls", []):
            start = time.perf_counter()
            if tc.parse_error:
                content, ok = json.dumps({"error": tc.parse_error}), False
            elif tc.name not in self.tools:
                content, ok = json.dumps({"error": f"Unknown tool '{tc.name}'",
                                          "available": list(self.tools)}), False
            else:
                try:
                    result = self.tools[tc.name].run(self.ledger, tc.arguments,
                                                     self.settings.agent.max_items_per_tool)
                    state["tool_results"].append(result.model_dump())
                    content, ok = result.model_dump_json(), True
                except ValidationError as exc:
                    content, ok = json.dumps({"error": f"Invalid arguments: {exc.errors()}"}), False
            state["messages"].append({"role": "tool", "tool_call_id": tc.id, "content": content})
            self._log(state, "tool", name=tc.name, ok=ok,
                      latency_s=round(time.perf_counter() - start, 3))
        state["pending_calls"] = []
        return state

    def write_report(self, state: State) -> State:
        results = [ToolResult.model_validate(r) for r in state["tool_results"]]
        schema = json.dumps(AuditReport.model_json_schema(), ensure_ascii=False)
        messages = state["messages"] + [{"role": "user",
                                         "content": self.settings.prompt("report").replace("{schema}", schema)}]
        for attempt in range(1, self.settings.agent.report_max_attempts + 1):
            try:
                resp = self.llm.chat(messages, json_mode=True)
            except LLMError as exc:
                self._log(state, "report", attempt=attempt, error=str(exc))
                break
            try:
                report = AuditReport.model_validate_json(resp.content)
                check_grounding(report, results)
                self._log(state, "report", resp, attempt=attempt, valid=True)
                state["report"], state["report_source"] = report.model_dump(), "llm"
                return state
            except (ValidationError, GroundingError) as exc:
                self._log(state, "report", resp, attempt=attempt, valid=False, error=str(exc)[:300])
                messages = messages + [
                    {"role": "assistant", "content": resp.content},
                    {"role": "user", "content": f"Your JSON was rejected: {str(exc)[:500]}. "
                                                "Fix it and return valid JSON only."},
                ]
        state["report"] = deterministic_report(results).model_dump()
        state["report_source"] = "deterministic_fallback"
        return state

    def _build_graph(self):
        g = StateGraph(State)
        g.add_node("plan", self.plan)
        g.add_node("tools", self.run_tools)
        g.add_node("report", self.write_report)
        g.set_entry_point("plan")
        g.add_conditional_edges("plan", self.route, {"tools": "tools", "report": "report"})
        g.add_edge("tools", "plan")
        g.add_edge("report", END)
        return g.compile()

    # ----- public API -----
    def run(self, question: str) -> AgentResult:
        start = time.perf_counter()
        state: State = {
            "messages": [{"role": "system", "content": self.settings.prompt("system")},
                         {"role": "user", "content": question}],
            "iterations": 0, "tool_results": [], "trace": [], "report": None,
        }
        final = self.graph.invoke(state, config={"recursion_limit": 4 * self.settings.agent.max_iterations + 4})
        trace = final["trace"]
        return AgentResult(
            question=question,
            report=AuditReport.model_validate(final["report"]),
            report_source=final["report_source"],
            run_id=uuid.uuid4().hex[:12],
            prompt_version=self.settings.prompt_version,
            iterations=final["iterations"],
            prompt_tokens=sum(t.get("prompt_tokens", 0) for t in trace),
            completion_tokens=sum(t.get("completion_tokens", 0) for t in trace),
            latency_s=round(time.perf_counter() - start, 3),
            trace=trace,
        )
