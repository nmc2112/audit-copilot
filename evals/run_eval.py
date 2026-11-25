"""Evaluate the agent on a synthetic FEC with known anomalies.

Metrics per run:
- tool_selection_accuracy: share of cases where every expected tool was called
- anomaly_recall / anomaly_precision: injected entries cited in the report
- llm_report_rate: share of reports that passed schema + grounding checks
- avg latency and tokens per question

Usage:
    python evals/run_eval.py                 # real model from config/settings.yaml
    python evals/run_eval.py --oracle        # scripted model, checks the harness (CI)
    python evals/run_eval.py --min-tool-accuracy 0.8   # fail (exit 1) below threshold
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_copilot.agent import AuditAgent  # noqa: E402
from audit_copilot.config import load_settings  # noqa: E402
from audit_copilot.llm import LLMResponse, ToolCall, build_client  # noqa: E402
from audit_copilot.synth import generate_fec  # noqa: E402


class OracleLLM:
    """Scripted model that calls the expected tools and cites their results.

    It does not measure model quality: it verifies that the graph, tools,
    grounding checks and metrics work end to end, without network access.
    """

    def __init__(self, expected_tools: list[str]):
        self.expected_tools = expected_tools

    def chat(self, messages, tools=None, json_mode=False) -> LLMResponse:
        if json_mode:
            results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
            findings = [{"title": r["summary"], "risk": "medium", "tool": r["tool"],
                         "entries": r["flagged_entries"], "explanation": r["summary"]}
                        for r in results if r.get("flagged_count")]
            report = {"answer": " ".join(r["summary"] for r in results), "findings": findings,
                      "tools_used": [r["tool"] for r in results], "limitations": ""}
            return LLMResponse(content=json.dumps(report), model="oracle")
        if any(m["role"] == "tool" for m in messages):
            return LLMResponse(content="I have enough information.", model="oracle")
        calls = [ToolCall(id=f"c{i}", name=name, arguments={}) for i, name in enumerate(self.expected_tools)]
        return LLMResponse(tool_calls=calls, model="oracle")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", action="store_true", help="use the scripted model (no network)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-tool-accuracy", type=float, default=None)
    args = parser.parse_args()

    settings = load_settings()
    ledger, truth = generate_fec(n_entries=800, seed=args.seed)
    cases = yaml.safe_load((ROOT / "evals" / "cases.yaml").read_text())
    client = None if args.oracle else build_client(settings.llm)

    rows = []
    for case in cases:
        llm = OracleLLM(case["expected_tools"]) if args.oracle else client
        start = time.perf_counter()
        result = AuditAgent(ledger, settings, llm).run(case["question"])
        called = [t["name"] for t in result.trace if t["node"] == "tool" and t.get("ok")]
        cited = {e for f in result.report.findings for e in f.entries}
        row = {
            "id": case["id"],
            "tools_ok": set(case["expected_tools"]) <= set(called),
            "extra_tool_calls": len([c for c in called if c not in case["expected_tools"]]),
            "report_source": result.report_source,
            "latency_s": round(time.perf_counter() - start, 2),
            "tokens": result.prompt_tokens + result.completion_tokens,
        }
        if case.get("anomaly"):
            expected = set(truth[case["anomaly"]])
            row["recall"] = len(cited & expected) / len(expected)
            row["precision"] = len(cited & expected) / len(cited) if cited else 0.0
        rows.append(row)
        print(json.dumps(row))

    anomaly_rows = [r for r in rows if "recall" in r]
    summary = {
        "mode": "oracle" if args.oracle else settings.llm.primary.model,
        "prompt_version": settings.prompt_version,
        "cases": len(rows),
        "tool_selection_accuracy": round(sum(r["tools_ok"] for r in rows) / len(rows), 3),
        "anomaly_recall": round(sum(r["recall"] for r in anomaly_rows) / len(anomaly_rows), 3),
        "anomaly_precision": round(sum(r["precision"] for r in anomaly_rows) / len(anomaly_rows), 3),
        "llm_report_rate": round(sum(r["report_source"] == "llm" for r in rows) / len(rows), 3),
        "avg_latency_s": round(sum(r["latency_s"] for r in rows) / len(rows), 2),
        "avg_tokens": round(sum(r["tokens"] for r in rows) / len(rows)),
    }
    print("\nSUMMARY", json.dumps(summary, indent=2))

    out_dir = ROOT / "evals" / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{summary['mode'].replace(':', '_')}_{int(time.time())}.json").write_text(
        json.dumps({"summary": summary, "cases": rows}, indent=2))

    if args.min_tool_accuracy is not None and summary["tool_selection_accuracy"] < args.min_tool_accuracy:
        print(f"FAIL: tool selection accuracy below {args.min_tool_accuracy}")
        sys.exit(1)


if __name__ == "__main__":
    main()
