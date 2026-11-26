"""Evaluate the agent on synthetic FEC ledgers (evaluation set v2, evals/cases.yaml).

The tools are deterministic, so the ground truth for a case is what the expected
tools return with the expected arguments. The eval therefore measures the agent:

- tool_selection:   every expected tool was called
- argument_accuracy: thresholds stated in the question reached the tool (cases with expected_args)
- entry_recall / entry_precision: entries cited in the report vs ground truth (micro-averaged)
- false_alarm_rate: reports citing entries when there is nothing to find (clean ledger, overview, Benford)
- out_of_scope_ok:  no audit tool called and no entry cited when the tools cannot answer
- llm_report_rate:  reports that passed schema + grounding checks (vs deterministic fallback)

Usage:
    python evals/run_eval.py                      # model from config (AUDIT_COPILOT_CONFIG or settings.yaml)
    python evals/run_eval.py --oracle             # scripted model: checks the harness, no network (CI)
    python evals/run_eval.py --only arguments,clean   # subset of categories
    python evals/run_eval.py --min-tool-accuracy 0.8  # exit 1 below threshold
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_copilot.agent import AuditAgent  # noqa: E402
from audit_copilot.config import load_settings  # noqa: E402
from audit_copilot.llm import LLMResponse, ToolCall, build_client  # noqa: E402
from audit_copilot.synth import generate_fec  # noqa: E402
from audit_copilot.tools import get_tools  # noqa: E402

LEDGERS = {
    "standard": {"n_entries": 800, "seed": 7},
    "sparse": {"n_entries": 800, "seed": 11, "n_per_anomaly": 2},
    "clean": {"n_entries": 800, "seed": 21, "n_per_anomaly": 0},
}
ENTRY_TOOLS = {"weekend_entries", "round_amounts", "duplicate_entries", "unbalanced_entries", "late_validation",
               "numbering_gaps", "period_end_entries", "suspense_accounts", "vague_entries"}


class OracleLLM:
    """Scripted model that calls the expected tools (with expected arguments) and cites their results.

    It does not measure model quality: it checks that the graph, tools, grounding checks
    and metrics work end to end, without network access.
    """

    def __init__(self, expected_tools: list[str], expected_args: dict[str, dict]):
        self.expected_tools, self.expected_args = expected_tools, expected_args

    def chat(self, messages, tools=None, json_mode=False) -> LLMResponse:
        if json_mode:
            results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
            findings = [{"title": r["summary"], "risk": "medium", "tool": r["tool"],
                         "entries": r["flagged_entries"], "explanation": r["summary"]}
                        for r in results if r.get("flagged_count")]
            answer = " ".join(r["summary"] for r in results) or "The available tools cannot answer this question."
            report = {"answer": answer, "findings": findings,
                      "tools_used": [r["tool"] for r in results], "limitations": ""}
            return LLMResponse(content=json.dumps(report), model="oracle")
        if any(m["role"] == "tool" for m in messages) or not self.expected_tools:
            return LLMResponse(content="I have enough information.", model="oracle")
        calls = [ToolCall(id=f"c{i}", name=name, arguments=self.expected_args.get(name, {}))
                 for i, name in enumerate(self.expected_tools)]
        return LLMResponse(tool_calls=calls, model="oracle")


def _same(expected, actual) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return bool(expected) == bool(actual)
    try:
        return math.isclose(float(expected), float(actual), rel_tol=1e-6)
    except (TypeError, ValueError):
        return expected == actual


def ground_truth(ledger: pd.DataFrame, case: dict) -> set[str]:
    tools = get_tools()
    expected: set[str] = set()
    for name in case["expected_tools"]:
        if name in ENTRY_TOOLS:
            args = case.get("expected_args", {}).get(name, {})
            expected |= set(tools[name].run(ledger, args, 10**6).flagged_entries)
    return expected


def score(case: dict, result, truth: set[str]) -> dict:
    tool_calls = [t for t in result.trace if t["node"] == "tool" and t.get("ok")]
    called = {t["name"] for t in tool_calls}
    cited = {e for f in result.report.findings for e in f.entries}
    expected_tools = set(case["expected_tools"])
    row: dict = {
        "id": case["id"], "category": case["category"],
        "tools_ok": expected_tools <= called,
        "extra_tools": sorted((called & ENTRY_TOOLS) - expected_tools),
        "report_source": result.report_source,
        "tokens": result.prompt_tokens + result.completion_tokens,
    }
    if case.get("expected_args"):
        ok = True
        for tool, args in case["expected_args"].items():
            calls = [t.get("args") or {} for t in tool_calls if t["name"] == tool]
            ok &= any(all(_same(v, c.get(k)) for k, v in args.items()) for c in calls)
        row["args_ok"] = ok
    if truth:
        hit = len(cited & truth)
        row.update(expected=len(truth), cited=len(cited), hit=hit)
    else:
        row["false_alarm"] = bool(cited)
    if case["category"] == "out_of_scope":
        row["out_of_scope_ok"] = not (called & ENTRY_TOOLS) and not cited
    return row


def summarize(rows: list[dict]) -> dict:
    def rate(key: str, subset: list[dict]) -> float | None:
        vals = [r[key] for r in subset if key in r]
        return round(sum(vals) / len(vals), 3) if vals else None

    entry_rows = [r for r in rows if "expected" in r]
    hits = sum(r["hit"] for r in entry_rows)
    cited = sum(r["cited"] for r in entry_rows)
    expected = sum(r["expected"] for r in entry_rows)
    return {
        "cases": len(rows),
        "tool_selection": rate("tools_ok", rows),
        "argument_accuracy": rate("args_ok", rows),
        "entry_recall": round(hits / expected, 3) if expected else None,
        "entry_precision": round(hits / cited, 3) if cited else None,
        "false_alarm_rate": rate("false_alarm", rows),
        "out_of_scope_ok": rate("out_of_scope_ok", rows),
        "cases_with_extra_tools": sum(bool(r["extra_tools"]) for r in rows),
        "llm_report_rate": round(sum(r["report_source"] == "llm" for r in rows) / len(rows), 3),
        "avg_latency_s": round(sum(r["latency_s"] for r in rows) / len(rows), 2),
        "avg_tokens": round(sum(r["tokens"] for r in rows) / len(rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", action="store_true", help="use the scripted model (no network)")
    parser.add_argument("--only", default="", help="comma-separated categories to run")
    parser.add_argument("--min-tool-accuracy", type=float, default=None)
    args = parser.parse_args()

    settings = load_settings()
    cases = yaml.safe_load((ROOT / "evals" / "cases.yaml").read_text(encoding="utf-8"))
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["category"] in wanted]
    ledgers = {name: generate_fec(**params)[0] for name, params in LEDGERS.items()}
    client = None if args.oracle else build_client(settings.llm)
    mode = "oracle" if args.oracle else settings.llm.primary.model

    rows = []
    for i, case in enumerate(cases, 1):
        ledger = ledgers[case["ledger"]]
        llm = OracleLLM(case["expected_tools"], case.get("expected_args", {})) if args.oracle else client
        start = time.perf_counter()
        try:
            result = AuditAgent(ledger, settings, llm).run(case["question"])
        except Exception as exc:  # a crash is a failed case, not a failed eval
            print(f"[{i}/{len(cases)}] {case['id']} ERROR {exc}")
            rows.append({"id": case["id"], "category": case["category"], "tools_ok": False, "extra_tools": [],
                         "report_source": "error", "tokens": 0, "latency_s": round(time.perf_counter() - start, 2),
                         "error": str(exc)[:200]})
            continue
        row = score(case, result, ground_truth(ledger, case))
        row["latency_s"] = round(time.perf_counter() - start, 2)
        rows.append(row)
        flags = " ".join(f"{k}={row[k]}" for k in ("tools_ok", "args_ok", "hit", "expected", "false_alarm",
                                                    "out_of_scope_ok") if k in row)
        print(f"[{i}/{len(cases)}] {case['id']:<16} {flags} {row['latency_s']}s", flush=True)

    summary = {"mode": mode, "prompt_version": settings.prompt_version, **summarize(rows)}
    by_category = defaultdict(list)
    for r in rows:
        by_category[r["category"]].append(r)
    per_category = {cat: summarize(rs) for cat, rs in by_category.items()}

    print("\nSUMMARY", json.dumps(summary, indent=2))
    print(f"\n{'category':<13} {'n':>3} {'tools':>6} {'args':>6} {'recall':>7} {'prec':>6} {'f.alarm':>8}")
    for cat, s in per_category.items():
        def fmt(v):
            return "-" if v is None else f"{v:.2f}"
        print(f"{cat:<13} {s['cases']:>3} {fmt(s['tool_selection']):>6} {fmt(s['argument_accuracy']):>6} "
              f"{fmt(s['entry_recall']):>7} {fmt(s['entry_precision']):>6} {fmt(s['false_alarm_rate']):>8}")

    out_dir = ROOT / "evals" / "results"
    out_dir.mkdir(exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9._+-]+", "_", mode)
    (out_dir / f"{slug}_{int(time.time())}.json").write_text(
        json.dumps({"summary": summary, "per_category": per_category, "cases": rows}, indent=2))

    if args.min_tool_accuracy is not None and (summary["tool_selection"] or 0) < args.min_tool_accuracy:
        print(f"FAIL: tool selection below {args.min_tool_accuracy}")
        sys.exit(1)


if __name__ == "__main__":
    main()
