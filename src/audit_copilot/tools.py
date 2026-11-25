"""Deterministic audit tools the agent can call.

Each tool takes validated arguments (a Pydantic model), runs a classic
journal-entry test on the ledger and returns a compact ToolResult. Results are
capped so the LLM never receives thousands of rows (token and cost control).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    tool: str
    flagged_count: int = 0
    flagged_entries: list[str] = Field(default_factory=list, description="EcritureNum values, capped")
    summary: str
    details: dict = Field(default_factory=dict)


# ---------- argument models ----------

class NoArgs(BaseModel):
    pass


class WeekendArgs(BaseModel):
    include_saturday: bool = Field(True, description="Flag Saturday postings as well as Sunday")


class RoundAmountArgs(BaseModel):
    min_amount: float = Field(10000, ge=0, description="Only consider lines at or above this amount (EUR)")
    multiple: float = Field(1000, gt=0, description="An amount is round if it is a multiple of this value")


class UnbalancedArgs(BaseModel):
    tolerance: float = Field(0.01, ge=0, description="Allowed debit/credit difference per entry (EUR)")


class LateValidationArgs(BaseModel):
    max_days: int = Field(30, ge=0, description="Maximum normal delay between entry date and validation date")


# ---------- helpers ----------

def _entry_amounts(ledger: pd.DataFrame) -> pd.DataFrame:
    return ledger.groupby("EcritureNum").agg(
        debit=("Debit", "sum"), credit=("Credit", "sum"),
        date=("EcritureDate", "first"), valid=("ValidDate", "first"),
        piece=("PieceRef", "first"), journal=("JournalCode", "first"),
    )


def _cap(nums: list[str], max_items: int) -> list[str]:
    return sorted(set(nums))[:max_items]


# ---------- tools ----------

def ledger_summary(ledger: pd.DataFrame, args: NoArgs, max_items: int) -> ToolResult:
    entries = _entry_amounts(ledger)
    return ToolResult(
        tool="ledger_summary",
        summary=(f"{len(ledger)} lines, {len(entries)} entries from "
                 f"{ledger['EcritureDate'].min():%Y-%m-%d} to {ledger['EcritureDate'].max():%Y-%m-%d}."),
        details={
            "journals": ledger.groupby("JournalCode")["EcritureNum"].nunique().to_dict(),
            "total_debit": round(float(ledger["Debit"].sum()), 2),
            "total_credit": round(float(ledger["Credit"].sum()), 2),
        },
    )


def weekend_entries(ledger: pd.DataFrame, args: WeekendArgs, max_items: int) -> ToolResult:
    days = {6} | ({5} if args.include_saturday else set())
    hits = ledger.loc[ledger["EcritureDate"].dt.weekday.isin(days), "EcritureNum"].unique().tolist()
    return ToolResult(tool="weekend_entries", flagged_count=len(hits), flagged_entries=_cap(hits, max_items),
                      summary=f"{len(hits)} entries posted on a weekend.")


def round_amounts(ledger: pd.DataFrame, args: RoundAmountArgs, max_items: int) -> ToolResult:
    amount = ledger[["Debit", "Credit"]].max(axis=1)
    is_round = (amount >= args.min_amount) & (np.isclose(np.mod(amount, args.multiple), 0))
    hits = ledger.loc[is_round, "EcritureNum"].unique().tolist()
    return ToolResult(tool="round_amounts", flagged_count=len(hits), flagged_entries=_cap(hits, max_items),
                      summary=f"{len(hits)} entries with round amounts >= {args.min_amount:,.0f} EUR.")


def duplicate_entries(ledger: pd.DataFrame, args: NoArgs, max_items: int) -> ToolResult:
    entries = _entry_amounts(ledger).reset_index()
    keys = ["piece", "date", "debit", "journal"]
    dup = entries[entries.duplicated(subset=keys, keep=False)]
    hits = dup["EcritureNum"].tolist()
    groups = int(dup.groupby(keys).ngroups) if not dup.empty else 0
    return ToolResult(tool="duplicate_entries", flagged_count=len(hits), flagged_entries=_cap(hits, max_items),
                      summary=f"{groups} groups of possible duplicate entries ({len(hits)} entries).")


def unbalanced_entries(ledger: pd.DataFrame, args: UnbalancedArgs, max_items: int) -> ToolResult:
    entries = _entry_amounts(ledger)
    gap = (entries["debit"] - entries["credit"]).abs()
    hits = entries.index[gap > args.tolerance].tolist()
    return ToolResult(tool="unbalanced_entries", flagged_count=len(hits), flagged_entries=_cap(hits, max_items),
                      summary=f"{len(hits)} entries where debit and credit differ by more than {args.tolerance} EUR.",
                      details={"total_gap": round(float(gap[gap > args.tolerance].sum()), 2)})


def late_validation(ledger: pd.DataFrame, args: LateValidationArgs, max_items: int) -> ToolResult:
    entries = _entry_amounts(ledger)
    delay = (entries["valid"] - entries["date"]).dt.days
    hits = entries.index[delay > args.max_days].tolist()
    return ToolResult(tool="late_validation", flagged_count=len(hits), flagged_entries=_cap(hits, max_items),
                      summary=f"{len(hits)} entries validated more than {args.max_days} days after their date.")


BENFORD = {d: math.log10(1 + 1 / d) for d in range(1, 10)}


def benford_test(ledger: pd.DataFrame, args: NoArgs, max_items: int) -> ToolResult:
    amounts = ledger[["Debit", "Credit"]].max(axis=1)
    amounts = amounts[amounts >= 10]
    first = amounts.astype(str).str.lstrip("0.").str[0].astype(int)
    observed = first.value_counts(normalize=True).reindex(range(1, 10), fill_value=0.0)
    mad = float(np.mean([abs(observed[d] - BENFORD[d]) for d in range(1, 10)]))
    # Nigrini's first-digit thresholds for the mean absolute deviation.
    if mad < 0.006:
        verdict = "close conformity"
    elif mad < 0.012:
        verdict = "acceptable conformity"
    elif mad < 0.015:
        verdict = "marginally acceptable conformity"
    else:
        verdict = "nonconformity"
    return ToolResult(tool="benford_test", summary=f"First-digit MAD = {mad:.4f} ({verdict}).",
                      details={"mad": round(mad, 4), "verdict": verdict,
                               "observed": {str(k): round(float(v), 3) for k, v in observed.items()}})


class Tool(BaseModel):
    name: str
    description: str
    args_model: type[BaseModel]
    func: Callable[..., ToolResult]

    model_config = {"arbitrary_types_allowed": True}

    def openai_schema(self) -> dict:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                  "parameters": schema}}

    def run(self, ledger: pd.DataFrame, raw_args: dict, max_items: int) -> ToolResult:
        args = self.args_model.model_validate(raw_args or {})
        return self.func(ledger, args, max_items)


ALL_TOOLS: dict[str, Tool] = {
    t.name: t for t in [
        Tool(name="ledger_summary", description="Overview of the ledger: period, number of entries, journals, totals.",
             args_model=NoArgs, func=ledger_summary),
        Tool(name="weekend_entries", description="Find journal entries posted on Saturday or Sunday.",
             args_model=WeekendArgs, func=weekend_entries),
        Tool(name="round_amounts", description="Find large entries with suspiciously round amounts.",
             args_model=RoundAmountArgs, func=round_amounts),
        Tool(name="duplicate_entries", description="Find possible duplicates (same piece, date, amount, journal).",
             args_model=NoArgs, func=duplicate_entries),
        Tool(name="unbalanced_entries", description="Find entries whose total debit differs from total credit.",
             args_model=UnbalancedArgs, func=unbalanced_entries),
        Tool(name="late_validation", description="Find entries validated long after their accounting date.",
             args_model=LateValidationArgs, func=late_validation),
        Tool(name="benford_test", description="Benford first-digit test on amounts, with Nigrini conformity level.",
             args_model=NoArgs, func=benford_test),
    ]
}


def get_tools(enabled: list[str] | None = None) -> dict[str, Tool]:
    """Return the tools enabled in the configuration (feature flags)."""
    if not enabled:
        return dict(ALL_TOOLS)
    unknown = set(enabled) - set(ALL_TOOLS)
    if unknown:
        raise ValueError(f"Unknown tools in config: {sorted(unknown)}")
    return {name: ALL_TOOLS[name] for name in enabled}
