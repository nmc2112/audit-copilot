"""Deterministic audit tools the agent can call.

Each tool takes validated arguments (a Pydantic model), runs a classic
journal-entry test on the ledger and returns a compact ToolResult. Results are
capped so the LLM never receives thousands of rows (token and cost control).
"""

from __future__ import annotations

import math
import re
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


class PeriodEndArgs(BaseModel):
    days: int = Field(5, ge=1, le=60, description="Number of days before the fiscal year end to examine")
    fiscal_year_end: str = Field("12-31", pattern=r"^\d{2}-\d{2}$", description="Fiscal year end as MM-DD")
    manual_only: bool = Field(True, description="Only manual journals (OD); false = all journals")


class SuspenseArgs(BaseModel):
    prefixes: list[str] = Field(default_factory=lambda: ["471", "472", "473"],
                                description="Account prefixes treated as suspense accounts (comptes d'attente)")
    max_uses: int = Field(2, ge=0, description="Also flag accounts used in at most this many lines (0 = off)")


class VagueArgs(BaseModel):
    keywords: list[str] = Field(default_factory=lambda: ["divers", "ajustement", "annulation", "correction",
                                                         "à régulariser"],
                                description="Labels considered vague (case-insensitive, whole words)")


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


def numbering_gaps(ledger: pd.DataFrame, args: NoArgs, max_items: int) -> ToolResult:
    parsed = pd.to_numeric(pd.Series(ledger["EcritureNum"].unique()), errors="coerce").dropna().astype(int)
    nums = np.sort(parsed.to_numpy())
    if len(nums) < 2:
        return ToolResult(tool="numbering_gaps", summary="Not enough numbered entries to check the sequence.")
    width = int(ledger["EcritureNum"].astype(str).str.len().max())
    breaks = np.where(np.diff(nums) > 1)[0]
    hits = [str(nums[i]).zfill(width) for i in breaks]
    missing = int(sum(nums[i + 1] - nums[i] - 1 for i in breaks))
    return ToolResult(tool="numbering_gaps", flagged_count=len(hits), flagged_entries=_cap(hits, max_items),
                      summary=f"{len(hits)} gaps in the EcritureNum sequence ({missing} missing numbers); "
                              "flagged entries are those just before each gap.",
                      details={"missing_numbers": missing})


def period_end_entries(ledger: pd.DataFrame, args: PeriodEndArgs, max_items: int) -> ToolResult:
    year = int(ledger["EcritureDate"].max().year)
    month, day = (int(x) for x in args.fiscal_year_end.split("-"))
    end = pd.Timestamp(year=year, month=month, day=day)
    if end < ledger["EcritureDate"].min():
        end = pd.Timestamp(year=year + 1, month=month, day=day)
    start = end - pd.Timedelta(days=args.days - 1)
    mask = ledger["EcritureDate"].between(start, end)
    if args.manual_only:
        mask &= ledger["JournalCode"].eq("OD")
    hits = ledger.loc[mask, "EcritureNum"].unique().tolist()
    scope = "manual entries" if args.manual_only else "entries"
    return ToolResult(tool="period_end_entries", flagged_count=len(hits), flagged_entries=_cap(hits, max_items),
                      summary=f"{len(hits)} {scope} dated in the last {args.days} days before {end:%Y-%m-%d}.")


def suspense_accounts(ledger: pd.DataFrame, args: SuspenseArgs, max_items: int) -> ToolResult:
    accounts = ledger["CompteNum"].astype(str)
    suspense = accounts.str.startswith(tuple(args.prefixes)) if args.prefixes else pd.Series(False, index=ledger.index)
    rare = pd.Series(False, index=ledger.index)
    if args.max_uses > 0:
        rare = accounts.map(accounts.value_counts()) <= args.max_uses
    hits = ledger.loc[suspense | rare, "EcritureNum"].unique().tolist()
    return ToolResult(tool="suspense_accounts", flagged_count=len(hits), flagged_entries=_cap(hits, max_items),
                      summary=f"{len(hits)} entries posted to suspense or rarely used accounts.",
                      details={"suspense_lines": int(suspense.sum()), "rare_account_lines": int(rare.sum())})


def vague_entries(ledger: pd.DataFrame, args: VagueArgs, max_items: int) -> ToolResult:
    no_piece = ledger["PieceRef"].fillna("").astype(str).str.strip().eq("")
    labels = ledger["EcritureLib"].fillna("").astype(str).str.lower()
    vague = pd.Series(False, index=ledger.index)
    for word in args.keywords:  # whole words: "divers" must not match the journal label "Opérations diverses"
        vague |= labels.str.contains(rf"(?<!\w){re.escape(word.lower())}(?!\w)", regex=True)
    hits = ledger.loc[no_piece | vague, "EcritureNum"].unique().tolist()
    return ToolResult(tool="vague_entries", flagged_count=len(hits), flagged_entries=_cap(hits, max_items),
                      summary=f"{len(hits)} entries without a supporting document reference or with a vague label.")


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
        Tool(name="numbering_gaps", description="Find gaps in the EcritureNum sequence (possible deleted entries).",
             args_model=NoArgs, func=numbering_gaps),
        Tool(name="period_end_entries",
             description="Find entries booked in the last days before the fiscal year end (cut-off risk).",
             args_model=PeriodEndArgs, func=period_end_entries),
        Tool(name="suspense_accounts",
             description="Find entries posted to suspense accounts (471/472/473) or to rarely used accounts.",
             args_model=SuspenseArgs, func=suspense_accounts),
        Tool(name="vague_entries",
             description="Find entries with no supporting document reference or a vague label (e.g. 'divers').",
             args_model=VagueArgs, func=vague_entries),
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
