"""Generate a synthetic FEC ledger with known, injected anomalies.

No real company data is used. The generator returns the ledger and a ground
truth mapping {anomaly type: [EcritureNum, ...]} so tools and the agent can be
evaluated with precision and recall.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .fec import FEC_COLUMNS, write_fec

ACCOUNTS = {
    "VE": [("411000", "Clients"), ("706000", "Prestations de services"), ("445710", "TVA collectée")],
    "AC": [("606000", "Achats non stockés"), ("401000", "Fournisseurs"), ("445660", "TVA déductible")],
    "BQ": [("512000", "Banque"), ("411000", "Clients")],
    "OD": [("641000", "Rémunérations du personnel"), ("421000", "Personnel - rémunérations dues")],
}
JOURNAL_LABELS = {"VE": "Ventes", "AC": "Achats", "BQ": "Banque", "OD": "Opérations diverses"}

ANOMALY_TYPES = ["weekend_entries", "round_amounts", "duplicate_entries", "unbalanced_entries", "late_validation"]


def _weekday(rng: np.random.Generator, year: int) -> date:
    while True:
        d = date(year, 1, 1) + timedelta(days=int(rng.integers(0, 365)))
        if d.weekday() < 5:
            return d


def _weekend(rng: np.random.Generator, year: int) -> date:
    while True:
        d = date(year, 1, 1) + timedelta(days=int(rng.integers(0, 365)))
        if d.weekday() >= 5:
            return d


def _amount(rng: np.random.Generator) -> float:
    # Non-round amounts with cents, log-normal like real invoices.
    value = round(float(rng.lognormal(mean=7.0, sigma=1.0)), 2)
    if value == int(value):
        value += 0.37
    return value


def _entry(num: str, journal: str, day: date, amount: float, piece: str, label: str,
           valid_delay: int = 2, credit_amount: float | None = None) -> list[dict]:
    """Build one balanced (or deliberately unbalanced) two-line entry.

    The first account of the journal is debited, the second credited
    (e.g. purchases: debit 606 expense, credit 401 supplier).
    """
    (debit_acc, debit_lib), (credit_acc, credit_lib) = ACCOUNTS[journal][0], ACCOUNTS[journal][1]
    credit_amount = amount if credit_amount is None else credit_amount
    common = {
        "JournalCode": journal,
        "JournalLib": JOURNAL_LABELS[journal],
        "EcritureNum": num,
        "EcritureDate": pd.Timestamp(day),
        "CompAuxNum": "",
        "CompAuxLib": "",
        "PieceRef": piece,
        "PieceDate": pd.Timestamp(day),
        "EcritureLib": label,
        "EcritureLet": "",
        "DateLet": pd.NaT,
        "ValidDate": pd.Timestamp(day + timedelta(days=valid_delay)),
        "Montantdevise": 0.0,
        "Idevise": "EUR",
    }
    return [
        {**common, "CompteNum": debit_acc, "CompteLib": debit_lib, "Debit": amount, "Credit": 0.0},
        {**common, "CompteNum": credit_acc, "CompteLib": credit_lib, "Debit": 0.0, "Credit": credit_amount},
    ]


def generate_fec(n_entries: int = 1000, year: int = 2025, seed: int = 42,
                 n_per_anomaly: int = 6) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Return (ledger, ground_truth)."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    truth: dict[str, list[str]] = {k: [] for k in ANOMALY_TYPES}
    counter = 0

    def next_num() -> str:
        nonlocal counter
        counter += 1
        return f"{counter:06d}"

    journals = list(ACCOUNTS)
    for _ in range(n_entries):
        j = journals[int(rng.integers(0, len(journals)))]
        num = next_num()
        rows += _entry(num, j, _weekday(rng, year), _amount(rng), f"P{num}", f"{JOURNAL_LABELS[j]} {num}")

    for _ in range(n_per_anomaly):
        num = next_num()
        rows += _entry(num, "OD", _weekend(rng, year), _amount(rng), f"P{num}", "Écriture manuelle")
        truth["weekend_entries"].append(num)

        num = next_num()
        round_value = float(int(rng.integers(2, 20)) * 5000)
        rows += _entry(num, "OD", _weekday(rng, year), round_value, f"P{num}", "Régularisation")
        truth["round_amounts"].append(num)

        num = next_num()
        rows += _entry(num, "AC", _weekday(rng, year), _amount(rng), f"P{num}", "Facture fournisseur",
                       valid_delay=int(rng.integers(60, 180)))
        truth["late_validation"].append(num)

        amount, day = _amount(rng), _weekday(rng, year)
        first, second = next_num(), next_num()
        piece = f"F{first}"
        rows += _entry(first, "AC", day, amount, piece, "Facture fournisseur")
        rows += _entry(second, "AC", day, amount, piece, "Facture fournisseur")
        truth["duplicate_entries"] += [first, second]

    for _ in range(max(1, n_per_anomaly // 2)):
        num = next_num()
        amount = _amount(rng)
        rows += _entry(num, "VE", _weekday(rng, year), amount, f"P{num}", "Vente",
                       credit_amount=round(amount - float(rng.integers(10, 500)), 2))
        truth["unbalanced_entries"].append(num)

    ledger = pd.DataFrame(rows)[FEC_COLUMNS]
    ledger = ledger.sort_values(["EcritureDate", "EcritureNum"]).reset_index(drop=True)
    return ledger, truth


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic FEC with injected anomalies")
    parser.add_argument("--out", default="data/sample_fec.txt")
    parser.add_argument("--entries", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ledger, truth = generate_fec(n_entries=args.entries, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_fec(ledger, out)
    out.with_suffix(".truth.json").write_text(json.dumps(truth, indent=2))
    print(f"Wrote {len(ledger)} lines to {out} and ground truth to {out.with_suffix('.truth.json')}")


if __name__ == "__main__":
    main()
