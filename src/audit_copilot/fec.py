"""Loading and validating FEC files (Fichier des Écritures Comptables).

A FEC is the standard ledger export French companies must provide to auditors
and the tax administration. It has 18 mandatory columns, dates as YYYYMMDD and
amounts with a decimal comma. Files are tab- or pipe-separated.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FEC_COLUMNS: list[str] = [
    "JournalCode",
    "JournalLib",
    "EcritureNum",
    "EcritureDate",
    "CompteNum",
    "CompteLib",
    "CompAuxNum",
    "CompAuxLib",
    "PieceRef",
    "PieceDate",
    "EcritureLib",
    "Debit",
    "Credit",
    "EcritureLet",
    "DateLet",
    "ValidDate",
    "Montantdevise",
    "Idevise",
]

DATE_COLUMNS = ["EcritureDate", "PieceDate", "DateLet", "ValidDate"]
AMOUNT_COLUMNS = ["Debit", "Credit", "Montantdevise"]


class FECFormatError(ValueError):
    """Raised when a file does not follow the FEC layout."""


def _detect_separator(first_line: str) -> str:
    if "\t" in first_line:
        return "\t"
    if "|" in first_line:
        return "|"
    raise FECFormatError("FEC files must be tab- or pipe-separated")


def _to_amount(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("").astype(str).str.replace(" ", "", regex=False)
    cleaned = cleaned.str.replace(",", ".", regex=False).replace("", "0")
    return pd.to_numeric(cleaned, errors="raise")


def parse_fec(df: pd.DataFrame) -> pd.DataFrame:
    """Validate columns and convert dates and amounts to proper types."""
    missing = [c for c in FEC_COLUMNS if c not in df.columns]
    if missing:
        raise FECFormatError(f"Missing FEC columns: {missing}")

    out = df[FEC_COLUMNS].copy()
    for col in DATE_COLUMNS:
        out[col] = pd.to_datetime(out[col].astype(str).str.strip(), format="%Y%m%d", errors="coerce")
    for col in AMOUNT_COLUMNS:
        out[col] = _to_amount(out[col])
    for col in ["JournalCode", "EcritureNum", "CompteNum", "PieceRef"]:
        out[col] = out[col].astype(str).str.strip()

    if out["EcritureDate"].isna().any():
        bad = int(out["EcritureDate"].isna().sum())
        raise FECFormatError(f"{bad} rows have an invalid EcritureDate (expected YYYYMMDD)")
    return out


def load_fec(path: str | Path) -> pd.DataFrame:
    """Read a FEC file from disk and return a typed DataFrame."""
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        sep = _detect_separator(fh.readline())
    raw = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    return parse_fec(raw)


def write_fec(df: pd.DataFrame, path: str | Path) -> None:
    """Write a typed DataFrame back to the FEC text format (tab-separated)."""
    out = df[FEC_COLUMNS].copy()
    for col in DATE_COLUMNS:
        out[col] = out[col].dt.strftime("%Y%m%d").fillna("")
    for col in AMOUNT_COLUMNS:
        out[col] = out[col].map(lambda x: f"{x:.2f}".replace(".", ","))
    out.to_csv(path, sep="\t", index=False)
