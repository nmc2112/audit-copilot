import pandas as pd
import pytest

from audit_copilot.fec import FEC_COLUMNS, FECFormatError, load_fec, parse_fec, write_fec


def test_roundtrip(tmp_path, ledger_and_truth):
    ledger, _ = ledger_and_truth
    path = tmp_path / "fec.txt"
    write_fec(ledger, path)
    loaded = load_fec(path)
    assert list(loaded.columns) == FEC_COLUMNS
    assert len(loaded) == len(ledger)
    assert loaded["Debit"].sum() == pytest.approx(ledger["Debit"].sum())


def test_decimal_comma_and_dates():
    row = {c: "" for c in FEC_COLUMNS}
    row.update(EcritureNum="1", EcritureDate="20250315", Debit="1 234,56", Credit="0")
    parsed = parse_fec(pd.DataFrame([row]))
    assert parsed.loc[0, "Debit"] == pytest.approx(1234.56)
    assert parsed.loc[0, "EcritureDate"] == pd.Timestamp("2025-03-15")


def test_missing_columns_rejected():
    with pytest.raises(FECFormatError):
        parse_fec(pd.DataFrame([{"EcritureNum": "1"}]))


def test_invalid_date_rejected():
    row = {c: "" for c in FEC_COLUMNS}
    row.update(EcritureNum="1", EcritureDate="2025-03-15", Debit="1", Credit="1")
    with pytest.raises(FECFormatError):
        parse_fec(pd.DataFrame([row]))
