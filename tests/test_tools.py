import pytest

from audit_copilot.tools import ALL_TOOLS, get_tools

ANOMALY_TOOLS = ["weekend_entries", "round_amounts", "duplicate_entries",
                 "unbalanced_entries", "late_validation"]


@pytest.mark.parametrize("name", ANOMALY_TOOLS)
def test_tools_find_exactly_the_injected_anomalies(ledger_and_truth, name):
    ledger, truth = ledger_and_truth
    result = ALL_TOOLS[name].run(ledger, {}, max_items=1000)
    assert set(result.flagged_entries) == set(truth[name])
    assert result.flagged_count == len(truth[name])


def test_results_are_capped(ledger_and_truth):
    ledger, _ = ledger_and_truth
    result = ALL_TOOLS["duplicate_entries"].run(ledger, {}, max_items=2)
    assert len(result.flagged_entries) == 2
    assert result.flagged_count > 2


def test_invalid_arguments_raise(ledger_and_truth):
    ledger, _ = ledger_and_truth
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ALL_TOOLS["round_amounts"].run(ledger, {"multiple": -5}, max_items=10)


def test_benford_returns_verdict(ledger_and_truth):
    ledger, _ = ledger_and_truth
    result = ALL_TOOLS["benford_test"].run(ledger, {}, max_items=10)
    assert result.details["verdict"] in {"close conformity", "acceptable conformity",
                                         "marginally acceptable conformity", "nonconformity"}


def test_feature_flags():
    assert list(get_tools(["weekend_entries"])) == ["weekend_entries"]
    with pytest.raises(ValueError):
        get_tools(["does_not_exist"])


def test_openai_schema_shape():
    schema = ALL_TOOLS["round_amounts"].openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "round_amounts"
    assert "min_amount" in schema["function"]["parameters"]["properties"]
