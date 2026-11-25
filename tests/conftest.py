import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_copilot.config import load_settings  # noqa: E402
from audit_copilot.synth import generate_fec  # noqa: E402


@pytest.fixture(scope="session")
def ledger_and_truth():
    return generate_fec(n_entries=300, seed=1)


@pytest.fixture
def settings():
    return load_settings()
