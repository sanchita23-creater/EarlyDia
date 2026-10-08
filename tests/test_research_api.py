"""Research endpoints serve already-generated artifacts (needs fastapi; skipped otherwise)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from backend.api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("path", ["summary", "calibration", "subgroups", "decision-curve", "feature-importance", "target-definition"])
def test_research_endpoints_ok(client, path):
    r = client.get(f"/api/research/{path}")
    assert r.status_code == 200, r.text


def test_summary_states_locked_thresholds(client):
    s = client.get("/api/research/summary").json()
    assert s["production_model"]["women"]["threshold"] == 0.026 and s["production_model"]["men"]["threshold"] == 0.027
