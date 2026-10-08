import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi.testclient import TestClient
from backend.api.main import app

client = TestClient(app)

print("--- Testing /api/health ---")
h = client.get("/api/health").json()
print("Health response:", h)
assert h["status"] == "ok"

print("\n--- Testing Women v2 Prediction Flow ---")
women_req = {
    "sex": "women", "age": 32, "education": 2, "residence": 1,
    "tobacco_use": False, "drinks_alcohol": False,
    "diet": {k: 2 for k in "abcdefghi"},
    "height_cm": 158.0, "weight_kg": 64.0,
    "bp_systolic": 130.0, "bp_diastolic": 85.0
}
res_w = client.post("/api/predict", json=women_req).json()
print("Women output keys:", list(res_w.keys()))
print(f"Women Probability: {res_w['percent']}%, Cutoff: {res_w['cutoff_percent']}%, Flagged: {res_w['flagged']}")
print("Technical details:", res_w["technical"])
assert res_w["cutoff"] == 0.024, f"Expected 0.024, got {res_w['cutoff']}"
assert res_w["technical"]["version"] == "v2"
assert "state_native_B" in res_w["technical"]["model"]
assert res_w["technical"]["test_metrics"]["roc_auc"] == 0.710

print("\n--- Testing Men v2 Prediction Flow ---")
men_req = {
    "sex": "men", "age": 35, "education": 3, "residence": 1,
    "tobacco_use": True, "tobacco_products": ["tob_mv463a"],
    "drinks_alcohol": True, "alcohol_frequency": 2, "alcohol_years": 8.0,
    "alcohol_types": ["alc_sm621c"],
    "diet": {k: 1 for k in "abcdefghi"},
    "bp_systolic": 138.0, "bp_diastolic": 90.0
}
res_m = client.post("/api/predict", json=men_req).json()
print(f"Men Probability: {res_m['percent']}%, Cutoff: {res_m['cutoff_percent']}%, Flagged: {res_m['flagged']}")
print("Technical details:", res_m["technical"])
assert res_m["cutoff"] == 0.029, f"Expected 0.029, got {res_m['cutoff']}"
assert res_m["technical"]["version"] == "v2"
assert "lr_state" in res_m["technical"]["model"]
assert res_m["technical"]["test_metrics"]["roc_auc"] == 0.671

print("\n--- Testing Static SPA serving ---")
spa_res = client.get("/")
print("GET / status code:", spa_res.status_code)
assert spa_res.status_code == 200
assert "EarlyDia" in spa_res.text

print("\n[ALL INTEGRATION CHECKS PASSED SUCCESSFULLY]")
