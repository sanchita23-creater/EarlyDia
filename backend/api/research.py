"""Read-only endpoints that serve ALREADY-GENERATED research artifacts (reports/research/).
No research computation happens here and nothing imports src.research (production isolation)."""
import csv
import json

from fastapi import APIRouter, HTTPException

from backend.configs.settings import PROJECT_ROOT

router = APIRouter(prefix="/api/research", tags=["research"])
RESEARCH = PROJECT_ROOT / "reports" / "research"


def _csv(name):
    p = RESEARCH / name
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"{name} has not been generated. Run: python -m src.research.run_all")
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _json(name):
    p = RESEARCH / name
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"{name} has not been generated. Run: python -m src.research.run_all")
    return json.loads(p.read_text(encoding="utf-8"))


@router.get("/summary")
def summary():
    return _json("research_summary.json")


@router.get("/calibration")
def calibration():
    return {"rows": _csv("calibration_analysis.csv")}


@router.get("/subgroups")
def subgroups():
    return {"rows": _csv("subgroup_performance.csv")}


@router.get("/decision-curve")
def decision_curve():
    return {"rows": _csv("decision_curve.csv")}


@router.get("/feature-importance")
def feature_importance():
    return {"rows": _csv("feature_importance.csv")}


@router.get("/target-definition")
def target_definition():
    return _json("target_definition_comparison.json")
