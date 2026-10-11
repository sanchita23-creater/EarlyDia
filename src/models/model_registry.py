"""
EarlyDia - Model Registry
===========================
Configuration-driven mechanism to select the approved production artifact.
Each registered model has an explicit version, sex scope, algorithm,
schema version, artifact location, and approval status.

No unapproved experimental model can become the production default.
The registry is the single source of truth for which artifact the API loads.
"""
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REGISTRY_PATH = _PROJECT_ROOT / "configs" / "model_registry.json"


@dataclass
class ModelEntry:
    """One registered model artifact."""
    model_id: str               # e.g. "v2_women_state_native_B"
    sex: str                    # "women" or "men"
    version: str                # "v1", "v2", "v2-xgb" etc.
    algorithm: str              # e.g. "HistGradientBoosting", "LogisticRegression", "XGBClassifier"
    feature_flags: str          # e.g. "state_native", "state"
    schema_version: str         # e.g. "v2"
    artifact_path: str          # relative to project root
    threshold: float
    calibration_method: str
    status: str                 # "approved", "experimental", "deprecated"
    notes: Optional[str] = None

    def absolute_path(self) -> Path:
        return _PROJECT_ROOT / self.artifact_path


def load_registry() -> dict:
    """Load the model registry from disk. Returns dict keyed by model_id."""
    if not _REGISTRY_PATH.exists():
        return {}
    data = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    return {k: ModelEntry(**v) for k, v in data.items()}


def save_registry(registry: dict) -> None:
    """Save the model registry to disk."""
    data = {k: asdict(v) for k, v in registry.items()}
    _REGISTRY_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_production_model(sex: str, registry: dict = None) -> ModelEntry:
    """Get the approved production model for the given sex.

    Raises ValueError if no approved model exists for this sex.
    """
    if registry is None:
        registry = load_registry()
    approved = [v for v in registry.values()
                if v.sex == sex and v.status == "approved"]
    if not approved:
        raise ValueError(f"No approved production model for sex='{sex}'")
    if len(approved) > 1:
        raise ValueError(f"Multiple approved models for sex='{sex}': "
                         f"{[m.model_id for m in approved]}. "
                         "Resolve ambiguity before proceeding.")
    return approved[0]


def promote_model(model_id: str, registry: dict = None) -> dict:
    """Promote an experimental model to approved. Demotes any previously
    approved model for the same sex to 'deprecated'.

    Returns the updated registry (also saved to disk).
    """
    if registry is None:
        registry = load_registry()
    if model_id not in registry:
        raise ValueError(f"Model '{model_id}' not found in registry")
    entry = registry[model_id]
    # Demote any existing approved model for the same sex
    for k, v in registry.items():
        if v.sex == entry.sex and v.status == "approved":
            registry[k] = ModelEntry(**{**asdict(v), "status": "deprecated"})
    # Promote
    registry[model_id] = ModelEntry(**{**asdict(entry), "status": "approved"})
    save_registry(registry)
    return registry


def rollback_model(sex: str, registry: dict = None) -> dict:
    """Roll back to the most recent deprecated model for the given sex.

    Returns the updated registry (also saved to disk).
    """
    if registry is None:
        registry = load_registry()
    # Find the current approved and most recent deprecated
    current = None
    deprecated = []
    for k, v in registry.items():
        if v.sex == sex and v.status == "approved":
            current = k
        elif v.sex == sex and v.status == "deprecated":
            deprecated.append(k)
    if not deprecated:
        raise ValueError(f"No deprecated model to roll back to for sex='{sex}'")
    # Demote current, promote most recent deprecated
    rollback_id = deprecated[-1]
    if current:
        registry[current] = ModelEntry(**{**asdict(registry[current]), "status": "deprecated"})
    registry[rollback_id] = ModelEntry(**{**asdict(registry[rollback_id]), "status": "approved"})
    save_registry(registry)
    return registry
