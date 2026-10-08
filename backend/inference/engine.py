"""Thin adapter between the API and the production inference modules.

Supports both v2 (default for EarlyDia v2) and locked v1 pipelines.
"""
import sys
from functools import lru_cache

from backend.configs.settings import PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict_module import EarlyDiaPredictor  # noqa: E402
from src.inference.predict_module_v2 import EarlyDiaPredictorV2  # noqa: E402

SEXES = ("women", "men")


@lru_cache(maxsize=None)
def get_predictor(sex: str, version: str = "v2"):
    if version == "v2":
        return EarlyDiaPredictorV2(sex)
    return EarlyDiaPredictor(sex)


def load_all(version: str = "v2") -> dict:
    """Load production pipelines (called at API start-up so problems surface immediately)."""
    return {s: get_predictor(s, version=version) for s in SEXES}
