"""
EarlyDia v2 - Production Prediction Module
===========================================
Loads the trained and calibrated v2 production pipelines saved under models_v2/
and applies them to a single person's input. Contains NO training or fitting logic.

Thresholds are read directly from the saved v2 artifacts:
  - models_v2/women_v2_pipeline.pkl (candidate lgbm_D, threshold 0.212)
  - models_v2/men_v2_pipeline.pkl (candidate cb_B, threshold 0.300)

This is a RESEARCH/PROTOTYPE SCREENING output on cross-sectional data,
not a diagnosis or future diabetes prediction.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config import PRODUCTION_PIPELINE_V2_PATHS
from configs.sklearn_compat import load_artifact
from src.calibration.calibration_utils import sigmoid_calibrate_predict
from src.inference.predict_module import (
    VALID_RANGES,
    BINARY_FIELDS_COMMON,
    WOMEN_ONLY_BINARY,
    WOMEN_ALC_TYPE,
    MEN_ALC_TYPE,
    DIET_CATEGORIES,
    EDUCATION_CATEGORIES,
    RESIDENCE_CATEGORIES,
    ALC_FREQ_CATEGORIES,
    TOB_FREQ_CATEGORIES,
)


@dataclass
class PredictionResultV2:
    sex: str
    model: str
    calibration_method: str
    calibrated_probability: float
    predicted_class: int
    threshold_used: float
    validation_warnings: list = field(default_factory=list)
    version: str = "v2"
    candidate: str = ""


class EarlyDiaPredictorV2:
    """Loads a saved v2 pipeline for one sex and serves predictions."""

    def __init__(self, sex: str):
        if sex not in PRODUCTION_PIPELINE_V2_PATHS:
            raise ValueError(f"sex must be 'women' or 'men', got {sex!r}")
        self.sex = sex
        artifact_path = PRODUCTION_PIPELINE_V2_PATHS[sex]
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"V2 pipeline artifact not found at {artifact_path}. "
                f"Ensure models_v2/{sex}_v2_pipeline.pkl exists."
            )
        artifact = load_artifact(artifact_path)
        self.base_pipeline = artifact["base_pipeline"]
        self.calibrator = artifact["calibrator"]
        self.calibration_method = artifact["calibration_method"]
        self.feature_cols = artifact["feature_cols"]
        self.threshold = artifact["selected_threshold"]
        self.candidate = artifact.get("candidate", "lgbm_D" if sex == "women" else "cb_B")
        self.model_name = (
            "LightGBM (v2 - lgbm_D)"
            if sex == "women"
            else "CatBoost (v2 - cb_B)"
        )
        self.version = "v2"

    def _binary_fields(self):
        fields = list(BINARY_FIELDS_COMMON)
        if self.sex == "women":
            fields += WOMEN_ONLY_BINARY
        return fields

    def _alc_type_fields(self):
        return WOMEN_ALC_TYPE if self.sex == "women" else MEN_ALC_TYPE

    def _validate_and_prepare(self, raw_input: dict):
        warnings = []
        row = {}

        for col in self.feature_cols:
            if col not in raw_input or raw_input[col] is None:
                row[col] = float("nan")
                if col != "state":
                    warnings.append(
                        f"'{col}' missing - imputed by the trained pipeline (same strategy as training)"
                    )
                continue
            val = raw_input[col]

            if col == "state":
                try:
                    val = float(val)
                    if not (1 <= val <= 37):
                        raise ValueError
                except (TypeError, ValueError):
                    raise ValueError(f"'state' must be a valid state code (1-37), got {val!r}")
                row[col] = val
                continue

            if col in VALID_RANGES:
                lo, hi = VALID_RANGES[col]
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    raise ValueError(f"'{col}' must be numeric, got {val!r}")
                if not (lo <= val <= hi):
                    raise ValueError(
                        f"'{col}'={val} is outside the plausible range [{lo}, {hi}] used for this population (age 18-40)"
                    )
                row[col] = val
                continue

            if col in self._binary_fields() or col in self._alc_type_fields():
                if val not in (0, 1, 0.0, 1.0):
                    raise ValueError(f"'{col}' must be 0 or 1, got {val!r}")
                row[col] = int(val)
                continue

            if col == "n_valid_systolic_readings" or col == "n_valid_diastolic_readings":
                if val not in (0, 1, 2, 3):
                    raise ValueError(f"'{col}' must be 0-3, got {val!r}")
                row[col] = int(val)
                continue

            if col.startswith("diet_"):
                if val not in DIET_CATEGORIES:
                    raise ValueError(
                        f"'{col}' must be one of {DIET_CATEGORIES} (0=Never,1=Daily,2=Weekly,3=Occasionally), got {val!r}"
                    )
                row[col] = int(val)
                continue

            if col == "education":
                if val not in EDUCATION_CATEGORIES:
                    raise ValueError(f"'education' must be one of {EDUCATION_CATEGORIES}, got {val!r}")
                row[col] = int(val)
                continue

            if col == "residence":
                if val not in RESIDENCE_CATEGORIES:
                    raise ValueError(f"'residence' must be one of {RESIDENCE_CATEGORIES} (1=Urban,2=Rural), got {val!r}")
                row[col] = int(val)
                continue

            if col == "alc_frequency":
                if val not in ALC_FREQ_CATEGORIES:
                    raise ValueError(f"'alc_frequency' must be one of {ALC_FREQ_CATEGORIES}, got {val!r}")
                row[col] = int(val)
                continue

            if col == "tob_v463aa":
                if val not in TOB_FREQ_CATEGORIES:
                    raise ValueError(f"'tob_v463aa' must be one of {TOB_FREQ_CATEGORIES}, got {val!r}")
                row[col] = int(val)
                continue

            row[col] = val

        return row, warnings

    def predict(self, person: dict) -> PredictionResultV2:
        import pandas as pd

        row, warnings = self._validate_and_prepare(person)
        X = pd.DataFrame([row], columns=self.feature_cols).astype("float64")

        p_raw = self.base_pipeline.predict_proba(X)[:, 1][0]
        if self.calibration_method == "isotonic":
            p_calibrated = float(self.calibrator.predict([p_raw])[0])
        else:
            p_calibrated = float(sigmoid_calibrate_predict(self.calibrator, [p_raw])[0])

        predicted_class = int(p_calibrated >= self.threshold)

        return PredictionResultV2(
            sex=self.sex,
            model=self.model_name,
            calibration_method=self.calibration_method,
            calibrated_probability=p_calibrated,
            predicted_class=predicted_class,
            threshold_used=self.threshold,
            validation_warnings=warnings,
            version="v2",
            candidate=self.candidate,
        )
