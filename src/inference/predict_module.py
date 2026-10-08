"""
EarlyDia - Production Prediction Module
========================================
Loads the ALREADY-TRAINED, ALREADY-CALIBRATED production pipelines saved under models/
and applies them to a single person's input. Contains NO training, fitting, or tuning
logic of any kind, and NEVER imports or references any final-test data file.

Thresholds are NOT hardcoded here - they are read from the saved production artifact
(models/women_production_pipeline.pkl / models/men_production_pipeline.pkl), which is
the single documented source of truth for the locked threshold values.

All paths come from configs/config.py - no machine-specific absolute paths appear here,
so this module works identically regardless of where the project is cloned.

This is a RESEARCH/PROTOTYPE SCREENING output, not a diagnosis. See README /
limitations documentation for the full responsible-use statement.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config import PRODUCTION_PIPELINE_PATHS  # noqa: E402
from configs.sklearn_compat import load_artifact  # noqa: E402
from src.calibration.calibration_utils import sigmoid_calibrate_predict  # noqa: E402

# Plausibility ranges used ONLY for input validation at prediction time.
# These mirror (do not redefine) the cleaning bounds used during training-data preparation.
VALID_RANGES = {
    "age": (18, 40),
    "avg_systolic_bp": (70, 270),
    "avg_diastolic_bp": (40, 150),
    "bmi": (10, 70),  # slightly wider than observed training range to avoid over-rejecting
    "alc_duration_years": (0, 90),
}
BINARY_FIELDS_COMMON = ["tob_v463a", "tob_v463b", "tob_v463c", "tob_v463d", "tob_v463f",
                         "tob_v463g", "tob_v463x", "tob_v463z", "alc_drinks"]
WOMEN_ONLY_BINARY = ["tob_v463j", "tob_v463k", "tob_v463l"]
WOMEN_ALC_TYPE = ["alc_s722a", "alc_s722b", "alc_s722c", "alc_s722d", "alc_s722e", "alc_s722x"]
MEN_ALC_TYPE = ["alc_sm621a", "alc_sm621b", "alc_sm621c", "alc_sm621d", "alc_sm621e", "alc_sm621x"]
DIET_CATEGORIES = {0, 1, 2, 3}
EDUCATION_CATEGORIES = {0, 1, 2, 3}
RESIDENCE_CATEGORIES = {1, 2}
ALC_FREQ_CATEGORIES = {0, 1, 2, 3}
TOB_FREQ_CATEGORIES = {0, 1, 2}  # tob_v463aa (women only)


@dataclass
class PredictionResult:
    sex: str
    model: str
    calibration_method: str
    calibrated_probability: float
    predicted_class: int
    threshold_used: float
    validation_warnings: list = field(default_factory=list)


class EarlyDiaPredictor:
    """Loads a saved production pipeline for one sex and serves predictions.
    Contains no training/fitting logic - purely an inference wrapper."""

    def __init__(self, sex: str):
        if sex not in PRODUCTION_PIPELINE_PATHS:
            raise ValueError(f"sex must be 'women' or 'men', got {sex!r}")
        self.sex = sex
        artifact_path = PRODUCTION_PIPELINE_PATHS[sex]
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Production pipeline artifact not found at {artifact_path}. "
                f"Ensure models/{sex}_production_pipeline.pkl exists.")
        artifact = load_artifact(artifact_path)   # read-only; never re-saved
        self.base_pipeline = artifact["base_pipeline"]      # already fit - never refit here
        self.calibrator = artifact["calibrator"]             # already fit - never refit here
        self.calibration_method = artifact["calibration_method"]
        self.feature_cols = artifact["feature_cols"]
        self.threshold = artifact["selected_threshold"]      # single source of truth
        self.model_name = "HistGradientBoosting"

    def _binary_fields(self):
        fields = list(BINARY_FIELDS_COMMON)
        if self.sex == "women":
            fields += WOMEN_ONLY_BINARY
        return fields

    def _alc_type_fields(self):
        return WOMEN_ALC_TYPE if self.sex == "women" else MEN_ALC_TYPE

    def _validate_and_prepare(self, raw_input: dict):
        """Returns (row_dict_ready_for_pipeline, warnings_list). Raises ValueError on
        hard invalid input (wrong type / impossible category). Missing fields are allowed
        through as None -> NaN, consistent with training-time imputation handling."""
        warnings = []
        row = {}

        for col in self.feature_cols:
            if col not in raw_input or raw_input[col] is None:
                row[col] = float("nan")
                warnings.append(f"'{col}' missing - imputed by the trained pipeline "
                                 f"(same strategy as training)")
                continue
            val = raw_input[col]

            if col in VALID_RANGES:
                lo, hi = VALID_RANGES[col]
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    raise ValueError(f"'{col}' must be numeric, got {val!r}")
                if not (lo <= val <= hi):
                    raise ValueError(f"'{col}'={val} is outside the plausible range "
                                      f"[{lo}, {hi}] used for this population (age 18-40)")
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
                    raise ValueError(f"'{col}' must be one of {DIET_CATEGORIES} "
                                      f"(0=Never,1=Daily,2=Weekly,3=Occasionally), got {val!r}")
                row[col] = int(val)
                continue

            if col == "education":
                if val not in EDUCATION_CATEGORIES:
                    raise ValueError(f"'education' must be one of {EDUCATION_CATEGORIES}, got {val!r}")
                row[col] = int(val)
                continue

            if col == "residence":
                if val not in RESIDENCE_CATEGORIES:
                    raise ValueError(f"'residence' must be one of {RESIDENCE_CATEGORIES} "
                                      f"(1=Urban,2=Rural), got {val!r}")
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

            # fallback: pass through numeric fields not explicitly covered above
            row[col] = val

        return row, warnings

    def predict(self, person: dict) -> PredictionResult:
        import pandas as pd

        row, warnings = self._validate_and_prepare(person)
        X = pd.DataFrame([row], columns=self.feature_cols).astype("float64")
        # dtype note: training data columns were float64 throughout (NaNs elsewhere in the
        # full dataset forced float dtype even for nominally-integer categorical/binary
        # columns); the fitted preprocessor's imputers/encoders expect float64 input,
        # so we cast uniformly here to match - this is inference-time alignment only,
        # not a modelling decision.

        p_raw = self.base_pipeline.predict_proba(X)[:, 1][0]  # no retraining - inference only
        if self.calibration_method == "isotonic":
            p_calibrated = float(self.calibrator.predict([p_raw])[0])
        else:
            p_calibrated = float(sigmoid_calibrate_predict(self.calibrator, [p_raw])[0])

        predicted_class = int(p_calibrated >= self.threshold)

        return PredictionResult(
            sex=self.sex,
            model=self.model_name,
            calibration_method=self.calibration_method,
            calibrated_probability=p_calibrated,
            predicted_class=predicted_class,
            threshold_used=self.threshold,
            validation_warnings=warnings,
        )
