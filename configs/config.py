"""
EarlyDia - Central configuration.
All paths are resolved relative to the project root (this file's parent directory),
so the project runs identically regardless of where it is cloned/copied on disk.
No machine-specific absolute paths (absolute paths of any kind) appear anywhere else in the
codebase - every other module imports paths from here.
"""
from pathlib import Path

# Scikit-learn HistGradientBoosting loss serialization compatibility: the pickled loss class
# is stored under the bare module name '_loss' (see configs/sklearn_compat.py for the full
# root-cause explanation). Registering the alias here keeps utility scripts that use plain
# pickle.load working; the production predictor uses sklearn_compat.load_artifact instead.
from configs.sklearn_compat import register_loss_alias
register_loss_alias()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
EXPLAINABILITY_REPORTS_DIR = REPORTS_DIR / "explainability"
FINAL_EVAL_REPORTS_DIR = REPORTS_DIR / "final_evaluation"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Production pipeline artifact paths (per sex) - these are the single source of truth
# for the trained base model, calibrator, feature list, and locked threshold.
PRODUCTION_PIPELINE_PATHS = {
    "women": MODELS_DIR / "women_production_pipeline.pkl",
    "men": MODELS_DIR / "men_production_pipeline.pkl",
}

# Reproducibility: master random seed used throughout the project (data split, CV,
# model fitting, calibration search). Recorded here as the single documented value;
# see configs/reproducibility_manifest.json for the full audit trail.
RANDOM_SEED = 42

# FINAL, LOCKED screening thresholds (women 0.026, men 0.027).
# Rule: highest threshold on the calibrated grouped-OOF predictions (0.001 grid) whose
# sensitivity is still >= 0.75. A documented screening-threshold choice, not a clinically
# optimal value. Pipeline scripts (src/models/final_refit.py) read this constant when
# building artifacts; inference (predict_module.py) reads the threshold stored in the
# artifact itself, and tests assert the two always agree.
DOCUMENTED_THRESHOLDS = {"women": 0.026, "men": 0.027}

# V2 candidate models and thresholds (separate from locked v1 artifacts)
MODELS_V2_DIR = PROJECT_ROOT / "models_v2"
PRODUCTION_PIPELINE_V2_PATHS = {
    "women": MODELS_V2_DIR / "women_v2_pipeline.pkl",
    "men": MODELS_V2_DIR / "men_v2_pipeline.pkl",
}
# Updated for XGBoost v2 candidate promotion
DOCUMENTED_V2_THRESHOLDS = {"women": 0.026, "men": 0.028}
