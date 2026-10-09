"""
Tests for the XGBoost pipeline integration, model registry, and artifact validation.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from configs.config import (
    PRODUCTION_PIPELINE_V2_PATHS, DOCUMENTED_V2_THRESHOLDS,
    PRODUCTION_PIPELINE_PATHS, MODELS_V2_DIR, PROJECT_ROOT,
)
from configs.sklearn_compat import load_artifact


# ---- XGBoost Pipeline Tests ----

class TestXGBoostPipelineConstruction:
    """Verify the XGBoost pipeline integrates with v2 preprocessing."""

    def test_xgb_configs_exist(self):
        from src.v2.features_v2 import XGB_CONFIGS
        assert len(XGB_CONFIGS) >= 3, "Need at least 3 XGB configs for comparison"
        for name, cfg in XGB_CONFIGS.items():
            assert name.startswith("xgb_"), f"Config name must start with 'xgb_': {name}"

    def test_build_xgb_pipeline_smoke(self):
        from src.v2.features_v2 import build_pipeline, XGB_CONFIGS
        for sex in ("women", "men"):
            for cfg in list(XGB_CONFIGS.keys())[:1]:  # test first config only for speed
                pipe = build_pipeline(sex, {"state"}, cfg)
                assert hasattr(pipe, "fit"), "Pipeline must have fit method"
                assert hasattr(pipe, "predict_proba"), "Pipeline must have predict_proba"

    def test_build_xgb_pipeline_unknown_config_raises(self):
        from src.v2.features_v2 import build_pipeline
        with pytest.raises(ValueError, match="unknown model"):
            build_pipeline("women", set(), "xgb_nonexistent")

    def test_xgb_pipeline_module(self):
        from src.models.xgboost_pipeline import build_xgb_pipeline, XGB_CONFIGS
        pipe = build_xgb_pipeline("women", {"state"}, "xgb_default")
        assert hasattr(pipe, "fit")
        assert hasattr(pipe, "predict_proba")

    def test_xgb_pipeline_module_unknown_config_raises(self):
        from src.models.xgboost_pipeline import build_xgb_pipeline
        with pytest.raises(ValueError, match="Unknown XGBoost config"):
            build_xgb_pipeline("women", {"state"}, "xgb_nonexistent")


class TestXGBoostExperiments:
    """Verify XGBoost experiments are registered in run_v2."""

    def test_xgb_experiments_registered(self):
        from src.v2.run_v2 import EXPERIMENTS, GROUPS
        xgb_names = [k for k in EXPERIMENTS if k.startswith("xgb")]
        assert len(xgb_names) >= 3, f"Need XGBoost experiments, found: {xgb_names}"
        assert "xgb" in GROUPS, "Need 'xgb' group in GROUPS"

    def test_xgb_experiments_use_known_configs(self):
        from src.v2.run_v2 import EXPERIMENTS
        from src.v2.features_v2 import XGB_CONFIGS
        for name, (flags, model) in EXPERIMENTS.items():
            if name.startswith("xgb"):
                assert model in XGB_CONFIGS, f"Experiment {name} uses unknown config {model}"


# ---- Model Registry Tests ----

class TestModelRegistry:
    """Verify model registry functionality."""

    def test_registry_loads(self):
        from src.models.model_registry import load_registry
        reg = load_registry()
        assert len(reg) >= 4, "Registry should have at least 4 entries (v1+v2, women+men)"

    def test_registry_has_approved_models(self):
        from src.models.model_registry import load_registry, get_production_model
        reg = load_registry()
        for sex in ("women", "men"):
            model = get_production_model(sex, reg)
            assert model.status == "approved"
            assert model.sex == sex

    def test_registry_approved_matches_v2_config(self):
        """The approved model paths must match what configs/config.py defines."""
        from src.models.model_registry import load_registry, get_production_model
        reg = load_registry()
        for sex in ("women", "men"):
            model = get_production_model(sex, reg)
            expected_path = PRODUCTION_PIPELINE_V2_PATHS[sex]
            actual_path = model.absolute_path()
            assert actual_path == expected_path, \
                f"Registry path {actual_path} != config path {expected_path}"

    def test_registry_threshold_matches_config(self):
        from src.models.model_registry import load_registry, get_production_model
        reg = load_registry()
        for sex in ("women", "men"):
            model = get_production_model(sex, reg)
            assert model.threshold == DOCUMENTED_V2_THRESHOLDS[sex], \
                f"Registry threshold {model.threshold} != config threshold {DOCUMENTED_V2_THRESHOLDS[sex]}"

    def test_registry_json_valid(self):
        path = PROJECT_ROOT / "configs" / "model_registry.json"
        assert path.exists(), "model_registry.json must exist"
        data = json.loads(path.read_text(encoding="utf-8"))
        for model_id, entry in data.items():
            assert "sex" in entry
            assert "status" in entry
            assert entry["status"] in ("approved", "experimental", "deprecated")
            assert "artifact_path" in entry


# ---- Artifact Integrity Tests ----

class TestArtifactIntegrity:
    """Verify production artifacts are loadable and structurally valid."""

    @pytest.mark.parametrize("sex", ["women", "men"])
    def test_v2_artifact_loads(self, sex):
        path = PRODUCTION_PIPELINE_V2_PATHS[sex]
        if not path.exists():
            pytest.skip(f"V2 artifact not found: {path}")
        artifact = load_artifact(path)
        required = ["base_pipeline", "calibrator", "calibration_method",
                     "feature_cols", "selected_threshold"]
        for key in required:
            assert key in artifact, f"Missing key '{key}' in {sex} v2 artifact"

    @pytest.mark.parametrize("sex", ["women", "men"])
    def test_v2_threshold_matches_config(self, sex):
        path = PRODUCTION_PIPELINE_V2_PATHS[sex]
        if not path.exists():
            pytest.skip(f"V2 artifact not found: {path}")
        artifact = load_artifact(path)
        assert artifact["selected_threshold"] == DOCUMENTED_V2_THRESHOLDS[sex], \
            f"Artifact threshold {artifact['selected_threshold']} != " \
            f"config threshold {DOCUMENTED_V2_THRESHOLDS[sex]}"

    @pytest.mark.parametrize("sex", ["women", "men"])
    def test_v2_artifact_produces_valid_probabilities(self, sex):
        """Smoke test: feed a minimal valid input through the artifact and check output."""
        path = PRODUCTION_PIPELINE_V2_PATHS[sex]
        if not path.exists():
            pytest.skip(f"V2 artifact not found: {path}")
        from src.inference.predict_module_v2 import EarlyDiaPredictorV2
        predictor = EarlyDiaPredictorV2(sex)
        # Build a minimal valid input
        from src.app.input_spec import build_person, ui_from_answers, blank_answers
        ans = blank_answers()
        ans.update(sex=sex, age=30, education=2, residence=1,
                   uses_tobacco=False, drinks=False,
                   bp_sys=120, bp_dia=80, bp_n=3, bp_unavailable=False)
        if sex == "women":
            ans["height_cm"] = 160
            ans["weight_kg"] = 60
        for col, _ in __import__("src.app.input_spec", fromlist=["DIET_ITEMS"]).DIET_ITEMS[sex]:
            ans[col] = 1  # Daily
        ui = ui_from_answers(ans)
        person = build_person(sex, ui)
        if "state" in predictor.feature_cols:
            person["state"] = 10  # arbitrary valid state code
        result = predictor.predict(person)
        assert 0 <= result.calibrated_probability <= 1, \
            f"Probability {result.calibrated_probability} outside [0, 1]"
        assert result.predicted_class in (0, 1)
        assert result.threshold_used == DOCUMENTED_V2_THRESHOLDS[sex]


# ---- Docker / Deployment Tests ----

class TestDockerConfiguration:
    """Verify Dockerfile includes all required artifacts."""

    def test_dockerfile_copies_models_v2(self):
        dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert "COPY models_v2/" in dockerfile, \
            "Dockerfile must copy models_v2/ directory"

    def test_dockerfile_copies_models(self):
        dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert "COPY models/" in dockerfile, \
            "Dockerfile must copy models/ directory"

    def test_dockerignore_excludes_data(self):
        dockerignore = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")
        assert "data" in dockerignore, ".dockerignore must exclude data/"


# ---- Dependency Tests ----

class TestDependencies:
    """Verify requirements files are complete."""

    def test_xgboost_in_requirements(self):
        req = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
        assert "xgboost" in req, "xgboost must be in requirements.txt"

    def test_xgboost_in_backend_requirements(self):
        req = (PROJECT_ROOT / "backend" / "requirements.txt").read_text(encoding="utf-8")
        assert "xgboost" in req, "xgboost must be in backend/requirements.txt"

    def test_networkx_in_requirements(self):
        req = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
        assert "networkx" in req, "networkx must be in requirements.txt"

    def test_dependencies_pinned(self):
        """All core dependencies should use == pinning for reproducibility."""
        req = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
        for line in req.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # All non-comment, non-empty lines should use ==
            assert ">=" not in line or line.startswith("#"), \
                f"Dependency should be pinned with ==, not >=: {line}"


# ---- Feature Schema Parity Tests ----

class TestFeatureSchemaParity:
    """Verify training and inference feature schemas match."""

    @pytest.mark.parametrize("sex", ["women", "men"])
    def test_v2_artifact_feature_cols_match_inference(self, sex):
        path = PRODUCTION_PIPELINE_V2_PATHS[sex]
        if not path.exists():
            pytest.skip(f"V2 artifact not found: {path}")
        artifact = load_artifact(path)
        from src.inference.predict_module_v2 import EarlyDiaPredictorV2
        predictor = EarlyDiaPredictorV2(sex)
        assert predictor.feature_cols == artifact["feature_cols"], \
            "Inference predictor feature_cols must match saved artifact"

    @pytest.mark.parametrize("sex", ["women", "men"])
    def test_no_target_features_in_model(self, sex):
        """No target-construction variables should appear in model features."""
        path = PRODUCTION_PIPELINE_V2_PATHS[sex]
        if not path.exists():
            pytest.skip(f"V2 artifact not found: {path}")
        artifact = load_artifact(path)
        forbidden = {"sb56", "sb57", "s728a", "sb74", "sb70",
                      "smb56", "smb57", "sm728a", "smb74", "smb70",
                      "target_a", "target", "glucose", "known_diabetes",
                      "v213"}  # pregnancy
        feature_set = set(artifact["feature_cols"])
        overlap = feature_set & forbidden
        assert not overlap, f"Forbidden target variables in features: {overlap}"
