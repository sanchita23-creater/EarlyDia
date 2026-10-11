# Earlydia Baseline Audit Report

## 1. Production Startup and Trace
- **Start command**: The application is designed to be served by FastAPI (`uvicorn backend.api.main:app --host 0.0.0.0 --port 8000`). It serves both the backend API and the compiled React frontend static files.
- **Default Inference Module**: The default is explicitly version `v2`. `backend.api.main` loads predictors via `backend.inference.engine.load_all(version="v2")`, which initializes `EarlyDiaPredictorV2(sex)`.
- **Request Flow**: 
  React form -> `ScreeningRequest` (Pydantic schema) -> `backend.api.service.run_screening(req)` -> input validation -> `src.app.input_spec` mapping -> `backend.inference.engine.get_predictor(sex, "v2")` -> `predict(person)` -> Base pipeline preprocessing -> Base model inference -> Probability Calibration (sigmoid/isotonic) -> Threshold check -> Response JSON -> React `Results.jsx` view.

## 2. Model Training and Versions
- **Legacy Training**: `src/train_model.py` (UCI BRFSS data, 16 features, non-grouped split, deprecated architecture).
- **V2 Training (Authoritative)**: Uses NFHS data.
  - V2 uses `src.v2.features_v2` for preprocessing and `src.v2.run_v2` for generating out-of-fold (OOF) predictions and evaluation.
  - V2 pipeline uses `StratifiedGroupKFold` on PSU clusters.
- **Existing Model Versions**:
  - `v1`: `women_production_pipeline.pkl`, `men_production_pipeline.pkl`.
  - `v2`: `women_v2_pipeline.pkl`, `men_v2_pipeline.pkl`.
- **Thresholds**: 
  - `v1`: Women (0.026), Men (0.027).
  - `v2`: Women (0.024), Men (0.029).

## 3. Test Suite Execution
- **Run command**: `python -m pytest tests/`
- **Results**: 168 passed, 2 failed.
- **Failures**: 
  - `tests/test_research_integrity.py::test_mapper_reproducible_and_target_free`
  - `tests/test_research_integrity.py::test_mapper_stability_and_null_model`
- **Cause**: `ModuleNotFoundError: No module named 'networkx'`. The `networkx` library is used by the research code but missing from `requirements.txt`.
- **Frontend tests**: Uncommitted changes are present in the working tree. Tests are set up via Vite but not executed during backend `pytest`.

## 4. Environment & Dependencies
- `fastapi`, `uvicorn`, `pydantic` have loose bindings (`>=`) in `requirements.txt` and `backend/requirements.txt`.
- Docker build missing `models_v2/` directory from the final image stage.
