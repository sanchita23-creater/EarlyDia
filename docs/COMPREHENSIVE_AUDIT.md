# Earlydia Comprehensive Audit Report

## 1. Docker-Models-V2
- **Severity**: Critical
- **File**: `Dockerfile`, lines 20-25
- **Behavior**: Copies `models/` directory, but fails to copy `models_v2/`.
- **Why it is wrong/risky**: The backend is configured to use `v2` as the default model version, but `models_v2/` is missing from the production container.
- **Root Cause**: The Dockerfile was not updated when the project transitioned to `v2` models.
- **Consequences**: The deployed Docker container will fail on startup because it cannot load the default v2 model artifacts.
- **Recommended correction**: Add `COPY models_v2/ models_v2/` to the Dockerfile.
- **Implementation status**: Pending.
- **Test**: Docker container build and healthcheck via `docker run` / `api/health`.

## 2. Dependency-Pinning-Loose
- **Severity**: Low
- **File**: `requirements.txt` and `backend/requirements.txt`
- **Behavior**: Uses `>=` for `fastapi`, `uvicorn`, `pydantic`.
- **Why it is wrong/risky**: Loose dependency pinning can cause unexpected breaking changes in production when underlying libraries are updated.
- **Root Cause**: Standard practice of loose bindings during prototyping was not tightened for production.
- **Consequences**: Irreproducible builds in the future.
- **Recommended correction**: Pin exact versions.
- **Implementation status**: Pending.
- **Test**: Verify tests run after pinning.

## 3. Missing-Test-Dependency-Networkx
- **Severity**: Low
- **File**: `requirements.txt`
- **Behavior**: `pytest` fails on `test_research_integrity.py` due to `ModuleNotFoundError: No module named 'networkx'`.
- **Why it is wrong/risky**: Breaks the baseline test suite.
- **Root Cause**: The `networkx` library is used by the `mapper_analysis.py` research code but is not documented in the requirements.
- **Consequences**: Developers cannot fully execute the test suite out of the box.
- **Recommended correction**: Add `networkx` to `requirements.txt`.
- **Implementation status**: Pending.
- **Test**: Run `python -m pytest tests/test_research_integrity.py`.

## 4. Legacy-Architecture-Confusion
- **Severity**: Medium
- **File**: `src/train_model.py` and `src/predict.py`
- **Behavior**: Parallel pipeline architectures exist. The legacy script uses `train_test_split` on a UCI dataset, while the `v2` production pipeline uses PSU-aware stratified splitting on NFHS data.
- **Why it is wrong/risky**: Ambiguity about which pipeline is authoritative.
- **Root Cause**: Older code was kept for backward compatibility and tests without explicit deprecation warnings in the files themselves.
- **Consequences**: Developers might mistakenly use the legacy unguarded scripts for the NFHS dataset, causing leakage.
- **Recommended correction**: Establish `v2` as the single authoritative production ML pipeline.
- **Implementation status**: Done. The architectural separation is clear in API loading and `v2` logic. V2 strictly uses PSU-aware folding. No deletion of legacy tests is needed per instructions, but usage is clear.

## 5. State-Imputation-Fallback
- **Severity**: Low
- **File**: `backend/api/service.py`, lines 197-208
- **Behavior**: If `state` is not provided to the v2 model, it imputes `DEFAULT_STATE_CODE` and issues a warning.
- **Why it is wrong/risky**: The v2 model utilizes `state` or `state_native` as a feature. Imputing a default region might slightly skew results for individuals from different contexts.
- **Root Cause**: A deliberate design choice to gracefully handle missing regional context.
- **Consequences**: Acceptable risk, handled via API warning in the technical block, which is correctly exposed in the response.
- **Recommended correction**: None required, but must ensure XGBoost integration handles missing/imputed state similarly.
- **Implementation status**: Verified safe.

## 6. XGBoost-Not-Integrated
- **Severity**: Critical
- **File**: Missing from `requirements.txt` and `src/v2/features_v2.py`.
- **Behavior**: The project does not currently support XGBoost.
- **Root Cause**: Not yet implemented.
- **Consequences**: Cannot evaluate XGBoost candidate models.
- **Recommended correction**: Implement `xgboost`, add to requirements, and add XGBoost pipeline definitions to `src/v2/features_v2.py`.
- **Implementation status**: Pending.
