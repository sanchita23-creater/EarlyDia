"""
src/train_model.py
------------------
Builds the preprocessing pipeline, splits the data into stratified train/test sets,
trains Logistic Regression and Random Forest models with class imbalance handling,
and serializes the trained pipelines and processed datasets into `models/` and `data/processed/`.
"""

import os
import sys
# Set thread limits to prevent OpenBLAS thread-buffer memory exhaustion on constrained machines
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Ensure src directory is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import time
import joblib
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from feature_selection import NUMERIC_FEATURES, BINARY_FEATURES, SELECTED_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a robust Scikit-learn ColumnTransformer for EarlyDia.
    - Numerical / continuous / ordinal features: StandardScaler.
    - Binary features: Passthrough (0 and 1 are already properly scaled).
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES)
        ],
        remainder="drop"
    )
    return preprocessor


def train_and_save_models(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    models_dir: str = "models"
) -> None:
    """
    Executes the training workflow:
    1. Loads raw dataset and extracts the 16 selected non-invasive features.
    2. Performs stratified 80/20 train/test split.
    3. Fits preprocessor on training data only (no data leakage).
    4. Trains Logistic Regression and Random Forest models with balanced class weights.
    5. Saves test data and trained pipelines.
    """
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load Data
    raw_features_path = os.path.join(raw_dir, "raw_features.csv")
    raw_targets_path = os.path.join(raw_dir, "raw_targets.csv")

    if os.path.exists(raw_features_path) and os.path.exists(raw_targets_path):
        print(f"[EarlyDia] Loading cached features from {raw_features_path}...")
        X_all = pd.read_csv(raw_features_path)
        y_all = pd.read_csv(raw_targets_path)
    else:
        print("[EarlyDia] Fetching directly from ucimlrepo (ID: 891)...")
        dataset = fetch_ucirepo(id=891)
        X_all = dataset.data.features
        y_all = dataset.data.targets

    target_col = y_all.columns[0]
    y = y_all[target_col].astype(int)

    # Filter strictly to the 16 selected features
    X = X_all[SELECTED_FEATURES].copy()
    print(f"[EarlyDia] Selected features ({len(SELECTED_FEATURES)}): {SELECTED_FEATURES}")
    print(f"[EarlyDia] Dataset shape: X={X.shape}, y={y.shape}")

    # 2. Stratified Train / Test Split (80% Train, 20% Test)
    print("\n[EarlyDia] Performing stratified 80/20 train/test split (random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"  - Training samples: {len(X_train):,} ({y_train.mean()*100:.1f}% positive)")
    print(f"  - Testing samples:  {len(X_test):,} ({y_test.mean()*100:.1f}% positive)")

    # Save split data for clean, leak-free evaluation
    test_data_path = os.path.join(processed_dir, "test_data.joblib")
    joblib.dump({"X_test": X_test, "y_test": y_test}, test_data_path)
    print(f"[EarlyDia] Saved test split to: {test_data_path}")

    # 3. Fit Standalone Preprocessing Pipeline
    print("\n[EarlyDia] Fitting preprocessing pipeline on training data...")
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    preproc_path = os.path.join(models_dir, "preprocessing_pipeline.joblib")
    joblib.dump(preprocessor, preproc_path)
    print(f"[EarlyDia] Saved preprocessing pipeline to: {preproc_path}")

    # 4. Model 1: Logistic Regression Pipeline
    print("\n" + "=" * 60)
    print("[EarlyDia] Training Model 1: Logistic Regression (class_weight='balanced')...")
    print("=" * 60)
    lr_pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=42
        ))
    ])
    start_lr = time.time()
    lr_pipeline.fit(X_train, y_train)
    lr_time = time.time() - start_lr
    print(f"[EarlyDia] Logistic Regression trained in {lr_time:.2f} seconds.")
    
    lr_model_path = os.path.join(models_dir, "logistic_regression_pipeline.joblib")
    joblib.dump(lr_pipeline, lr_model_path)
    print(f"[EarlyDia] Saved Logistic Regression pipeline to: {lr_model_path}")

    # 5. Model 2: Random Forest Classifier Pipeline
    print("\n" + "=" * 60)
    print("[EarlyDia] Training Model 2: Random Forest (class_weight='balanced')...")
    print("=" * 60)
    rf_pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=2
        ))
    ])
    start_rf = time.time()
    rf_pipeline.fit(X_train, y_train)
    rf_time = time.time() - start_rf
    print(f"[EarlyDia] Random Forest trained in {rf_time:.2f} seconds.")
    
    rf_model_path = os.path.join(models_dir, "random_forest_pipeline.joblib")
    joblib.dump(rf_pipeline, rf_model_path)
    print(f"[EarlyDia] Saved Random Forest pipeline to: {rf_model_path}")

    print("\n[EarlyDia] All models trained and serialized successfully!\n")


if __name__ == "__main__":
    train_and_save_models()
