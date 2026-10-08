"""
src/fetch_data.py
-----------------
Fetches the CDC Diabetes Health Indicators dataset directly from the
UCI Machine Learning Repository (Dataset ID: 891) using the ucimlrepo library.
Saves the raw feature and target data to `data/raw/` for reproducible local access.
"""

import os
import pandas as pd
from ucimlrepo import fetch_ucirepo


def fetch_and_save_data(output_dir: str = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetches dataset ID 891 from UCI ML Repository and saves raw copies locally.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Features (X) and Target (y) DataFrames.
    """
    print("[EarlyDia] Fetching CDC Diabetes Health Indicators dataset from UCI (ID: 891)...")
    
    # Fetch from UCI repository
    cdc_diabetes = fetch_ucirepo(id=891)
    
    X = cdc_diabetes.data.features
    y = cdc_diabetes.data.targets
    
    print(f"[EarlyDia] Successfully fetched dataset!")
    print(f"[EarlyDia] Features shape: {X.shape}")
    print(f"[EarlyDia] Targets shape:  {y.shape}")
    
    # Ensure raw data directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw data as CSV for transparency and beginner readability
    raw_features_path = os.path.join(output_dir, "raw_features.csv")
    raw_targets_path = os.path.join(output_dir, "raw_targets.csv")
    
    print(f"[EarlyDia] Saving raw features to: {raw_features_path}")
    X.to_csv(raw_features_path, index=False)
    
    print(f"[EarlyDia] Saving raw targets to:  {raw_targets_path}")
    y.to_csv(raw_targets_path, index=False)
    
    print("[EarlyDia] Data fetch and local caching complete!\n")
    return X, y


if __name__ == "__main__":
    fetch_and_save_data()
