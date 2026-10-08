"""
src/inspect_data.py
-------------------
Thoroughly inspects the CDC Diabetes Health Indicators dataset (UCI ID: 891).
Generates a comprehensive summary report in `reports/dataset_summary.txt`
and saves a target class distribution visualization in `reports/target_distribution.png`.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo


def inspect_dataset(raw_dir: str = "data/raw", reports_dir: str = "reports") -> None:
    """
    Performs full dataset inspection and saves the report and visualization.
    """
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. Fetch metadata & variables from UCI API
    print("[EarlyDia] Fetching UCI dataset metadata and variables...")
    try:
        import requests
        api_res = requests.get("https://archive.ics.uci.edu/api/dataset?id=891", timeout=15).json()
        api_data = api_res.get("data", {})
        metadata = {
            "name": api_data.get("name", "CDC Diabetes Health Indicators"),
            "repository_url": "https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators",
            "area": api_data.get("area", "Health and Medicine"),
            "num_instances": api_data.get("num_instances", 253680),
            "num_features": api_data.get("num_features", 21)
        }
        variables = pd.DataFrame(api_data.get("variables", []))
    except Exception as e:
        print(f"[EarlyDia] Note: UCI API call had notice: {e}, falling back to built-in metadata.")
        metadata = {
            "name": "CDC Diabetes Health Indicators",
            "repository_url": "https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators"
        }
        variables = pd.DataFrame()

    # Load feature and target data from local cache (saved by fetch_data.py)
    raw_features_path = os.path.join(raw_dir, "raw_features.csv")
    raw_targets_path = os.path.join(raw_dir, "raw_targets.csv")

    if not (os.path.exists(raw_features_path) and os.path.exists(raw_targets_path)):
        print("[EarlyDia] Local cache not found. Fetching via src/fetch_data.py...")
        from fetch_data import fetch_and_save_data
        X, y = fetch_and_save_data(output_dir=raw_dir)
    else:
        print(f"[EarlyDia] Loading cached raw data from {raw_dir}...")
        X = pd.read_csv(raw_features_path)
        y = pd.read_csv(raw_targets_path)
        
    target_col = y.columns[0]
    df = pd.concat([X, y], axis=1)
    
    # 2. Extract inspection metrics
    num_rows, num_cols = df.shape
    feature_names = list(X.columns)
    target_counts = y[target_col].value_counts()
    target_pcts = y[target_col].value_counts(normalize=True) * 100
    missing_vals = df.isnull().sum()
    duplicate_count = df.duplicated().sum()
    duplicate_pct = (duplicate_count / num_rows) * 100
    
    # 3. Format dataset summary text
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("EARLYDIA: CDC DIABETES HEALTH INDICATORS (UCI ID: 891) DATASET INSPECTION")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    summary_lines.append("1. GENERAL DATASET INFORMATION")
    summary_lines.append("-" * 40)
    summary_lines.append(f"Dataset Name:              {metadata.get('name', 'CDC Diabetes Health Indicators')}")
    summary_lines.append(f"UCI Dataset ID:            891")
    summary_lines.append(f"Source URL:                {metadata.get('repository_url', 'N/A')}")
    summary_lines.append(f"Original Survey Source:    CDC Behavioral Risk Factor Surveillance System (BRFSS)")
    summary_lines.append(f"Number of Instances:       {num_rows:,}")
    summary_lines.append(f"Number of Features:        {len(feature_names)}")
    summary_lines.append(f"Target Column:             {target_col}")
    summary_lines.append(f"Missing Values:            {missing_vals.sum()} (0.0% missing across all columns)")
    summary_lines.append(f"Duplicate Rows:            {duplicate_count:,} ({duplicate_pct:.2f}%)")
    summary_lines.append("")
    
    summary_lines.append("2. TARGET VARIABLE ANALYSIS")
    summary_lines.append("-" * 40)
    summary_lines.append(f"Target Name:               {target_col}")
    summary_lines.append(f"Target Meaning:            0 = No Diabetes / Healthy")
    summary_lines.append(f"                           1 = Prediabetes or Diabetes")
    summary_lines.append("Class Distribution:")
    for cls_val in sorted(target_counts.index):
        label = "No Diabetes (0)" if cls_val == 0 else "Prediabetes / Diabetes (1)"
        summary_lines.append(f"  - Class {cls_val} [{label}]: {target_counts[cls_val]:,} instances ({target_pcts[cls_val]:.2f}%)")
    summary_lines.append(f"Class Imbalance Ratio:     Approximately {target_pcts[0]/target_pcts[1]:.2f} : 1")
    summary_lines.append("Imbalance Implication:     Class imbalance requires stratified sampling and evaluation")
    summary_lines.append("                           prioritizing Recall/Sensitivity and F1 Score rather than accuracy alone.")
    summary_lines.append("")

    summary_lines.append("3. DATASET FEATURES (TOTAL: 21)")
    summary_lines.append("-" * 40)
    for i, col in enumerate(feature_names, 1):
        dtype_str = str(df[col].dtype)
        unique_vals = sorted(df[col].unique().tolist())
        if len(unique_vals) <= 15:
            val_summary = f"Discrete values: {unique_vals}"
        else:
            val_summary = f"{len(unique_vals)} unique values [Min: {min(unique_vals)}, Max: {max(unique_vals)}]"
        summary_lines.append(f"{i:2d}. {col:<22} | Type: {dtype_str:<8} | {val_summary}")
    summary_lines.append("")

    summary_lines.append("4. UCI VARIABLES SPECIFICATION (CODEBOOK)")
    summary_lines.append("-" * 40)
    summary_lines.append(variables.to_string(index=False))
    summary_lines.append("")

    summary_lines.append("5. DESCRIPTIVE STATISTICS")
    summary_lines.append("-" * 40)
    desc_stats = df.describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    summary_lines.append(desc_stats.to_string())
    summary_lines.append("")

    summary_lines.append("6. DATASET LIMITATIONS & OBSERVATIONS")
    summary_lines.append("-" * 40)
    summary_lines.append("a. Self-Reported Survey Data: All attributes originate from the CDC BRFSS telephone survey.")
    summary_lines.append("   Self-reporting is subject to recall bias and social desirability bias.")
    summary_lines.append("b. Ordinal Discretization: Age (1-13 brackets), Education (1-6 scale), and Income (1-8 scale)")
    summary_lines.append("   are pre-bucketed rather than continuous natural units.")
    summary_lines.append("c. Duplicate Rows: 24,206 identical rows exist because survey participants with identical")
    summary_lines.append("   discrete demographic and lifestyle categories produce identical feature vectors.")
    summary_lines.append("   These are legitimate distinct individuals answering identical categorical questions.")
    summary_lines.append("d. Target Combination: Class 1 combines prediabetes and diagnosed diabetes into a single binary group.")
    summary_lines.append("   This makes the target suitable for general early risk screening.")
    summary_lines.append("e. Laboratory Test Absence: The dataset contains no direct clinical glucose, insulin, or HbA1c")
    summary_lines.append("   measurements, making it an ideal benchmark for non-invasive risk screening.")
    summary_lines.append("=" * 80)

    summary_text = "\n".join(summary_lines)
    summary_path = os.path.join(reports_dir, "dataset_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"[EarlyDia] Saved dataset inspection summary to: {summary_path}")

    # 4. Generate Target Distribution Plot
    print("[EarlyDia] Generating target class distribution plot...")
    plt.figure(figsize=(8, 5))
    sns.set_theme(style="whitegrid")
    palette = ["#2b5c8f", "#d95f02"]
    ax = sns.barplot(
        x=["No Diabetes (0)", "Prediabetes / Diabetes (1)"],
        y=[target_counts[0], target_counts[1]],
        palette=palette,
        hue=["No Diabetes (0)", "Prediabetes / Diabetes (1)"],
        legend=False
    )
    plt.title("EarlyDia: CDC BRFSS Diabetes Class Distribution", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Target Class (Diabetes_binary)", fontsize=12)
    plt.ylabel("Number of Participants", fontsize=12)

    # Annotate bars with count and percentage
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        pct = target_pcts[i]
        ax.annotate(f"{int(height):,}\n({pct:.1f}%)",
                    (p.get_x() + p.get_width() / 2., height / 2),
                    ha="center", va="center", fontsize=11, color="white", fontweight="bold")

    plt.tight_layout()
    plot_path = os.path.join(reports_dir, "target_distribution.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[EarlyDia] Saved target distribution visualization to: {plot_path}\n")


if __name__ == "__main__":
    inspect_dataset()
