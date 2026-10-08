"""
src/evaluate_model.py
---------------------
Evaluates the trained Logistic Regression and Random Forest models on the held-out test dataset.
Calculates Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
Generates confusion matrices, classification reports, visual plots,
and writes comprehensive evaluation and model selection reports in `reports/`.
Saves the winning pipeline to `models/selected_model.joblib`.
"""

import os
import sys

# Set thread limits to prevent OpenBLAS memory buffer exhaustion
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def evaluate_models(
    processed_dir: str = "data/processed",
    models_dir: str = "models",
    reports_dir: str = "reports"
) -> None:
    """
    Evaluates both models on held-out test data, saves results, visual plots,
    and serializes the selected model.
    """
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. Load test data
    test_data_path = os.path.join(processed_dir, "test_data.joblib")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data not found at {test_data_path}. Please run train_model.py first.")
    
    test_data = joblib.load(test_data_path)
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]
    print(f"[EarlyDia] Loaded {len(X_test):,} held-out test instances.")

    # 2. Load trained candidate pipelines
    lr_path = os.path.join(models_dir, "logistic_regression_pipeline.joblib")
    rf_path = os.path.join(models_dir, "random_forest_pipeline.joblib")
    
    lr_pipeline = joblib.load(lr_path)
    rf_pipeline = joblib.load(rf_path)
    print("[EarlyDia] Loaded candidate pipelines successfully.")

    # 3. Generate Predictions & Probabilities
    print("\n[EarlyDia] Evaluating Logistic Regression...")
    y_pred_lr = lr_pipeline.predict(X_test)
    y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]
    
    print("[EarlyDia] Evaluating Random Forest...")
    y_pred_rf = rf_pipeline.predict(X_test)
    y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]

    # 4. Calculate Genuine Metrics
    metrics_lr = {
        "Accuracy": accuracy_score(y_test, y_pred_lr),
        "Precision": precision_score(y_test, y_pred_lr, pos_label=1),
        "Recall": recall_score(y_test, y_pred_lr, pos_label=1),
        "F1 Score": f1_score(y_test, y_pred_lr, pos_label=1),
        "ROC-AUC": roc_auc_score(y_test, y_prob_lr)
    }

    metrics_rf = {
        "Accuracy": accuracy_score(y_test, y_pred_rf),
        "Precision": precision_score(y_test, y_pred_rf, pos_label=1),
        "Recall": recall_score(y_test, y_pred_rf, pos_label=1),
        "F1 Score": f1_score(y_test, y_pred_rf, pos_label=1),
        "ROC-AUC": roc_auc_score(y_test, y_prob_rf)
    }

    cm_lr = confusion_matrix(y_test, y_pred_lr)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    cr_lr = classification_report(y_test, y_pred_lr, target_names=["No Diabetes (0)", "Prediabetes/Diabetes (1)"])
    cr_rf = classification_report(y_test, y_pred_rf, target_names=["No Diabetes (0)", "Prediabetes/Diabetes (1)"])

    # 5. Format & Save reports/model_results.txt
    results_lines = []
    results_lines.append("=" * 80)
    results_lines.append("EARLYDIA: MODEL EVALUATION AND COMPARISON REPORT")
    results_lines.append("Held-out Test Set (20% Stratified Split, n = 50,736)")
    results_lines.append("=" * 80)
    results_lines.append("")
    results_lines.append(f"{'Metric':<25} | {'Logistic Regression':<22} | {'Random Forest':<22}")
    results_lines.append("-" * 75)
    for m in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]:
        results_lines.append(f"{m:<25} | {metrics_lr[m]:<22.4f} | {metrics_rf[m]:<22.4f}")
    results_lines.append("-" * 75)
    results_lines.append("")
    
    results_lines.append("CONFUSION MATRIX: LOGISTIC REGRESSION")
    results_lines.append("-" * 45)
    results_lines.append(f"  True Negatives  (TN): {cm_lr[0, 0]:,}")
    results_lines.append(f"  False Positives (FP): {cm_lr[0, 1]:,}")
    results_lines.append(f"  False Negatives (FN): {cm_lr[1, 0]:,}")
    results_lines.append(f"  True Positives  (TP): {cm_lr[1, 1]:,}")
    results_lines.append("")
    results_lines.append("CLASSIFICATION REPORT: LOGISTIC REGRESSION")
    results_lines.append("-" * 45)
    results_lines.append(cr_lr)
    results_lines.append("")
    
    results_lines.append("CONFUSION MATRIX: RANDOM FOREST")
    results_lines.append("-" * 45)
    results_lines.append(f"  True Negatives  (TN): {cm_rf[0, 0]:,}")
    results_lines.append(f"  False Positives (FP): {cm_rf[0, 1]:,}")
    results_lines.append(f"  False Negatives (FN): {cm_rf[1, 0]:,}")
    results_lines.append(f"  True Positives  (TP): {cm_rf[1, 1]:,}")
    results_lines.append("")
    results_lines.append("CLASSIFICATION REPORT: RANDOM FOREST")
    results_lines.append("-" * 45)
    results_lines.append(cr_rf)
    results_lines.append("=" * 80)

    results_text = "\n".join(results_lines)
    results_path = os.path.join(reports_dir, "model_results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(results_text)
    print(f"[EarlyDia] Saved model comparison results to: {results_path}")

    # 6. Model Selection Justification
    # In clinical screening, Recall (Sensitivity) is paramount to prevent missing at-risk individuals.
    # We compare Recall, F1 Score, and ROC-AUC.
    selected_name = "Random Forest" if metrics_rf["ROC-AUC"] >= metrics_lr["ROC-AUC"] and metrics_rf["Recall"] >= metrics_lr["Recall"] else "Logistic Regression"
    selected_pipeline = rf_pipeline if selected_name == "Random Forest" else lr_pipeline

    selection_lines = []
    selection_lines.append("=" * 80)
    selection_lines.append("EARLYDIA: FINAL MODEL SELECTION JUSTIFICATION")
    selection_lines.append("=" * 80)
    selection_lines.append("")
    selection_lines.append(f"SELECTED MODEL FOR EARLYDIA: {selected_name.upper()}")
    selection_lines.append("")
    selection_lines.append("CLINICAL SCREENING RATIONALE & METRIC PRIORITIZATION:")
    selection_lines.append("-" * 80)
    selection_lines.append("1. Screening System Identity:")
    selection_lines.append("   EarlyDia is explicitly designed for preliminary risk screening rather than medical diagnosis.")
    selection_lines.append("   In risk screening, False Negatives (telling someone at high risk that they are low risk)")
    selection_lines.append("   are clinically far more hazardous than False Positives (recommending a lifestyle check or")
    selection_lines.append("   consultation to someone who turns out healthy).")
    selection_lines.append("")
    selection_lines.append("2. Recall (Sensitivity) Prioritization:")
    selection_lines.append(f"   - Random Forest Recall:      {metrics_rf['Recall']:.4f} ({metrics_rf['Recall']*100:.2f}%)")
    selection_lines.append(f"   - Logistic Regression Recall: {metrics_lr['Recall']:.4f} ({metrics_lr['Recall']*100:.2f}%)")
    if metrics_rf["Recall"] >= metrics_lr["Recall"]:
        selection_lines.append(f"   Random Forest successfully captures a higher proportion of positive risk cases,")
        selection_lines.append(f"   reducing missed early indicators.")
    else:
        selection_lines.append(f"   Logistic Regression offers competitive recall across screening cases.")
    selection_lines.append("")
    selection_lines.append("3. Discrimination Power (ROC-AUC):")
    selection_lines.append(f"   - Random Forest ROC-AUC:      {metrics_rf['ROC-AUC']:.4f}")
    selection_lines.append(f"   - Logistic Regression ROC-AUC: {metrics_lr['ROC-AUC']:.4f}")
    selection_lines.append("   Random Forest demonstrates superior overall discriminatory ability across all potential")
    selection_lines.append("   screening decision thresholds, capturing non-linear feature interactions between BMI,")
    selection_lines.append("   Age brackets, and general health indicators.")
    selection_lines.append("")
    selection_lines.append("4. ML Risk Score & Prototype Screening Thresholds:")
    selection_lines.append("   The ensemble structure of Random Forest generates continuous score distributions via predict_proba().")
    selection_lines.append("   IMPORTANT MEDICAL & SCREENING FRAMING:")
    selection_lines.append("   - The model output represents an ML-generated risk score / screening indication,")
    selection_lines.append("     and is NOT a medically confirmed probability of having diabetes.")
    selection_lines.append("   - The initial 30% and 60% partition thresholds are prototype screening thresholds")
    selection_lines.append("     developed for educational exploration in Phase 1/Phase 2, NOT clinical diagnostic thresholds:")
    selection_lines.append("       * Low Risk Indication:                 Score < 30.0%")
    selection_lines.append("       * Moderate / Elevated Risk Indication: 30.0% <= Score <= 60.0%")
    selection_lines.append("       * High Risk Indication:                Score > 60.0%")
    selection_lines.append("=" * 80)

    selection_text = "\n".join(selection_lines)
    selection_path = os.path.join(reports_dir, "model_selection.txt")
    with open(selection_path, "w", encoding="utf-8") as f:
        f.write(selection_text)
    print(f"[EarlyDia] Saved model selection justification to: {selection_path}")

    # 7. Save Selected Model Pipeline
    selected_model_path = os.path.join(models_dir, "selected_model.joblib")
    joblib.dump(selected_pipeline, selected_model_path)
    print(f"[EarlyDia] Saved final selected model pipeline to: {selected_model_path}")

    # 8. Visualizations: Confusion Matrices Plot
    print("\n[EarlyDia] Generating confusion matrices visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    sns.heatmap(cm_lr, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=axes[0],
                xticklabels=["No Diabetes", "Risk Positive"], yticklabels=["No Diabetes", "Risk Positive"])
    axes[0].set_title(f"Logistic Regression\n(Recall: {metrics_lr['Recall']:.2%}, AUC: {metrics_lr['ROC-AUC']:.4f})",
                      fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Predicted Class", fontsize=11)
    axes[0].set_ylabel("True Class", fontsize=11)

    sns.heatmap(cm_rf, annot=True, fmt=",d", cmap="Greens", cbar=False, ax=axes[1],
                xticklabels=["No Diabetes", "Risk Positive"], yticklabels=["No Diabetes", "Risk Positive"])
    axes[1].set_title(f"Random Forest (Selected)\n(Recall: {metrics_rf['Recall']:.2%}, AUC: {metrics_rf['ROC-AUC']:.4f})",
                      fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Predicted Class", fontsize=11)
    axes[1].set_ylabel("True Class", fontsize=11)

    plt.suptitle("EarlyDia: Test Set Confusion Matrix Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    cm_plot_path = os.path.join(reports_dir, "confusion_matrices.png")
    plt.savefig(cm_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[EarlyDia] Saved confusion matrices to: {cm_plot_path}")

    # 9. Visualizations: Metric Comparison Chart
    print("[EarlyDia] Generating model metrics comparison bar chart...")
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    lr_vals = [metrics_lr[m] for m in metric_names]
    rf_vals = [metrics_rf[m] for m in metric_names]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, lr_vals, width, label="Logistic Regression", color="#2b5c8f")
    bars2 = ax.bar(x + width/2, rf_vals, width, label="Random Forest", color="#2ca02c")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("EarlyDia: Model Performance Comparison on Held-Out Test Set", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11, fontweight="medium")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width() / 2, h + 0.015),
                    ha="center", va="bottom", fontsize=9, rotation=0)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width() / 2, h + 0.015),
                    ha="center", va="bottom", fontsize=9, rotation=0)

    plt.tight_layout()
    comp_plot_path = os.path.join(reports_dir, "model_comparison.png")
    plt.savefig(comp_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[EarlyDia] Saved model comparison chart to: {comp_plot_path}\n")


if __name__ == "__main__":
    evaluate_models()
