"""Publication-style figures from the research outputs (matplotlib, Agg). Only figures whose inputs exist are drawn."""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.research.common import OUT, FIG, THRESHOLDS, ensure_dirs, load_final_eval

C = {"women": "#0F766E", "men": "#B45309"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})


def _csv(name):
    with open(OUT / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / name, bbox_inches="tight"); plt.close(fig)


def population_flow():
    # counts recorded in the project's locked documentation (women: README / target_construction asserts)
    steps = [("Women 18-40", 510289), ("Not pregnant", 482463), ("Usable Target A (non-pregnant)", 434502)]
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    ax.barh([s for s, _ in steps][::-1], [n for _, n in steps][::-1], color="#0F766E")
    for i, (_, n) in enumerate(steps[::-1]):
        ax.text(n, i, f" {n:,}", va="center")
    ax.set_xlim(0, 620000); ax.set_title("Study population flow - women (final test: 86,669 of the usable rows)")
    _save(fig, "01_study_population_flow.png")


def prevalence_by_sex():
    fe = load_final_eval(); fig, ax = plt.subplots(figsize=(4, 3.2))
    for i, s in enumerate(("women", "men")):
        ax.bar(i, fe[s]["overall"]["prevalence"] * 100, color=C[s])
    ax.set_xticks([0, 1], ["Women", "Men"]); ax.set_ylabel("Target A prevalence, final test (%)")
    ax.set_title("Unweighted test-set prevalence"); _save(fig, "03_prevalence_by_sex.png")


def calibration():
    rows = _csv("calibration_curves/reliability_points.csv"); fig, ax = plt.subplots(figsize=(4.8, 4.4))
    for s in ("women", "men"):
        r = [x for x in rows if x["sex"] == s]
        ax.plot([float(x["mean_predicted"]) for x in r], [float(x["observed_fraction"]) for x in r], "o-", color=C[s], label=s.title())
    ax.plot([0, .11], [0, .11], "k--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability (decile)"); ax.set_ylabel("Observed fraction positive")
    ax.set_title("Calibration on locked final test"); ax.legend(); _save(fig, "06_calibration.png")


def decision_curve():
    rows = _csv("decision_curve.csv"); fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=False)
    for ax, s in zip(axes, ("women", "men")):
        r = [x for x in rows if x["sex"] == s and x["kind"] == "curve"]; t = [float(x["threshold"]) for x in r]
        ax.plot(t, [float(x["net_benefit_model"]) for x in r], color=C[s], label="EarlyDia")
        ax.plot(t, [float(x["net_benefit_treat_all"]) for x in r], "k:", label="Treat all")
        ax.axhline(0, color="gray", lw=1, label="Treat none"); ax.axvline(THRESHOLDS[s], color="red", lw=.8, ls="--")
        ax.set_title(f"{s.title()} (approx., decile bins)"); ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
    axes[0].legend(); fig.suptitle("Decision curve (potential decision usefulness; not clinical utility)", y=1.02)
    _save(fig, "07_decision_curve.png")


def age_subgroups():
    rows = [r for r in _csv("subgroup_performance.csv") if r["subgroup"].startswith("age")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, (m, title) in zip(axes, [("sensitivity", "Sensitivity"), ("specificity", "Specificity"), ("roc_auc", "ROC-AUC")]):
        for j, s in enumerate(("women", "men")):
            for i, g in enumerate(("age_18_29", "age_30_40")):
                v = float([r for r in rows if r["sex"] == s and r["subgroup"] == g][0][m])
                ax.bar(i + j * 0.38 - 0.19, v, 0.36, color=C[s], label=s.title() if i == 0 else None)
        ax.set_xticks([0, 1], ["18-29", "30-40"]); ax.set_title(title); ax.set_ylim(0, 1)
    axes[0].legend(); fig.suptitle("Age-subgroup performance at the locked sex-specific thresholds", y=1.02)
    _save(fig, "08_age_subgroup_performance.png")


def feature_importance():
    rows = _csv("feature_importance.csv"); fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, s in zip(axes, ("women", "men")):
        r = [x for x in rows if x["sex"] == s][:12][::-1]
        ax.barh([x["label"] for x in r], [float(x["importance_mean"]) for x in r], xerr=[float(x["importance_std"]) for x in r], color=C[s])
        ax.set_title(f"{s.title()} - permutation importance"); ax.set_xlabel("Mean drop in score")
    _save(fig, "10_feature_importance.png")


def run():
    ensure_dirs()
    for f in (population_flow, prevalence_by_sex, calibration, decision_curve, age_subgroups, feature_importance):
        f()
