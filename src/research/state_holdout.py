"""State-stratified evaluation of the LOCKED production model on the final test set (research-only).
Leave-one-state-out retraining is NOT performed (would require retraining; production stays locked).
Geographic differences are descriptive only and must not be read causally.
State codes come from v024/mv024 joined by caseid (not a model predictor)."""
import numpy as np
import pandas as pd

from src.research.common import OUT, DATA_PROCESSED, THRESHOLDS, ensure_dirs, metrics_by_group, write_json


def run():
    ensure_dirs()
    probe = DATA_PROCESSED / "women_test_predictions.parquet"
    if not probe.exists() and not (DATA_PROCESSED / "women_test_predictions.pkl").exists():
        write_json(OUT / "state_holdout_status.json", {"status": "NOT_RUN",
                   "reason": "Needs row-level locked test predictions + state codes (data/processed); not present in this build."})
        return None
    rows = []
    for sex in ("women", "men"):
        d = pd.read_pickle(DATA_PROCESSED / f"{sex}_test_predictions.pkl")   # columns: y, p, state
        for r in metrics_by_group(d["y"], d["p"], d["state"], THRESHOLDS[sex], min_n=300, min_pos=15):
            r["sex"] = sex; rows.append(r)
    pd.DataFrame(rows).rename(columns={"group": "state"}).to_csv(OUT / "state_holdout.csv", index=False)
    return rows
