"""Exploratory error analysis for (stable) Mapper communities and arbitrary groupings, at the LOCKED thresholds.
Does not modify the production model."""
import pandas as pd

from src.research.common import metrics_by_group


def community_error_table(point_community, y, p, thr, stable_mask=None, min_n=200, min_pos=10):
    rows = metrics_by_group(y, p, point_community, thr, min_n, min_pos)
    df = pd.DataFrame(rows)
    df = df[df["group"] >= 0] if "group" in df and len(df) else df
    if stable_mask is not None and len(df):
        df["stable"] = [bool(stable_mask[int(c)]) if int(c) < len(stable_mask) else False for c in df["group"]]
    if "false_negative_rate" in df.columns or "fnr" in df.columns:
        for col in ("fnr", "fpr", "brier_score"):
            if col in df.columns and df[col].notna().sum() > 2:
                df[f"{col}_outlier"] = df[col] > df[col].mean() + df[col].std()
    return df
