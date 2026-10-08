"""
EarlyDia - Final predictor construction (women + men)
Operates on the validated women_target_a.pkl / men_target_a.pkl (Target A already verified).
Produces modelling-ready predictor matrices, keeping only target_a in {0,1} rows,
while the full audit datasets remain untouched on disk.
"""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from configs.config import (DATA_RAW_DIR, DATA_INTERIM_DIR, DATA_PROCESSED_DIR,
                            EXPLAINABILITY_REPORTS_DIR)
import pandas as pd
import numpy as np

TARGET_CONSTRUCTION_VARS_W = {"sb56", "sb57", "s728a", "sb74"}
TARGET_CONSTRUCTION_VARS_M = {"smb56", "smb57", "sm627a", "smb74"}

BP_SYSTOLIC_RANGE = (70, 270)   # plausible physiological bounds, mmHg
BP_DIASTOLIC_RANGE = (40, 150)  # plausible physiological bounds, mmHg
BP_SPECIAL = {994, 995, 996}    # Refused / Technical problem / Other


def valid_bp(series, lo, hi):
    """Return series with special codes and out-of-range values set to NaN."""
    s = series.copy()
    s = s.where(~s.isin(BP_SPECIAL), np.nan)
    s = s.where((s >= lo) & (s <= hi), np.nan)
    return s


def average_valid_bp(df, cols, lo, hi):
    valids = [valid_bp(df[c], lo, hi) for c in cols]
    stacked = pd.concat(valids, axis=1)
    return stacked.mean(axis=1, skipna=True), stacked.notna().sum(axis=1)


def decode_duration_to_years(series, never_code=995):
    """Decode DHS base+offset duration encoding: 1xx=weeks, 2xx=months, 3xx=years,
    xx99 = 'number missing' sentinel -> NaN, never_code -> 0 years."""
    s = series.copy()
    out = pd.Series(np.nan, index=s.index, dtype="float64")

    is_never = s == never_code
    out[is_never] = 0.0

    weeks_mask = (s >= 100) & (s < 200) & (s % 100 != 99)
    months_mask = (s >= 200) & (s < 300) & (s % 100 != 99)
    years_mask = (s >= 300) & (s < 400) & (s % 100 != 99)

    out[weeks_mask] = (s[weeks_mask] - 100) / 52.0
    out[months_mask] = (s[months_mask] - 200) / 12.0
    out[years_mask] = (s[years_mask] - 300)

    # xx99 "number missing" sentinels and anything else unrecognized -> stay NaN
    return out


def decode_bmi(series):
    """v445/mv445-style BMI*100 encoding. 9998 = flagged case -> NaN."""
    s = series.copy()
    s = s.where(s != 9998, np.nan)
    return s / 100.0


def build_women_predictors():
    df = pd.read_pickle(str(DATA_PROCESSED_DIR) + "/women_target_a.pkl")
    # BUGFIX: must also exclude pregnant women (exclusion_reason_final=="pregnant").
    # Filtering on target_a.isin([0,1]) alone is insufficient because pregnancy exclusion
    # is applied independently of Target A logic - a pregnant woman can still have a valid
    # target_a value. Correct usable population = valid target_a AND not excluded for any reason.
    usable = df[df.target_a.isin([0, 1]) & df.exclusion_reason_final.isna()].copy()

    out = pd.DataFrame(index=usable.index)
    out["caseid"] = usable["caseid"]
    out["age"] = usable["v012"]
    out["bmi"] = decode_bmi(usable["v445"])

    sys_avg, sys_n = average_valid_bp(usable, ["sb18s", "sb25s", "sb29s"], *BP_SYSTOLIC_RANGE)
    dia_avg, dia_n = average_valid_bp(usable, ["sb18d", "sb25d", "sb29d"], *BP_DIASTOLIC_RANGE)
    out["avg_systolic_bp"] = sys_avg
    out["avg_diastolic_bp"] = dia_avg
    out["n_valid_systolic_readings"] = sys_n
    out["n_valid_diastolic_readings"] = dia_n

    # tobacco (compact women's set - 12 items, verified earlier)
    tobacco_cols = ["v463a", "v463b", "v463c", "v463d", "v463f", "v463g",
                     "v463j", "v463k", "v463l", "v463x", "v463z", "v463aa"]
    for c in tobacco_cols:
        out[f"tob_{c}"] = usable[c]

    # alcohol (compact women's set)
    out["alc_drinks"] = usable["s720"]
    out["alc_frequency"] = usable["s721"]  # NaN for non-drinkers (skip pattern, not an error)
    for c in ["s722a", "s722b", "s722c", "s722d", "s722e", "s722x"]:
        out[f"alc_{c}"] = usable[c]
    out["alc_duration_years"] = decode_duration_to_years(usable["s723"], never_code=995)

    # diet (9 items)
    diet_cols = ["s731a", "s731b", "s731c", "s731d", "s731e", "s731f", "s731g", "s731h", "s731i"]
    for c in diet_cols:
        out[f"diet_{c}"] = usable[c]

    out["education"] = usable["v106"]
    out["residence"] = usable["v025"]

    # retained metadata (NOT predictors) - cluster/design + weight + audit
    out["v021_psu"] = usable["v021"]
    out["v022_stratum"] = usable["v022"]
    out["v023_domain"] = usable["v023"]
    out["v005_weight"] = usable["v005"]
    out["target_a"] = usable["target_a"]
    out["exclusion_reason_final"] = usable["exclusion_reason_final"]

    return out


def build_men_predictors():
    df = pd.read_pickle(str(DATA_PROCESSED_DIR) + "/men_target_a.pkl")
    # men have no pregnancy exclusion, but apply the same consistent filter logic
    # (equivalent to target_a.isin([0,1]) alone for men, verified below)
    usable = df[df.target_a.isin([0, 1]) & df.exclusion_reason_final.isna()].copy()

    out = pd.DataFrame(index=usable.index)
    out["caseid"] = usable["mcaseid"]
    out["age"] = usable["mv012"]
    # NO BMI for men - confirmed unavailable in NFHS men's file

    sys_avg, sys_n = average_valid_bp(usable, ["smb18s", "smb25s", "smb29s"], *BP_SYSTOLIC_RANGE)
    dia_avg, dia_n = average_valid_bp(usable, ["smb18d", "smb25d", "smb29d"], *BP_DIASTOLIC_RANGE)
    out["avg_systolic_bp"] = sys_avg
    out["avg_diastolic_bp"] = dia_avg
    out["n_valid_systolic_readings"] = sys_n
    out["n_valid_diastolic_readings"] = dia_n

    # tobacco (compact men's set - 8 usable items; v463j/k/l have no men's equivalent (documented earlier),
    # and mv463aa is dropped here: confirmed 100% missing / not administered in the men's questionnaire
    # (its own codebook label is "NA - Frequency currently smokes tobacco"), unlike women's v463aa.
    tobacco_cols = ["mv463a", "mv463b", "mv463c", "mv463d", "mv463f", "mv463g",
                     "mv463x", "mv463z"]
    for c in tobacco_cols:
        out[f"tob_{c}"] = usable[c]

    # alcohol (compact men's set)
    out["alc_drinks"] = usable["sm619"]
    out["alc_frequency"] = usable["sm620"]  # NOTE: men's coding has explicit 0="None"; women's uses NaN for non-drinkers (raw NFHS design difference, not introduced by us)
    for c in ["sm621a", "sm621b", "sm621c", "sm621d", "sm621e", "sm621x"]:
        out[f"alc_{c}"] = usable[c]
    out["alc_duration_years"] = decode_duration_to_years(usable["sm622"], never_code=995)

    # diet (9 items)
    diet_cols = ["sm630a", "sm630b", "sm630c", "sm630d", "sm630e", "sm630f", "sm630g", "sm630h", "sm630i"]
    for c in diet_cols:
        out[f"diet_{c}"] = usable[c]

    out["education"] = usable["mv106"]
    out["residence"] = usable["mv025"]

    out["mv021_psu"] = usable["mv021"]
    out["mv022_stratum"] = usable["mv022"]
    out["mv023_domain"] = usable["mv023"]
    out["mv005_weight"] = usable["mv005"]
    out["target_a"] = usable["target_a"]
    out["exclusion_reason_final"] = usable["exclusion_reason_final"]

    return out


if __name__ == "__main__":
    women = build_women_predictors()
    men = build_men_predictors()

    print("women predictors shape:", women.shape)
    print("men predictors shape:", men.shape)

    women.to_pickle(str(DATA_PROCESSED_DIR) + "/women_predictors.pkl")
    men.to_pickle(str(DATA_PROCESSED_DIR) + "/men_predictors.pkl")
    print("saved.")
