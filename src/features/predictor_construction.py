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

    # Physical activity (occupational activity level, if available)
    out["physical_activity"] = usable.get("v481", pd.Series(np.nan, index=usable.index))

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

    # Fasting/measurement context variables (sb53, sb54, sb55, sb73)
    # These are in the blood glucose block and likely encode fasting status/timing
    # Added as predictors to allow model to learn their importance
    out["glucose_context_sb53"] = usable["sb53"]
    out["glucose_context_sb54"] = usable["sb54"]
    out["glucose_context_sb55"] = usable["sb55"]
    out["glucose_context_sb73"] = usable["sb73"]

    # Self-reported hypertension (s728b) - added as predictor
    # Hypertension is a comorbidity associated with diabetes risk, not a proxy for glucose target
    out["hypertension_self_report"] = usable["s728b"]

    # V4: BP medication (sb21 = currently taking BP-lowering medicine)
    out["bp_medication"] = usable["sb21"]

    # V4: comorbidities (chronic conditions, NOT part of Target A)
    comorbidity_map_w = {
        "comorbidity_respiratory": "s728c",
        "comorbidity_thyroid": "s728d",
        "comorbidity_heart": "s728e",
        "comorbidity_cancer": "s728f",
        "comorbidity_kidney": "s728g",
    }
    for feat, col in comorbidity_map_w.items():
        out[feat] = usable[col]
    out["n_comorbidities"] = sum(usable[c] == 1 for c in comorbidity_map_w.values())

    # V4: waist/hip circumference (from PR file merge)
    out["waist_circumference"] = usable.get("sh305", pd.Series(np.nan, index=usable.index))
    out["hip_circumference"] = usable.get("sh306", pd.Series(np.nan, index=usable.index))
    out["waist_hip_ratio"] = out["waist_circumference"] / out["hip_circumference"].replace(0, np.nan)

    # V6: Waist-to-height ratio (non-invasive, simple anthropometric measure)
    # Height v437 is in mm (NFHS-5 encoding), convert to cm for ratio calculation
    height_cm = usable.get("v437", pd.Series(np.nan, index=usable.index)) / 10.0
    out["waist_height_ratio"] = out["waist_circumference"] / height_cm.replace(0, np.nan)

    # V6: Tobacco chewing indicator (non-invasive, self-reported)
    # v463j=gutkha, v463k=khaini, v463l=paan with tobacco (all binary 0/1)
    chewing_cols = ["v463j", "v463k", "v463l"]
    out["tobacco_chewing"] = (usable[chewing_cols].max(axis=1) == 1).astype(int)

    # V6: Age categories (capture non-linear age effects)
    out["age_18_24"] = ((out["age"] >= 18) & (out["age"] <= 24)).astype(int)
    out["age_25_34"] = ((out["age"] >= 25) & (out["age"] <= 34)).astype(int)
    out["age_35_40"] = ((out["age"] >= 35) & (out["age"] <= 40)).astype(int)

    # Interaction features (clinically meaningful for diabetes risk)
    out["age_x_bmi"] = out["age"] * out["bmi"]
    out["bmi_x_systolic"] = out["bmi"] * out["avg_systolic_bp"]
    out["bmi_x_diastolic"] = out["bmi"] * out["avg_diastolic_bp"]
    out["age_x_systolic"] = out["age"] * out["avg_systolic_bp"]
    out["age_x_diastolic"] = out["age"] * out["avg_diastolic_bp"]

    # V5: BP aggregates (mean arterial pressure, pulse pressure)
    out["mean_arterial_pressure"] = (2 * out["avg_diastolic_bp"] + out["avg_systolic_bp"]) / 3
    out["pulse_pressure"] = out["avg_systolic_bp"] - out["avg_diastolic_bp"]

    # V5: waist interactions (waist × age, waist × bp_medication, comorbidity × waist)
    out["waist_x_age"] = out["waist_circumference"] * out["age"]
    out["waist_x_bp_med"] = out["waist_circumference"] * out["bp_medication"]
    out["comorbidity_x_waist"] = out["n_comorbidities"] * out["waist_circumference"]

    # V5: Diet diversity score (count of food groups consumed at least once)
    # Diet items are coded 0=never, 1=once, 2=few times, 3=often — treat >=1 as consumed
    diet_cols_w = ["s731a", "s731b", "s731c", "s731d", "s731e", "s731f", "s731g", "s731h", "s731i"]
    out["diet_diversity_score"] = sum((usable[c] >= 1).astype(int) for c in diet_cols_w)

    # V5: BMI category indicators (clinical thresholds)
    out["bmi_underweight"] = (out["bmi"] < 18.5).astype(int)
    out["bmi_normal"] = ((out["bmi"] >= 18.5) & (out["bmi"] < 25)).astype(int)
    out["bmi_overweight"] = ((out["bmi"] >= 25) & (out["bmi"] < 30)).astype(int)
    out["bmi_obese"] = (out["bmi"] >= 30).astype(int)

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

    # Physical activity (occupational activity level, if available)
    out["physical_activity"] = usable.get("mv481", pd.Series(np.nan, index=usable.index))

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

    # Fasting/measurement context variables (smb53, smb54, smb55, smb73)
    # These are in the blood glucose block and likely encode fasting status/timing
    # Added as predictors to allow model to learn their importance
    out["glucose_context_smb53"] = usable["smb53"]
    out["glucose_context_smb54"] = usable["smb54"]
    out["glucose_context_smb55"] = usable["smb55"]
    out["glucose_context_smb73"] = usable["smb73"]

    # Self-reported hypertension (sm627b) - added as predictor
    # Hypertension is a comorbidity associated with diabetes risk, not a proxy for glucose target
    out["hypertension_self_report"] = usable["sm627b"]

    # V4: BP medication (smb21 = currently taking BP-lowering medicine)
    out["bp_medication"] = usable["smb21"]

    # V4: comorbidities (chronic conditions, NOT part of Target A)
    comorbidity_map_m = {
        "comorbidity_respiratory": "sm627c",
        "comorbidity_thyroid": "sm627d",
        "comorbidity_heart": "sm627e",
        "comorbidity_cancer": "sm627f",
        "comorbidity_kidney": "sm627g",
    }
    for feat, col in comorbidity_map_m.items():
        out[feat] = usable[col]
    out["n_comorbidities"] = sum(usable[c] == 1 for c in comorbidity_map_m.values())

    # V4: waist/hip circumference (directly in men's MR file)
    out["waist_circumference"] = usable["smb305"]
    out["hip_circumference"] = usable["smb306"]
    out["waist_hip_ratio"] = out["waist_circumference"] / out["hip_circumference"].replace(0, np.nan)

    # Interaction features (clinically meaningful for diabetes risk)
    # Men have no BMI, so only age × BP interactions
    out["age_x_systolic"] = out["age"] * out["avg_systolic_bp"]
    out["age_x_diastolic"] = out["age"] * out["avg_diastolic_bp"]

    # V5: BP aggregates (mean arterial pressure, pulse pressure)
    out["mean_arterial_pressure"] = (2 * out["avg_diastolic_bp"] + out["avg_systolic_bp"]) / 3
    out["pulse_pressure"] = out["avg_systolic_bp"] - out["avg_diastolic_bp"]

    # V5: waist interactions (waist × age, waist × bp_medication, comorbidity × waist)
    out["waist_x_age"] = out["waist_circumference"] * out["age"]
    out["waist_x_bp_med"] = out["waist_circumference"] * out["bp_medication"]
    out["comorbidity_x_waist"] = out["n_comorbidities"] * out["waist_circumference"]

    # V5: Diet diversity score (count of food groups consumed at least once)
    diet_cols_m = ["sm630a", "sm630b", "sm630c", "sm630d", "sm630e", "sm630f", "sm630g", "sm630h", "sm630i"]
    out["diet_diversity_score"] = sum((usable[c] >= 1).astype(int) for c in diet_cols_m)

    # V6: Age categories (capture non-linear age effects)
    out["age_18_24"] = ((out["age"] >= 18) & (out["age"] <= 24)).astype(int)
    out["age_25_34"] = ((out["age"] >= 25) & (out["age"] <= 34)).astype(int)
    out["age_35_40"] = ((out["age"] >= 35) & (out["age"] <= 40)).astype(int)

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
