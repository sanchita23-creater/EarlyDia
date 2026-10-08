import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from configs.config import (DATA_RAW_DIR, DATA_INTERIM_DIR, DATA_PROCESSED_DIR,
                            EXPLAINABILITY_REPORTS_DIR)
import pandas as pd
import numpy as np

TARGET_VARS_W = {"sb56", "sb57", "s728a", "sb74"}
TARGET_VARS_M = {"smb56", "smb57", "sm627a", "smb74"}

women = pd.read_pickle(str(DATA_PROCESSED_DIR) + "/women_predictors.pkl")
men = pd.read_pickle(str(DATA_PROCESSED_DIR) + "/men_predictors.pkl")

print("=" * 70)
print("CHECK 1: Row counts")
print("=" * 70)
print(f"women: {len(women)} (usable target_a rows)")
print(f"men:   {len(men)}")

print()
print("=" * 70)
print("CHECK 2: Missingness for every final predictor")
print("=" * 70)
pred_cols_w = [c for c in women.columns if c not in
               ["caseid", "target_a", "exclusion_reason_final",
                "v021_psu", "v022_stratum", "v023_domain", "v005_weight"]]
pred_cols_m = [c for c in men.columns if c not in
               ["caseid", "target_a", "exclusion_reason_final",
                "mv021_psu", "mv022_stratum", "mv023_domain", "mv005_weight"]]

print("--- WOMEN ---")
miss_w = women[pred_cols_w].isna().sum().sort_values(ascending=False)
miss_w_pct = (miss_w / len(women) * 100).round(2)
for c in miss_w.index:
    print(f"  {c:30s} missing={miss_w[c]:>7d}  ({miss_w_pct[c]}%)")

print("--- MEN ---")
miss_m = men[pred_cols_m].isna().sum().sort_values(ascending=False)
miss_m_pct = (miss_m / len(men) * 100).round(2)
for c in miss_m.index:
    print(f"  {c:30s} missing={miss_m[c]:>7d}  ({miss_m_pct[c]}%)")

print()
print("=" * 70)
print("CHECK 3: Impossible / out-of-range values")
print("=" * 70)
print("--- women age range ---")
print(women.age.describe()[["min", "max"]])
print("--- women bmi range (post-cleaning) ---")
print(women.bmi.describe()[["min", "max"]])
print("  BMI < 10 or > 60 (implausible):", ((women.bmi < 10) | (women.bmi > 60)).sum())
print("--- women BP range (post-cleaning, should be within physiological bounds) ---")
print("  systolic:", women.avg_systolic_bp.min(), "-", women.avg_systolic_bp.max())
print("  diastolic:", women.avg_diastolic_bp.min(), "-", women.avg_diastolic_bp.max())
print("--- men age / BP range ---")
print(men.age.describe()[["min", "max"]])
print("  systolic:", men.avg_systolic_bp.min(), "-", men.avg_systolic_bp.max())
print("  diastolic:", men.avg_diastolic_bp.min(), "-", men.avg_diastolic_bp.max())
print("--- alcohol duration years (should be >=0 and plausible <= ~40 given age<=40) ---")
print("  women max:", women.alc_duration_years.max(), "| implausible (> age):",
      (women.alc_duration_years > women.age).sum())
print("  men max:", men.alc_duration_years.max(), "| implausible (> age):",
      (men.alc_duration_years > men.age).sum())

print()
print("=" * 70)
print("CHECK 4: BP special-code handling verification")
print("=" * 70)
print("women n_valid_systolic_readings distribution:")
print(women.n_valid_systolic_readings.value_counts().sort_index())
print("men n_valid_systolic_readings distribution:")
print(men.n_valid_systolic_readings.value_counts().sort_index())
print("Rows with ZERO valid systolic readings (avg will be NaN) - women:",
      (women.n_valid_systolic_readings == 0).sum(), "| men:", (men.n_valid_systolic_readings == 0).sum())

print()
print("=" * 70)
print("CHECK 5: BMI special-code handling verification (women only)")
print("=" * 70)
raw = pd.read_pickle(str(DATA_PROCESSED_DIR) + "/women_target_a.pkl")
raw_usable = raw[raw.target_a.isin([0, 1])]
print("raw v445 == 9998 (flagged) count:", (raw_usable.v445 == 9998).sum())
print("resulting NaN bmi count in predictors:", women.bmi.isna().sum())
print("raw v445 null count:", raw_usable.v445.isna().sum())
print("check: flagged(9998) + raw_null == final NaN bmi?",
      (raw_usable.v445 == 9998).sum() + raw_usable.v445.isna().sum() == women.bmi.isna().sum())

print()
print("=" * 70)
print("CHECK 6: Tobacco/alcohol/diet recoding spot-check against codebook")
print("=" * 70)
print("women tob_v463aa (freq) categories:", sorted(women.tob_v463aa.dropna().unique()))
print("  expected {0,1,2} = Does not smoke/Every day/Some days")
print("men tob_mv463aa: DROPPED (confirmed 100% not administered - see predictor_construction.py note)")
print("women diet_s731a categories:", sorted(women.diet_s731a.dropna().unique()))
print("  expected {0,1,2,3} = Never/Daily/Weekly/Occasionally")
print("men diet_sm630a categories:", sorted(men.diet_sm630a.dropna().unique()))
print("women alc_frequency categories:", sorted(women.alc_frequency.dropna().unique()), "(NaN=non-drinker via skip pattern)")
print("men alc_frequency categories:", sorted(men.alc_frequency.dropna().unique()), "(0=None is explicit in men's coding)")

print()
print("=" * 70)
print("CHECK 7: Categorical variables have expected categories")
print("=" * 70)
print("women education (v106) categories:", sorted(women.education.dropna().unique()), "expected {0,1,2,3}")
print("men education categories:", sorted(men.education.dropna().unique()), "expected {0,1,2,3}")
print("women residence (v025) categories:", sorted(women.residence.dropna().unique()), "expected {1,2}")
print("men residence categories:", sorted(men.residence.dropna().unique()), "expected {1,2}")

print()
print("=" * 70)
print("CHECK 8: LEAKAGE ASSERTION - target-construction vars must NOT be in X")
print("=" * 70)
leak_w = TARGET_VARS_W & set(women.columns)
leak_m = TARGET_VARS_M & set(men.columns)
print("women leaked target-construction columns:", leak_w if leak_w else "NONE - PASS")
print("men leaked target-construction columns:", leak_m if leak_m else "NONE - PASS")
assert not leak_w, f"LEAKAGE DETECTED in women predictors: {leak_w}"
assert not leak_m, f"LEAKAGE DETECTED in men predictors: {leak_m}"
print("LEAKAGE ASSERTION PASSED for both women and men.")

print()
print("=" * 70)
print("CHECK 9: Cluster variables available for grouped splitting")
print("=" * 70)
print("women cluster cols present:", [c for c in women.columns if c in ["v021_psu","v022_stratum","v023_domain"]])
print("  v021_psu nulls:", women.v021_psu.isna().sum(), "| unique clusters:", women.v021_psu.nunique())
print("men cluster cols present:", [c for c in men.columns if c in ["mv021_psu","mv022_stratum","mv023_domain"]])
print("  mv021_psu nulls:", men.mv021_psu.isna().sum(), "| unique clusters:", men.mv021_psu.nunique())

print()
print("=" * 70)
print("CHECK 10: No silent row drops - confirm counts match target_a usable counts exactly")
print("=" * 70)
w_target = pd.read_pickle(str(DATA_PROCESSED_DIR) + "/women_target_a.pkl")
m_target = pd.read_pickle(str(DATA_PROCESSED_DIR) + "/men_target_a.pkl")
print("women: predictors rows =", len(women), " target_a usable rows =", w_target.target_a.isin([0,1]).sum(),
      " MATCH:", len(women) == w_target.target_a.isin([0,1]).sum())
print("men:   predictors rows =", len(men), " target_a usable rows =", m_target.target_a.isin([0,1]).sum(),
      " MATCH:", len(men) == m_target.target_a.isin([0,1]).sum())

print()
print("Forbidden-variable check (waist/hip, men's BMI, hypertension self-report, wealth-in-X):")
forbidden_w = {"v437", "v438", "s728b", "s728bb", "v190", "v190a", "smb305", "smb306"}
forbidden_m = {"mv445", "sm627b", "sm627bb", "mv190", "mv190a"}
print("women forbidden vars present in X:", forbidden_w & set(women.columns) or "NONE - PASS")
print("men forbidden vars present in X:", forbidden_m & set(men.columns) or "NONE - PASS")
