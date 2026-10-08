"""
EarlyDia - Target A construction (men)
Identical Target A logic to women's, using verified men's-file equivalents:
  sb56 -> smb56, sb57 -> smb57, s728a -> sm627a, sb74 -> smb74
Population: age 18-40. NO pregnancy exclusion (mv213 = partner's pregnancy status,
explicitly NOT used per locked instruction).
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

df = pd.read_pickle(str(DATA_INTERIM_DIR) + "/men_key_columns_raw.pkl")
age = df[(df.mv012 >= 18) & (df.mv012 <= 40)].copy()
assert len(age) == 64918, f"age filter mismatch: got {len(age)}"

GLUCOSE_SPECIAL = {995, 996, 998}

def compute_target_a(d, flag_cols, glucose_col):
    f1, f2, f3 = d[flag_cols[0]], d[flag_cols[1]], d[flag_cols[2]]
    glu = d[glucose_col]
    valid_glucose = glu.notna() & (~glu.isin(GLUCOSE_SPECIAL))
    K = (f1 == 1) | (f2 == 1) | (f3 == 1)
    all_explicit_no = (f1 == 0) & (f2 == 0) & (f3 == 0)
    POS = K | (valid_glucose & (glu >= 200))
    NEG = all_explicit_no & valid_glucose & (glu <= 140)

    target = pd.Series(np.nan, index=d.index, dtype="float64")
    reason = pd.Series(pd.NA, index=d.index, dtype="object")
    target[POS] = 1
    target[NEG] = 0
    remaining = ~(POS | NEG)
    ambiguous_flag = remaining & (~all_explicit_no)
    reason[ambiguous_flag] = "ambiguous_known_flag"
    clean_flags_remaining = remaining & all_explicit_no
    no_glucose = clean_flags_remaining & (~valid_glucose)
    reason[no_glucose] = "no_valid_glucose_no_known_flag"
    borderline = clean_flags_remaining & valid_glucose & (glu >= 141) & (glu <= 199)
    reason[borderline] = "borderline_glucose_no_known_flag"
    unclassified = remaining & reason.isna()
    if unclassified.sum() > 0:
        raise AssertionError(f"{unclassified.sum()} rows fell through all exclusion categories")
    return target, reason

target, reason = compute_target_a(age, ["smb56", "smb57", "sm627a"], "smb74")
age["target_a"] = target
age["exclusion_reason_final"] = reason
# no pregnancy exclusion for men - final reason IS the target-based reason (or none)

pos = (age.target_a == 1).sum()
neg = (age.target_a == 0).sum()
usable = pos + neg
print("=== MEN: age 18-40, Target A (no pregnancy exclusion applies) ===")
print(f"N age 18-40: {len(age)}")
print(f"POS = {pos}")
print(f"NEG = {neg}")
print(f"usable = {usable}")
print(f"positive share = {pos/usable*100:.2f}%")
print()
print("exclusion_reason breakdown (excluded rows only):")
print(age.loc[age.target_a.isna(), "exclusion_reason_final"].value_counts())

age.to_pickle(str(DATA_PROCESSED_DIR) + "/men_target_a.pkl")
print("\nsaved: data/processed/men_target_a.pkl")
