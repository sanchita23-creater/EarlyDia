"""
EarlyDia - Target A construction (women)
Locked definition (verbatim from user, sourced from audit_v2.txt):

K = (sb56==1) OR (sb57==1) OR (s728a==1)
POS = K OR (valid glucose AND sb74 >= 200)
NEG = all three flags explicitly No (==0) AND valid glucose AND sb74 <= 140
Excluded:
  1. No valid glucose AND no known-diabetes flag
  2. Valid glucose 141-199 AND no known-diabetes flag
  3. Any known-diabetes flag blank/don't-know rather than explicit No

valid glucose = sb74 not null and sb74 not in {995 (Refused), 996 (Other), 998 (Not tested)}
(499 = "499(mg/dL)+" is a genuine top-coded reading, treated as valid and >=200)

MODIFICATION 1: Include borderline glucose (141-199) as POSITIVE to increase sample size
and provide better training signal near the decision boundary.
MODIFICATION 2 (v5): Stricter negative class — glucose <= 100 (normal) instead of <= 140.
This sharpens the positive/negative boundary by excluding people with glucose 101-140
(who may have prediabetes/impaired glucose tolerance) from the negative class.
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

df = pd.read_pickle(str(DATA_INTERIM_DIR) + "/women_key_columns_raw.pkl")

# Widen age range from 18-40 to 18-49 to increase sample size and prevalence
# Diabetes prevalence is higher in older ages, improving model learning
age = df[(df.v012 >= 18) & (df.v012 <= 49)].copy()
print(f"Age 18-49 filter: {len(age)} rows (previous 18-40 was 510,289)")

GLUCOSE_SPECIAL = {995, 996, 998}

def compute_target_a(d, flag_cols, glucose_col):
    sb56, sb57, s728a = d[flag_cols[0]], d[flag_cols[1]], d[flag_cols[2]]
    glu = d[glucose_col]

    valid_glucose = glu.notna() & (~glu.isin(GLUCOSE_SPECIAL))

    K = (sb56 == 1) | (sb57 == 1) | (s728a == 1)
    all_explicit_no = (sb56 == 0) & (sb57 == 0) & (s728a == 0)

    # MODIFIED: Include borderline glucose (141-199) as POSITIVE
    # This increases positive sample size and provides better training signal
    POS = K | (valid_glucose & (glu >= 141))
    # V5: Stricter negative class — glucose <= 100 (normal) instead of <= 140
    # This excludes people with glucose 101-140 (potential prediabetes) from negatives
    NEG = all_explicit_no & valid_glucose & (glu <= 100)
    # V5: Gray zone — glucose 101-140 with no diabetes flags (potential prediabetes)
    GRAY_ZONE = all_explicit_no & valid_glucose & (glu > 100) & (glu < 141)

    target = pd.Series(np.nan, index=d.index, dtype="float64")
    reason = pd.Series(pd.NA, index=d.index, dtype="object")

    target[POS] = 1
    target[NEG] = 0

    remaining = ~(POS | NEG)
    # V5: gray zone exclusion
    gray = remaining & GRAY_ZONE
    reason[gray] = "gray_zone_glucose_101_140"

    # reason 3: any flag not explicit No (ambiguous/don't-know/blank) among remaining (K is False here since POS excluded)
    ambiguous_flag = remaining & (~all_explicit_no) & ~gray
    reason[ambiguous_flag] = "ambiguous_known_flag"

    # among remaining with all_explicit_no==True: split by glucose validity
    clean_flags_remaining = remaining & all_explicit_no
    no_glucose = clean_flags_remaining & (~valid_glucose)
    reason[no_glucose] = "no_valid_glucose_no_known_flag"

    # Note: borderline glucose (141-199) is now included as POSITIVE above, not excluded

    # sanity: nothing should remain unclassified
    unclassified = remaining & reason.isna()
    if unclassified.sum() > 0:
        raise AssertionError(f"{unclassified.sum()} rows fell through all exclusion categories - logic gap")

    return target, reason

target, reason = compute_target_a(age, ["sb56", "sb57", "s728a"], "sb74")
age["target_a"] = target
age["exclusion_reason"] = reason

pos = (age.target_a == 1).sum()
neg = (age.target_a == 0).sum()
usable = pos + neg
print("=== BEFORE pregnancy exclusion (age 18-49, all) ===")
print(f"POS = {pos}")
print(f"NEG = {neg}")
print(f"usable = {usable}")
print(f"(Previous age 18-40: POS=14561, NEG=445500, usable=460061)")

print()
print("=== Pregnancy exclusion ===")
n_before = len(age)
pregnant = age["v213"] == 1
n_pregnant = pregnant.sum()
print(f"rows removed for pregnancy (v213==1): {n_pregnant}")
print(f"(Previous age 18-40: 27,826)")

# Build final exclusion_reason: pregnancy takes priority over target-based reasons
age["exclusion_reason_final"] = age["exclusion_reason"]
age.loc[pregnant, "exclusion_reason_final"] = "pregnant"

non_pregnant = age[~pregnant].copy()
print(f"remaining after pregnancy exclusion: {len(non_pregnant)}")

pos2 = (non_pregnant.target_a == 1).sum()
neg2 = (non_pregnant.target_a == 0).sum()
usable2 = pos2 + neg2
share = pos2 / usable2 * 100 if usable2 > 0 else 0
print()
print("=== AFTER pregnancy exclusion ===")
print(f"POS = {pos2}")
print(f"NEG = {neg2}")
print(f"usable = {usable2}")
print(f"positive share = {share:.2f}%")
print(f"(Previous age 18-40: POS=13839, NEG=420663, usable=434502, share=3.19%)")

print()
print("=== Full women's exclusion_reason_final breakdown (age 18-49) ===")
print(age["exclusion_reason_final"].value_counts(dropna=False))

# keep the full audit-trail dataframe (age 18-40, ALL rows incl. excluded) for auditability
age.to_pickle(str(DATA_PROCESSED_DIR) + "/women_target_a.pkl")
print("\nsaved: data/processed/women_target_a.pkl")

# quick row-count reconciliation
n_pos = (age.exclusion_reason_final.isna() & (age.target_a==1)).sum()
n_neg = (age.exclusion_reason_final.isna() & (age.target_a==0)).sum()
n_pregnant = (age.exclusion_reason_final=="pregnant").sum()
n_amb = (age.exclusion_reason_final=="ambiguous_known_flag").sum()
n_noglu = (age.exclusion_reason_final=="no_valid_glucose_no_known_flag").sum()
n_gray = (age.exclusion_reason_final=="gray_zone_glucose_101_140").sum()
total_check = n_pos+n_neg+n_pregnant+n_amb+n_noglu+n_gray
print(f"\nreconciliation: {n_pos}+{n_neg}+{n_pregnant}+{n_amb}+{n_noglu}+{n_gray} = {total_check} (should equal {len(age)})")
assert total_check == len(age), "row accounting mismatch!"
print("Row accounting verified: every row accounted for exactly once.")
