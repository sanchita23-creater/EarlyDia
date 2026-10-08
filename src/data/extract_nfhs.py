"""
EarlyDia - NFHS-5 raw data extraction (raw .DTA -> slim interim datasets)
=========================================================================

WHAT THIS DOES
    Reads ONLY the variables EarlyDia needs from the two NFHS-5 individual recode files
    and writes one slim dataset per sex. Everything downstream (Target A construction,
    predictor construction, modelling) starts from these two files.

REQUIRED INPUT (NOT distributed with this project - see docs/DATA_REPRODUCTION.md)
    data/raw/IAIR7EFL.DTA   NFHS-5 (India 2019-21) Individual Recode, women 15-49
                            (~724,115 rows x 5,972 variables, ~5.2 GB)
    data/raw/IAMR7EFL.DTA   NFHS-5 Men's Recode, men 15-54
                            (~101,839 rows, ~100 MB)
    The Household/Person recode (IAPR7EFL) is NOT needed: every variable EarlyDia uses is
    available in the individual recodes, and the person recode only repeats them.

OUTPUT
    data/interim/women_key_columns_raw.pkl   (69 columns, all women 15-49 rows)
    data/interim/men_key_columns_raw.pkl     (57 columns, all men 15-54 rows)
    Rows are NOT filtered here; the age 18-40 filter, pregnancy exclusion and Target A are
    applied downstream (src/target/). Values are left raw (no recoding, special codes kept).

MEMORY
    The women's file is read in chunks with `columns=` so only the needed variables are kept
    in memory (a full read needs far more RAM than a typical laptop/container has).
    pandas' pure-Python Stata reader is slower than pyreadstat (~1 min for the women's file);
    pyreadstat would also work but is not required.

USAGE
    python -m src.data.extract_nfhs                      # both sexes, default folders
    python -m src.data.extract_nfhs --sex men            # men only
    python -m src.data.extract_nfhs --raw-dir D:/nfhs --out-dir D:/nfhs_out

VALIDATION
    After extraction the script checks row counts against the values recorded when the
    project was built (total rows and rows aged 18-40). A mismatch means a different data
    release/file version; the script stops unless --no-strict is passed.

VARIABLE NOTES (verified against the NFHS-5 codebook labels; women / men names differ)
    Men's variables carry an 'm' prefix (mv012, smb74, sm627a ...) and several question
    numbers differ (diabetes: s728a women vs sm627a men). Men's BMI (mv445) does not exist.
    Women's v463j/k/l (gutkha, khaini, paan with tobacco) have no men's equivalent, and
    mv463aa is labelled "NA -" (not administered) - it is extracted for completeness but is
    100% missing and is dropped in predictor construction.
"""
import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402

from configs.config import DATA_INTERIM_DIR, DATA_RAW_DIR  # noqa: E402

WOMEN_FILE = "IAIR7EFL.DTA"
MEN_FILE = "IAMR7EFL.DTA"

# Grouped by role so the purpose of every extracted variable is documented.
WOMEN_COLUMN_GROUPS = {
    "identifiers / survey design (cluster, strata, weight)":
        ["caseid", "v001", "v002", "v003", "v005", "v021", "v022", "v023", "v024", "v025"],
    "age": ["v012", "v013"],
    "education / wealth (wealth retained for later subgroup analysis only - NOT a predictor)":
        ["v106", "v190", "v190a"],
    "pregnancy / reproductive status (population exclusion only - NOT predictors)":
        ["v213", "v214", "v215", "v226"],
    "anthropometry (v445 = BMI x100; v437/v438 weight/height kept for audit; no waist/hip)":
        ["v437", "v438", "v445"],
    "tobacco":
        ["v463a", "v463b", "v463c", "v463d", "v463f", "v463g", "v463j", "v463k", "v463l",
         "v463x", "v463z", "v463aa"],
    "alcohol":
        ["s720", "s721", "s722a", "s722b", "s722c", "s722d", "s722e", "s722x", "s723"],
    "self-reported diabetes / hypertension (s728a feeds Target A; s728b/bb NOT used)":
        ["s728a", "s728ab", "s728b", "s728bb"],
    "blood pressure, 3 readings (systolic/diastolic)":
        ["sb18s", "sb18d", "sb25s", "sb25d", "sb29s", "sb29d"],
    "blood glucose block (sb56, sb57, sb74 feed Target A; the rest is fasting/measurement context)":
        ["sb53", "sb54", "sb55", "sb56", "sb57", "sb73", "sb74"],
    "diet frequency (9 items)":
        ["s731a", "s731b", "s731c", "s731d", "s731e", "s731f", "s731g", "s731h", "s731i"],
}

MEN_COLUMN_GROUPS = {
    "identifiers / survey design (cluster, strata, weight)":
        ["mcaseid", "mv001", "mv002", "mv003", "mv005", "mv021", "mv022", "mv023", "mv024", "mv025"],
    "age": ["mv012", "mv013"],
    "education / wealth (wealth retained for later subgroup analysis only - NOT a predictor)":
        ["mv106", "mv190", "mv190a"],
    "self-reported diabetes (sm627a feeds Target A)": ["sm627a", "sm627ab"],
    "blood pressure, 3 readings (systolic/diastolic)":
        ["smb18s", "smb18d", "smb25s", "smb25d", "smb29s", "smb29d"],
    "blood glucose block (smb56, smb57, smb74 feed Target A; the rest is context)":
        ["smb53", "smb54", "smb55", "smb56", "smb57", "smb73", "smb74"],
    "tobacco (mv463aa is not administered in this survey - 100% missing)":
        ["mv463a", "mv463b", "mv463c", "mv463d", "mv463f", "mv463g", "mv463x", "mv463z", "mv463aa"],
    "alcohol": ["sm619", "sm620", "sm621a", "sm621b", "sm621c", "sm621d", "sm621e", "sm621x", "sm622"],
    "diet frequency (9 items)":
        ["sm630a", "sm630b", "sm630c", "sm630d", "sm630e", "sm630f", "sm630g", "sm630h", "sm630i"],
    # NOTE: no men's BMI (mv445 does not exist) and no pregnancy variable (mv213 is the
    # PARTNER's pregnancy status and must not be used to exclude men).
}


def _flatten(groups):
    cols, seen = [], set()
    for group in groups.values():
        for c in group:
            if c not in seen:
                seen.add(c)
                cols.append(c)
    return cols


WOMEN_COLUMNS = _flatten(WOMEN_COLUMN_GROUPS)
MEN_COLUMNS = _flatten(MEN_COLUMN_GROUPS)

SPECS = {
    "women": {"file": WOMEN_FILE, "columns": WOMEN_COLUMNS, "age_col": "v012",
              "out": "women_key_columns_raw.pkl",
              "expected_rows": 724_115, "expected_age_18_40": 510_289},
    "men": {"file": MEN_FILE, "columns": MEN_COLUMNS, "age_col": "mv012",
            "out": "men_key_columns_raw.pkl",
            "expected_rows": 101_839, "expected_age_18_40": 64_918},
}


def find_raw_file(raw_dir: Path, filename: str) -> Path:
    for candidate in (raw_dir / filename, raw_dir / filename.lower()):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"NFHS-5 raw file '{filename}' not found in {raw_dir}. NFHS microdata is not "
        f"distributed with this project: request it from the DHS Program "
        f"(https://dhsprogram.com/data/) and place the unzipped .DTA file there. "
        f"See docs/DATA_REPRODUCTION.md.")


def check_columns_exist(path: Path, columns):
    """Fail early (before the long read) if the file lacks any requested variable."""
    with pd.read_stata(path, iterator=True, convert_categoricals=False) as reader:
        available = set(reader.variable_labels().keys())
    missing = [c for c in columns if c not in available]
    if missing:
        raise KeyError(f"{path.name} is missing expected variables: {missing}")


def read_columns_chunked(path: Path, columns, chunksize: int = 20_000) -> pd.DataFrame:
    """Read only `columns` from a large .DTA in chunks (bounded memory). Raw codes are kept:
    convert_categoricals=False so special codes (e.g. 994/995/996/998) stay numeric."""
    reader = pd.read_stata(path, columns=columns, convert_categoricals=False, chunksize=chunksize)
    chunks = [chunk.copy() for chunk in reader]
    df = pd.concat(chunks, ignore_index=True)
    return df[columns]


def validate_counts(df, age_col, expected_rows, expected_age, label, strict=True):
    n_age = int(df[age_col].between(18, 40).sum())
    print(f"  [{label}] rows={len(df):,} (expected {expected_rows:,}); "
          f"age 18-40={n_age:,} (expected {expected_age:,})")
    if (len(df), n_age) != (expected_rows, expected_age):
        msg = (f"[{label}] row counts differ from the values recorded when EarlyDia was built - "
               f"different NFHS-5 file version?")
        if strict:
            raise ValueError(msg + " (use --no-strict to continue anyway)")
        print("  WARNING:", msg)


def extract(sex: str, raw_dir: Path, out_dir: Path, chunksize: int = 20_000,
            strict: bool = True, validate: bool = True) -> Path:
    spec = SPECS[sex]
    src = find_raw_file(Path(raw_dir), spec["file"])
    print(f"[{sex}] reading {len(spec['columns'])} variables from {src.name} ...")
    check_columns_exist(src, spec["columns"])
    df = read_columns_chunked(src, spec["columns"], chunksize)
    if validate:
        validate_counts(df, spec["age_col"], spec["expected_rows"], spec["expected_age_18_40"],
                        sex, strict)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / spec["out"]
    df.to_pickle(out_path)
    print(f"[{sex}] saved {df.shape[0]:,} x {df.shape[1]} -> {out_path}")
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Extract required NFHS-5 variables into slim interim datasets.")
    ap.add_argument("--raw-dir", default=str(DATA_RAW_DIR), help="folder containing the raw .DTA files")
    ap.add_argument("--out-dir", default=str(DATA_INTERIM_DIR), help="folder for the slim outputs")
    ap.add_argument("--sex", choices=["women", "men", "both"], default="both")
    ap.add_argument("--chunksize", type=int, default=20_000)
    ap.add_argument("--no-strict", action="store_true", help="warn (do not fail) on row-count mismatch")
    args = ap.parse_args(argv)
    for sex in (["women", "men"] if args.sex == "both" else [args.sex]):
        extract(sex, Path(args.raw_dir), Path(args.out_dir), args.chunksize, strict=not args.no_strict)


if __name__ == "__main__":
    main()
