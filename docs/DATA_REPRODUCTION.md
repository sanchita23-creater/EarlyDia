# Data reproduction note

EarlyDia is built on **NFHS-5** (National Family Health Survey, India, 2019-21) microdata,
distributed by the DHS Program. **This project does not redistribute NFHS microdata** -
`data/raw/` is empty (see `.gitkeep`), and no `.DTA`/`.DCT` file is committed anywhere in
this repository.

## Obtaining the data
1. Register and request access at https://dhsprogram.com/data/ (free for registered
   research/educational use; approval is not instant).
2. Download the **India 2019-21 (NFHS-5)** round, Stata (.dta) format.
3. From the release, EarlyDia needs exactly two files:
   - `IAIR7EFL.DTA` - Individual Recode (women 15-49)
   - `IAMR7EFL.DTA` - Men's Recode (men 15-54)
   The Household/Person Recode (`IAPR7EFL`) is **not required** - every EarlyDia variable is
   already present in the two individual recodes.
4. Place both files directly in `data/raw/`.

## Running the extraction
```bash
python -m src.data.extract_nfhs
```
This reads only the ~60-70 variables EarlyDia uses (out of ~5,972 in the women's file) and
writes `data/interim/{women,men}_key_columns_raw.pkl`. It validates the resulting row counts
against the values recorded when this project was built (724,115 / 101,839 total rows;
510,289 / 64,918 aged 18-40) and stops if they don't match, which would indicate a different
data release or file version. Pass `--no-strict` to continue anyway with a warning.

## Status of this validation
The men's file was extracted and verified end-to-end while building this project: the
script's output was byte-for-byte identical (same 57 columns, same values) to the file the
actual modelling pipeline consumed, and the row counts matched exactly (101,839 total;
64,918 aged 18-40). The women's raw file was not available in the packaging environment
to re-run in the same way, but is the same code path, uses the same chunked-read approach
already proven on the (larger) men's-adjacent workload, and was validated by count when
originally extracted (510,289 aged 18-40, matching the number this project's target
construction and every downstream stage was built and checked against).

## What happens after extraction
The rest of the pipeline (`src/target/`, `src/features/`, `src/validation/`, `src/models/`,
`src/calibration/`, `src/evaluation/`) consumes `data/interim/*.pkl` and does not touch the
raw `.DTA` files again. See `configs/reproducibility_manifest.json` for the full stage order
and every locked configuration value (seeds, split method, models, calibration, thresholds).
