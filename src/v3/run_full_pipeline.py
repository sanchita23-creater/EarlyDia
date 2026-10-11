"""
EarlyDia v3 - Full Pipeline Re-execution Script
================================================

This script re-runs the entire EarlyDia pipeline with v3 enhancements:
- Widened age ranges (18-49 women, 18-54 men)
- Borderline glucose (141-199) included as positive
- Physical activity variables
- Hypertension self-report
- Fasting context variables
- Interaction features

REQUIREMENTS:
    - Raw NFHS-5 data files must be in data/raw/:
        * IAIR7EFL.DTA (women)
        * IAMR7EFL.DTA (men)
    - See docs/DATA_REPRODUCTION.md for data access instructions

USAGE:
    python -m src.v3.run_full_pipeline

This will:
    1. Extract variables from raw NFHS-5 files
    2. Construct targets with new age ranges and definitions
    3. Build predictors with new features
    4. Train/test split
    5. Run baseline models
    6. Generate evaluation reports
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main():
    print("=" * 80)
    print("EarlyDia v3 - Full Pipeline Re-execution")
    print("=" * 80)
    print()

    # Step 1: Extract from raw NFHS-5
    print("[1/6] Extracting variables from raw NFHS-5 files...")
    print("-" * 80)
    from src.data.extract_nfhs import main as extract_main
    try:
        extract_main(["--sex", "both", "--no-strict"])
        print("✓ Extraction complete")
    except FileNotFoundError as e:
        print(f"✗ ERROR: {e}")
        print("\nRaw NFHS-5 data files not found. Please obtain from DHS Program:")
        print("  https://dhsprogram.com/data/")
        print("See docs/DATA_REPRODUCTION.md for instructions.")
        return False
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        print("\nNote: If physical activity variables (v481/mv481) don't exist in NFHS-5,")
        print("you'll need to remove them from src/data/extract_nfhs.py and retry.")
        return False
    print()

    # Step 2: Construct targets
    print("[2/6] Constructing targets with v3 definition...")
    print("-" * 80)
    print("Running women's target construction...")
    try:
        exec(open("src/target/women_target_construction.py").read())
        print("✓ Women's target complete")
    except Exception as e:
        print(f"✗ Women's target failed: {e}")
        return False

    print("\nRunning men's target construction...")
    try:
        exec(open("src/target/men_target_construction.py").read())
        print("✓ Men's target complete")
    except Exception as e:
        print(f"✗ Men's target failed: {e}")
        return False
    print()

    # Step 3: Build predictors
    print("[3/6] Building predictors with v3 features...")
    print("-" * 80)
    try:
        from src.features.predictor_construction import build_women_predictors, build_men_predictors
        from configs.config import DATA_PROCESSED_DIR

        print("Building women's predictors...")
        women = build_women_predictors()
        print(f"  Shape: {women.shape}")
        print(f"  Features: {women.shape[1] - 6}")  # exclude metadata columns

        print("Building men's predictors...")
        men = build_men_predictors()
        print(f"  Shape: {men.shape}")
        print(f"  Features: {men.shape[1] - 6}")

        women.to_pickle(str(DATA_PROCESSED_DIR) + "/women_predictors.pkl")
        men.to_pickle(str(DATA_PROCESSED_DIR) + "/men_predictors.pkl")
        print("✓ Predictors saved")
    except Exception as e:
        print(f"✗ Predictor construction failed: {e}")
        return False
    print()

    # Step 4: Train/test split
    print("[4/6] Creating train/test split...")
    print("-" * 80)
    try:
        from src.data.train_test_split import main as split_main
        split_main()
        print("✓ Train/test split complete")
    except Exception as e:
        print(f"✗ Train/test split failed: {e}")
        return False
    print()

    # Step 5: Run baseline models
    print("[5/6] Running baseline models...")
    print("-" * 80)
    print("Note: This will take several hours for full cross-validation.")
    print("You can also run models individually:")
    print("  python -m src.v2.run_v2 run --sex women")
    print("  python -m src.v2.run_v2 run --sex men")
    print("  python -m src.v2.benchmark_alternatives run --sex women --group all")
    print("  python -m src.v2.benchmark_alternatives run --sex men --group all")
    print()
    response = input("Run full baseline now? (y/n): ")
    if response.lower() == 'y':
        try:
            from src.models.grouped_cv_baseline import main as baseline_main
            baseline_main()
            print("✓ Baseline models complete")
        except Exception as e:
            print(f"✗ Baseline models failed: {e}")
            return False
    else:
        print("Skipping baseline models. Run manually when ready.")
    print()

    # Step 6: Summary
    print("[6/6] Pipeline complete!")
    print("=" * 80)
    print()
    print("V3 ENHANCEMENTS APPLIED:")
    print("  ✓ Age range: 18-49 (women), 18-54 (men)")
    print("  ✓ Target: Glucose ≥141 mg/dL = positive (includes prediabetes)")
    print("  ✓ Physical activity: v481/mv481")
    print("  ✓ Hypertension self-report: s728b/sm627b")
    print("  ✓ Fasting context: sb53-55/73, smb53-55/73")
    print("  ✓ Interaction features: age×BMI, BMI×BP, age×BP")
    print()
    print("NEXT STEPS:")
    print("  1. Check model performance:")
    print("     python -m src.v2.run_v2 summary --sex women")
    print("     python -m src.v2.run_v2 summary --sex men")
    print()
    print("  2. Compare with v2 baseline:")
    print("     python -m src.v2.benchmark_alternatives compare_v2 --sex women")
    print("     python -m src.v2.benchmark_alternatives compare_v2 --sex men")
    print()
    print("  3. Review V3_CHANGES.md for detailed documentation")
    print()
    print("=" * 80)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
