"""Guards the LOCKED production artifacts: any change to the saved models (retraining,
recalibration, threshold edits) changes these SHA-256 hashes and fails this test."""
import hashlib
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"
LOCKED_SHA256 = {
    "women_production_pipeline.pkl": "b933f0880472325addbeccc421b36050cdcb692d2e4682c3e058ef5054011445",
    "men_production_pipeline.pkl": "bf003571b963a831f76aee637b29df9bc557595feda64297ed8dea5b2d41115c",
    "production_pipeline_manifest.json": "a728f45dbaaf96628e2cbceb816c4770f2f135e9cb081f2978c9ce0e305eef0d",
}


def test_production_artifacts_unchanged():
    for name, expected in LOCKED_SHA256.items():
        digest = hashlib.sha256((MODELS / name).read_bytes()).hexdigest()
        assert digest == expected, f"{name} was modified"


if __name__ == "__main__":
    test_production_artifacts_unchanged()
    print("production artifacts match their locked hashes")
