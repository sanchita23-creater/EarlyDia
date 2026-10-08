"""
src/predict.py
--------------
Reusable prediction script for EarlyDia: Non-Invasive Early Diabetes Risk Screening.
Loads the serialized end-to-end model pipeline (`models/selected_model.joblib`)
and demonstrates risk score generation on sample user inputs.

IMPORTANT CORRECTIONS & IMPLEMENTATION RULES:
1. BMI Calculation:
   Users often do not know their BMI. In Phase 2, the frontend must collect Height and Weight,
   and automatically compute BMI. Height and Weight themselves must NOT be added as model features.
2. ML Risk Score / Screening Indication:
   The output of predict_proba is an ML-generated risk score / screening indication,
   NOT a medically confirmed probability of having diabetes.
3. Prototype Thresholds:
   The 30% and 60% risk boundaries are preliminary prototype screening thresholds,
   NOT clinical diagnostic thresholds.
4. Dataset Encodings:
   Provides helper mappings for Age, BMI, and discrete scales for Phase 2 integration.
"""

import os
import sys

# Set thread limits to prevent OpenBLAS memory buffer exhaustion
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import joblib
import pandas as pd
from feature_selection import SELECTED_FEATURES


def calculate_bmi(height_cm: float, weight_kg: float) -> float:
    """
    Computes Body Mass Index (BMI) from user-entered Height and Weight.
    
    Formula: BMI = weight_kg / (height_m ** 2)
    
    NOTE FOR PHASE 2 FRONTEND:
    The frontend should collect Height and Weight from the user, run this calculation,
    and pass only the resulting BMI to the model pipeline.
    Height and Weight themselves must NOT be passed as features.
    """
    if height_cm <= 0 or weight_kg <= 0:
        raise ValueError("Height and weight must be positive numbers.")
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m ** 2)
    return round(bmi, 1)


def map_age_to_bracket(age_in_years: int) -> int:
    """
    Maps an age in years to the 13-bracket CDC BRFSS encoding scale.
    
    1: 18-24, 2: 25-29, 3: 30-34, 4: 35-39, 5: 40-44, 6: 45-49,
    7: 50-54, 8: 55-59, 9: 60-64, 10: 65-69, 11: 70-74, 12: 75-79, 13: 80+
    """
    if age_in_years < 18:
        raise ValueError("EarlyDia is calibrated for adult screening (18 years or older).")
    elif age_in_years <= 24:
        return 1
    elif age_in_years <= 29:
        return 2
    elif age_in_years <= 34:
        return 3
    elif age_in_years <= 39:
        return 4
    elif age_in_years <= 44:
        return 5
    elif age_in_years <= 49:
        return 6
    elif age_in_years <= 54:
        return 7
    elif age_in_years <= 59:
        return 8
    elif age_in_years <= 64:
        return 9
    elif age_in_years <= 69:
        return 10
    elif age_in_years <= 74:
        return 11
    elif age_in_years <= 79:
        return 12
    else:
        return 13


def load_model(model_path: str = "models/selected_model.joblib"):
    """
    Loads the serialized end-to-end prediction pipeline.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Run train_model.py and evaluate_model.py first.")
    model = joblib.load(model_path)
    return model


def map_score_to_risk_tier(risk_score: float) -> tuple[str, str]:
    """
    Maps model predicted score to a non-diagnostic risk screening tier.
    
    PROTOTYPE THRESHOLD NOTICE:
    The 30% and 60% thresholds are prototype screening thresholds developed for exploratory
    educational awareness and are NOT clinical diagnostic thresholds.
    
    Returns:
        tuple[str, str]: Risk tier label and health advisory message.
    """
    pct = risk_score * 100
    if pct < 30.0:
        tier = "Low Risk Indication"
        advisory = (
            "Screening indicates a low risk pattern for prediabetes or diabetes based on self-reported inputs. "
            "Maintaining regular physical activity and a balanced diet continues to support long-term metabolic health."
        )
    elif pct < 60.0:
        tier = "Moderate / Elevated Risk Indication"
        advisory = (
            "Screening indicates moderate or elevated lifestyle and physical risk factors. "
            "Consider discussing lifestyle modifications and scheduling a routine health checkup with a qualified healthcare provider."
        )
    else:
        tier = "High Risk Indication"
        advisory = (
            "Screening identifies significant risk indicators strongly associated with prediabetes or diabetes. "
            "A formal clinical assessment and standard blood testing (such as fasting glucose or HbA1c) "
            "conducted by a licensed healthcare professional is strongly recommended."
        )
    return tier, advisory


def predict_risk(user_features: dict, model=None) -> dict:
    """
    Generates a non-invasive diabetes risk screening indication for a single user.
    
    Args:
        user_features (dict): Dictionary mapping the 16 feature names to encoded values.
        model: Optional preloaded model pipeline. If None, loads from default path.
        
    Returns:
        dict: Assessment results including ML risk score, risk tier, and screening guidance.
    """
    if model is None:
        model = load_model()

    # Verify that all 16 required features are present
    missing_cols = [c for c in SELECTED_FEATURES if c not in user_features]
    if missing_cols:
        raise ValueError(f"Missing required features: {missing_cols}")

    # Build 1-row DataFrame preserving strict column ordering
    input_df = pd.DataFrame([user_features])[SELECTED_FEATURES]

    # Generate prediction and continuous risk score
    pred_class = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    risk_score = float(probabilities[1])  # ML-generated risk score for Class 1

    risk_tier, advisory = map_score_to_risk_tier(risk_score)

    return {
        "raw_prediction": pred_class,
        "risk_score": risk_score,
        "risk_score_percentage": round(risk_score * 100, 1),
        "risk_tier": risk_tier,
        "advisory": advisory,
        "threshold_note": "Initial 30% and 60% thresholds are prototype screening thresholds and not clinical diagnostic thresholds."
    }


def run_sample_predictions():
    """
    Demonstrates sample predictions on two representative profiles,
    illustrating automated BMI calculation and age mapping.
    """
    print("=" * 80)
    print("EARLYDIA: PREDICTION & RISK SCREENING DEMONSTRATION")
    print("Non-Invasive Early Diabetes Risk Screening System")
    print("=" * 80)

    model = load_model()
    print("[EarlyDia] Successfully loaded serialized model pipeline (`models/selected_model.joblib`).\n")

    # Profile 1: Younger, active individual
    # In Phase 2, user inputs Height = 172 cm, Weight = 65 kg -> BMI is calculated automatically
    height_1, weight_1 = 172.0, 65.0
    bmi_1 = calculate_bmi(height_1, weight_1)
    age_1 = 26  # mapped to bracket 2 (25-29)

    sample_profile_1 = {
        "BMI": bmi_1,                                # 22.0 (calculated from 172cm, 65kg)
        "Age": map_age_to_bracket(age_1),            # 2 (age 25-29)
        "Sex": 0,                                    # Female
        "GenHlth": 1,                                # Excellent general health
        "MentHlth": 2,                               # 2 days poor mental health
        "PhysHlth": 0,                               # 0 days poor physical health
        "HighBP": 0,                                 # No high blood pressure
        "HighChol": 0,                               # No high cholesterol
        "Stroke": 0,                                 # No stroke
        "HeartDiseaseorAttack": 0,                   # No heart disease
        "DiffWalk": 0,                               # No difficulty walking
        "PhysActivity": 1,                           # Regularly active
        "Fruits": 1,                                 # Eats fruit daily
        "Veggies": 1,                                # Eats vegetables daily
        "Smoker": 0,                                 # Non-smoker
        "HvyAlcoholConsump": 0                       # No heavy alcohol
    }

    # Profile 2: Older, hypertensive, sedentary individual
    # In Phase 2, user inputs Height = 175 cm, Weight = 104 kg -> BMI is calculated automatically
    height_2, weight_2 = 175.0, 104.0
    bmi_2 = calculate_bmi(height_2, weight_2)
    age_2 = 67  # mapped to bracket 10 (65-69)

    sample_profile_2 = {
        "BMI": bmi_2,                                # 34.0 (calculated from 175cm, 104kg)
        "Age": map_age_to_bracket(age_2),            # 10 (age 65-69)
        "Sex": 1,                                    # Male
        "GenHlth": 4,                                # Fair general health
        "MentHlth": 5,                               # 5 days poor mental health
        "PhysHlth": 15,                              # 15 days poor physical health
        "HighBP": 1,                                 # Diagnosed high blood pressure
        "HighChol": 1,                               # Diagnosed high cholesterol
        "Stroke": 0,                                 # No stroke
        "HeartDiseaseorAttack": 1,                   # History of heart disease
        "DiffWalk": 1,                               # Difficulty walking/climbing stairs
        "PhysActivity": 0,                           # Sedentary (no exercise in past 30 days)
        "Fruits": 0,                                 # Does not eat fruit daily
        "Veggies": 0,                                # Does not eat vegetables daily
        "Smoker": 1,                                 # Smoker
        "HvyAlcoholConsump": 0                       # No heavy alcohol
    }

    profiles = [
        (
            "Profile 1 (Younger, active, healthy indicators)",
            f"Height: {height_1}cm, Weight: {weight_1}kg -> Calculated BMI: {bmi_1}, Age: {age_1}",
            sample_profile_1
        ),
        (
            "Profile 2 (Older, elevated BMI, hypertension, sedentary)",
            f"Height: {height_2}cm, Weight: {weight_2}kg -> Calculated BMI: {bmi_2}, Age: {age_2}",
            sample_profile_2
        )
    ]

    for title, biometric_summary, profile in profiles:
        print("-" * 80)
        print(f"INPUT: {title}")
        print(f"BIOMETRICS: {biometric_summary}")
        print("-" * 80)
        result = predict_risk(profile, model)
        print(f"ML Risk Score:         {result['risk_score_percentage']:.1f}%")
        print(f"Risk Screening Tier:   {result['risk_tier']}")
        print(f"Binary Indication:     {result['raw_prediction']} (0 = Low Baseline, 1 = Elevated Risk Pattern)")
        print(f"Screening Advisory:    {result['advisory']}")
        print(f"Threshold Note:        {result['threshold_note']}")
        print()

    print("=" * 80)
    print("IMPORTANT DISCLAIMER:")
    print("EarlyDia is a Machine Learning-based preliminary risk screening tool and is not intended")
    print("to diagnose diabetes or replace professional medical advice or clinical testing.")
    print("=" * 80)


if __name__ == "__main__":
    run_sample_predictions()
