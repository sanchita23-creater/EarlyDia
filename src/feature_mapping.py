"""
src/feature_mapping.py
-----------------------
Feature processing, validation, and biometric mapping for EarlyDia Phase 2.
Maps user-entered self-reported health/lifestyle responses to the exact 16 model features
and dataset encodings established in Phase 1 (UCI Dataset ID 891).
"""

from typing import Tuple, Dict, Any
import pandas as pd

# The exact 16 features and exact column order expected by the trained pipeline
MODEL_FEATURE_ORDER = [
    "BMI",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "Age",
    "HighBP",
    "HighChol",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "DiffWalk",
    "Sex"
]

# Standard CDC BRFSS _AGEG5YR 13-bracket codebook mapping
AGE_BRACKETS = {
    1: {"label": "18 – 24 years", "range": (18, 24)},
    2: {"label": "25 – 29 years", "range": (25, 29)},
    3: {"label": "30 – 34 years", "range": (30, 34)},
    4: {"label": "35 – 39 years", "range": (35, 39)},
    5: {"label": "40 – 44 years", "range": (40, 44)},
    6: {"label": "45 – 49 years", "range": (45, 49)},
    7: {"label": "50 – 54 years", "range": (50, 54)},
    8: {"label": "55 – 59 years", "range": (55, 59)},
    9: {"label": "60 – 64 years", "range": (60, 64)},
    10: {"label": "65 – 69 years", "range": (65, 69)},
    11: {"label": "70 – 74 years", "range": (70, 74)},
    12: {"label": "75 – 79 years", "range": (75, 79)},
    13: {"label": "80 years or older", "range": (80, 125)},
}

# General Health Rating (BRFSS GENHLTH 1-5 scale)
GEN_HLTH_OPTIONS = {
    "Excellent": 1,
    "Very Good": 2,
    "Good": 3,
    "Fair": 4,
    "Poor": 5
}

GEN_HLTH_DESCRIPTIONS = {
    "Excellent": "Optimal overall well-being, boundless energy, and no functional limits.",
    "Very Good": "Consistently strong health with minimal daily fatigue or physical restrictions.",
    "Good": "Generally healthy and functional, with occasional minor physical complaints.",
    "Fair": "Moderate persistent health concerns or recurring physical limitations.",
    "Poor": "Substantial health challenges significantly affecting daily living activities."
}


def calculate_bmi(height_cm: float, weight_kg: float) -> Tuple[float, str, str]:
    """
    Computes Body Mass Index (BMI) from Height (cm) and Weight (kg).
    Formula: BMI = weight_kg / ((height_cm / 100) ** 2)
    
    Returns:
        Tuple[float, str, str]: (bmi_value, category_label, category_color)
    """
    if height_cm <= 0 or weight_kg <= 0:
        raise ValueError("Height and weight must be positive numbers.")
        
    height_m = height_cm / 100.0
    bmi = round(weight_kg / (height_m ** 2), 1)
    
    # WHO & CDC Standard Adult BMI Categories
    if bmi < 18.5:
        category = "Underweight"
        color = "#3b82f6"  # Blue
    elif bmi < 25.0:
        category = "Normal Weight"
        color = "#10b981"  # Green / Emerald
    elif bmi < 30.0:
        category = "Overweight"
        color = "#f59e0b"  # Amber
    else:
        category = "Obesity"
        color = "#ef4444"  # Red / Rose
        
    return bmi, category, color


def map_age_to_bracket(age_in_years: int) -> Tuple[int, str]:
    """
    Maps adult age in years to the 13-bracket CDC BRFSS _AGEG5YR encoding scale.
    
    Returns:
        Tuple[int, str]: (bracket_code, bracket_label)
    """
    if age_in_years < 18:
        raise ValueError("EarlyDia is calibrated for adult risk screening (18 years or older).")
    
    for bracket_code, info in AGE_BRACKETS.items():
        low, high = info["range"]
        if low <= age_in_years <= high:
            return bracket_code, info["label"]
            
    # For ages >= 80
    return 13, AGE_BRACKETS[13]["label"]


def build_model_input_dataframe(user_profile: Dict[str, Any]) -> pd.DataFrame:
    """
    Converts user profile dictionary into a 1-row Pandas DataFrame
    with the exact 16 feature names and order expected by selected_model.joblib.
    """
    # Verify all 16 features exist
    for feature in MODEL_FEATURE_ORDER:
        if feature not in user_profile:
            raise KeyError(f"Missing required model feature: '{feature}'")
            
    df = pd.DataFrame([user_profile])[MODEL_FEATURE_ORDER]
    return df
