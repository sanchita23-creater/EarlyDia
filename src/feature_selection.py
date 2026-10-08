"""
src/feature_selection.py
------------------------
Defines and documents feature selection for EarlyDia: Non-Invasive Early Diabetes Risk Screening.
Distinguishes features selected for preliminary screening from excluded administrative/socioeconomic features.
Generates `reports/feature_selection.txt` documenting rationales, encodings, and frontend collection mappings.
"""

import os

# ==============================================================================
# FEATURE CONSTANTS
# ==============================================================================

# Numeric / Ordinal Features requiring scaling in linear models
NUMERIC_FEATURES = [
    "BMI",       # Continuous / Integer (12-98)
    "GenHlth",   # Ordinal rating (1-5)
    "MentHlth",  # Integer days (0-30)
    "PhysHlth",  # Integer days (0-30)
    "Age"        # Ordinal age bracket (1-13)
]

# Binary Features (0 or 1)
BINARY_FEATURES = [
    "HighBP",               # 0 = No, 1 = Yes
    "HighChol",             # 0 = No, 1 = Yes
    "Smoker",               # 0 = No, 1 = Yes
    "Stroke",               # 0 = No, 1 = Yes
    "HeartDiseaseorAttack", # 0 = No, 1 = Yes
    "PhysActivity",         # 0 = No, 1 = Yes
    "Fruits",               # 0 = No, 1 = Yes
    "Veggies",              # 0 = No, 1 = Yes
    "HvyAlcoholConsump",    # 0 = No, 1 = Yes
    "DiffWalk",             # 0 = No, 1 = Yes
    "Sex"                   # 0 = Female, 1 = Male
]

# Complete list of 16 selected features for EarlyDia
SELECTED_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES

# Excluded features from UCI Dataset ID 891
EXCLUDED_FEATURES = {
    "CholCheck": {
        "meaning": "Had cholesterol check within the past 5 years (0 = No, 1 = Yes)",
        "reason": "Measures healthcare utilization frequency rather than physiological or lifestyle diabetes risk. Additionally, 96.3% of respondents answered 'Yes', providing negligible discriminative signal."
    },
    "AnyHealthcare": {
        "meaning": "Have any kind of healthcare coverage/insurance (0 = No, 1 = Yes)",
        "reason": "Administrative and insurance access indicator rather than an individual physiological or lifestyle risk factor. Insurance status varies widely by nation and does not reflect metabolic health."
    },
    "NoDocbcCost": {
        "meaning": "Could not see a doctor because of cost in past 12 months (0 = No, 1 = Yes)",
        "reason": "Financial and healthcare system barrier metric rather than an individual health risk determinant."
    },
    "Education": {
        "meaning": "Highest level of education completed (scale 1-6)",
        "reason": "Socioeconomic demographic variable that serves as a distal proxy rather than a direct biological or lifestyle determinant. Excluded to prevent socioeconomic bias in clinical screening."
    },
    "Income": {
        "meaning": "Annual household income bracket (scale 1-8 based on US dollar thresholds)",
        "reason": "US-specific dollar income scale that induces user hesitation/privacy discomfort during screening, lacks direct metabolic relevance, and introduces geographical and economic bias."
    }
}

# Detailed metadata for selected features with exhaustive encoding details for Phase 2 frontend mapping
SELECTED_FEATURES_METADATA = [
    {
        "name": "BMI",
        "category": "Anthropometric Indicator",
        "meaning": "Body Mass Index (weight in kg / height in meters squared)",
        "encoding": "Continuous / Integer: Range in dataset is 12 to 98 (mean ~28.38). Typical values: <18.5 Underweight, 18.5-24.9 Normal, 25-29.9 Overweight, 30+ Obese.",
        "why_selected": "Established primary risk factor for type 2 diabetes. Easily computed non-invasively without laboratory blood draws.",
        "frontend_collection": (
            "IMPORTANT FRONTEND INSTRUCTION FOR PHASE 2:\n"
            "Users typically do not know their BMI directly. The frontend must collect 'Height' (cm or ft/in) "
            "and 'Weight' (kg or lbs) using user-friendly input fields and automatically calculate BMI using:\n"
            "   BMI = weight_in_kg / ((height_in_cm / 100) ** 2)\n"
            "The calculated BMI value is then passed to the model. Height and Weight themselves must NOT be added "
            "as model features because the CDC trained dataset uses BMI."
        )
    },
    {
        "name": "Age",
        "category": "Demographic Information",
        "meaning": "13-level discrete age category (_AGEG5YR BRFSS codebook)",
        "encoding": (
            "Discrete Integer scale (1 to 13):\n"
            "   1 = Age 18 - 24\n"
            "   2 = Age 25 - 29\n"
            "   3 = Age 30 - 34\n"
            "   4 = Age 35 - 39\n"
            "   5 = Age 40 - 44\n"
            "   6 = Age 45 - 49\n"
            "   7 = Age 50 - 54\n"
            "   8 = Age 55 - 59\n"
            "   9 = Age 60 - 64\n"
            "  10 = Age 65 - 69\n"
            "  11 = Age 70 - 74\n"
            "  12 = Age 75 - 79\n"
            "  13 = Age 80 or older"
        ),
        "why_selected": "Insulin resistance and type 2 diabetes risk increase significantly with age.",
        "frontend_collection": (
            "Dropdown with labeled age ranges (e.g. '18-24', '25-29', ..., '80+'), "
            "or a single age input box with a helper function mapping age to the 1-13 category."
        )
    },
    {
        "name": "Sex",
        "category": "Demographic Information",
        "meaning": "Biological sex of the participant",
        "encoding": "Binary: 0 = Female, 1 = Male",
        "why_selected": "Biological sex influences body fat distribution, hormone profiles, and metabolic patterns.",
        "frontend_collection": "Radio buttons or toggle: 'Biological Sex' (Options: Female -> 0, Male -> 1)."
    },
    {
        "name": "GenHlth",
        "category": "General Well-Being",
        "meaning": "Self-assessed overall health status (GENHLTH BRFSS scale)",
        "encoding": (
            "Ordinal Integer scale (1 to 5):\n"
            "   1 = Excellent\n"
            "   2 = Very Good\n"
            "   3 = Good\n"
            "   4 = Fair\n"
            "   5 = Poor"
        ),
        "why_selected": "Strongest single predictor of metabolic health in the CDC BRFSS dataset (correlation = 0.294).",
        "frontend_collection": "Select box or button group: 'In general, how would you describe your health?' (Options: Excellent -> 1, Very Good -> 2, Good -> 3, Fair -> 4, Poor -> 5)."
    },
    {
        "name": "MentHlth",
        "category": "General Well-Being",
        "meaning": "Days of poor mental health (stress, depression, emotional difficulties) during the past 30 days",
        "encoding": "Discrete Integer: 0 to 30 days (0 = 0 days, 30 = every day)",
        "why_selected": "Chronic psychological distress and depression correlate with metabolic dysregulation and poorer lifestyle adherence.",
        "frontend_collection": "Slider or number input (0 - 30): 'During the past 30 days, about how many days was your mental health not good?'"
    },
    {
        "name": "PhysHlth",
        "category": "General Well-Being",
        "meaning": "Days of physical illness and injury during the past 30 days",
        "encoding": "Discrete Integer: 0 to 30 days (0 = 0 days, 30 = every day)",
        "why_selected": "Measures ongoing chronic physical illness burden and functional impairment.",
        "frontend_collection": "Slider or number input (0 - 30): 'During the past 30 days, about how many days was your physical health not good?'"
    },
    {
        "name": "HighBP",
        "category": "Physical Condition / Medical History",
        "meaning": "Ever told by doctor, nurse, or health professional that blood pressure is high",
        "encoding": "Binary: 0 = No, 1 = Yes",
        "why_selected": "Hypertension is a hallmark comorbidity of insulin resistance and metabolic syndrome.",
        "frontend_collection": "Radio buttons: 'Have you ever been diagnosed with high blood pressure?' (No -> 0, Yes -> 1)."
    },
    {
        "name": "HighChol",
        "category": "Physical Condition / Medical History",
        "meaning": "Ever told by doctor, nurse, or health professional that blood cholesterol is high",
        "encoding": "Binary: 0 = No, 1 = Yes",
        "why_selected": "Dyslipidemia frequently co-occurs with elevated diabetes risk and cardiovascular compromise.",
        "frontend_collection": "Radio buttons: 'Have you ever been told by a health professional that your cholesterol is high?' (No -> 0, Yes -> 1)."
    },
    {
        "name": "Stroke",
        "category": "Physical Condition / Medical History",
        "meaning": "Ever told by a health professional that you had a stroke",
        "encoding": "Binary: 0 = No, 1 = Yes",
        "why_selected": "Major cardiovascular/cerebrovascular event indicating systemic vascular and metabolic risk.",
        "frontend_collection": "Radio buttons: 'Have you ever been diagnosed with a stroke?' (No -> 0, Yes -> 1)."
    },
    {
        "name": "HeartDiseaseorAttack",
        "category": "Physical Condition / Medical History",
        "meaning": "History of coronary heart disease (CHD) or myocardial infarction (heart attack)",
        "encoding": "Binary: 0 = No, 1 = Yes",
        "why_selected": "Coronary heart disease shares primary metabolic and vascular risk pathways with type 2 diabetes.",
        "frontend_collection": "Radio buttons: 'Have you ever had coronary heart disease or a heart attack?' (No -> 0, Yes -> 1)."
    },
    {
        "name": "DiffWalk",
        "category": "Physical Condition / Mobility",
        "meaning": "Serious difficulty walking or climbing stairs",
        "encoding": "Binary: 0 = No, 1 = Yes",
        "why_selected": "Indicator of physical mobility impairment, frailty, and sedentary risk profile (correlation = 0.218).",
        "frontend_collection": "Radio buttons: 'Do you have serious difficulty walking or climbing stairs?' (No -> 0, Yes -> 1)."
    },
    {
        "name": "PhysActivity",
        "category": "Lifestyle Information",
        "meaning": "Participated in physical activity or exercise during the past 30 days other than regular job",
        "encoding": "Binary: 0 = No, 1 = Yes",
        "why_selected": "Regular physical activity directly enhances cellular insulin sensitivity and assists glucose regulation.",
        "frontend_collection": "Radio buttons: 'Did you participate in any physical exercise during the past 30 days outside work?' (No -> 0, Yes -> 1)."
    },
    {
        "name": "Fruits",
        "category": "Lifestyle Information",
        "meaning": "Consumes fruit 1 or more times per day",
        "encoding": "Binary: 0 = No, 1 = Yes",
        "why_selected": "Reflects regular dietary intake of micronutrients, antioxidants, and whole food fiber.",
        "frontend_collection": "Radio buttons: 'Do you consume fruit at least once per day?' (No -> 0, Yes -> 1)."
    },
    {
        "name": "Veggies",
        "category": "Lifestyle Information",
        "meaning": "Consumes vegetables 1 or more times per day",
        "encoding": "Binary: 0 = No, 1 = Yes",
        "why_selected": "Regular vegetable intake provides dietary fiber crucial for metabolic regulation and gut health.",
        "frontend_collection": "Radio buttons: 'Do you consume vegetables at least once per day?' (No -> 0, Yes -> 1)."
    },
    {
        "name": "Smoker",
        "category": "Lifestyle Information",
        "meaning": "Smoked at least 100 cigarettes in entire life (5 packs)",
        "encoding": "Binary: 0 = No, 1 = Yes",
        "why_selected": "Tobacco smoking contributes to chronic systemic inflammation and impaired insulin sensitivity.",
        "frontend_collection": "Radio buttons: 'Have you smoked at least 100 cigarettes in your lifetime?' (No -> 0, Yes -> 1)."
    },
    {
        "name": "HvyAlcoholConsump",
        "category": "Lifestyle Information",
        "meaning": "Heavy alcohol consumption (>14 drinks/week for adult men, >7 drinks/week for adult women)",
        "encoding": "Binary: 0 = No, 1 = Yes",
        "why_selected": "Excessive alcohol consumption contributes to hepatic fat accumulation and metabolic strain.",
        "frontend_collection": "Radio buttons: 'Do you consume alcohol heavily (more than 14 drinks/week for men, 7 for women)?' (No -> 0, Yes -> 1)."
    }
]


def generate_feature_documentation(reports_dir: str = "reports") -> None:
    """
    Writes reports/feature_selection.txt detailing all selected and excluded features.
    """
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "feature_selection.txt")
    
    lines = []
    lines.append("=" * 85)
    lines.append("EARLYDIA: FEATURE SELECTION DOCUMENTATION")
    lines.append("Non-Invasive Early Diabetes Risk Screening Through Accessible Insights")
    lines.append("=" * 85)
    lines.append("")
    lines.append("PROJECT PHILOSOPHY & NON-INVASIVE IDENTITY:")
    lines.append("-" * 85)
    lines.append("EarlyDia is engineered strictly as a preliminary non-invasive risk screening system.")
    lines.append("It does NOT rely on laboratory blood-test values (no glucose, insulin, HbA1c, etc.).")
    lines.append("Every selected feature is self-reportable, accessible without specialized clinical")
    lines.append("equipment, and suitable for direct user collection via a future interactive frontend.")
    lines.append("")
    lines.append(f"Total Features in UCI Dataset 891:  21")
    lines.append(f"Features Selected for EarlyDia:     {len(SELECTED_FEATURES)}")
    lines.append(f"Features Excluded from EarlyDia:    {len(EXCLUDED_FEATURES)}")
    lines.append("")
    
    lines.append("=" * 85)
    lines.append("SECTION 1: FEATURES USED BY EARLYDIA (16 FEATURES)")
    lines.append("=" * 85)
    lines.append("")
    
    for i, f in enumerate(SELECTED_FEATURES_METADATA, 1):
        lines.append(f"Feature #{i}: {f['name']}")
        lines.append(f"  - Category:            {f['category']}")
        lines.append(f"  - Meaning:             {f['meaning']}")
        lines.append(f"  - Dataset Encoding:    {f['encoding']}")
        lines.append(f"  - Why Selected:        {f['why_selected']}")
        lines.append(f"  - Frontend Collection: {f['frontend_collection']}")
        lines.append("-" * 85)
    lines.append("")

    lines.append("=" * 85)
    lines.append("SECTION 2: FEATURES EXCLUDED FROM EARLYDIA (5 FEATURES)")
    lines.append("=" * 85)
    lines.append("")
    
    for i, (col, details) in enumerate(EXCLUDED_FEATURES.items(), 1):
        lines.append(f"Excluded #{i}: {col}")
        lines.append(f"  - Meaning:            {details['meaning']}")
        lines.append(f"  - Reason for Exclusion: {details['reason']}")
        lines.append("-" * 85)
    lines.append("")
    
    lines.append("=" * 85)
    lines.append("SECTION 3: FRONTEND MAPPING & IMPLEMENTATION RULES FOR PHASE 2")
    lines.append("=" * 85)
    lines.append("")
    lines.append("A. BMI COLLECTION & CALCULATION RULE:")
    lines.append("-" * 85)
    lines.append("  - Users often do not know their BMI directly.")
    lines.append("  - The Phase 2 frontend MUST collect 'Height' (cm or ft/in) and 'Weight' (kg or lbs).")
    lines.append("  - The frontend MUST automatically compute BMI using:")
    lines.append("        BMI = weight_in_kg / ((height_in_cm / 100) ** 2)")
    lines.append("  - Height and Weight themselves MUST NOT be added as model features.")
    lines.append("  - Only the calculated BMI value is passed to the prediction pipeline.")
    lines.append("")
    lines.append("B. EXHAUSTIVE DATASET ENCODING REFERENCE TABLE:")
    lines.append("-" * 85)
    lines.append("  Feature       | Type      | Model Encoding / Value Format")
    lines.append("  --------------+-----------+-----------------------------------------------------------")
    lines.append("  BMI           | Numeric   | Continuous/Integer (typical 12 - 98)")
    lines.append("  Age           | Discrete  | 1=18-24, 2=25-29, 3=30-34, 4=35-39, 5=40-44, 6=45-49,")
    lines.append("                |           | 7=50-54, 8=55-59, 9=60-64, 10=65-69, 11=70-74,")
    lines.append("                |           | 12=75-79, 13=80 or older")
    lines.append("  Sex           | Binary    | 0 = Female, 1 = Male")
    lines.append("  GenHlth       | Ordinal   | 1 = Excellent, 2 = Very Good, 3 = Good, 4 = Fair, 5 = Poor")
    lines.append("  MentHlth      | Discrete  | Integer 0 to 30 (days of poor mental health in past 30 days)")
    lines.append("  PhysHlth      | Discrete  | Integer 0 to 30 (days of physical illness in past 30 days)")
    lines.append("  HighBP        | Binary    | 0 = No, 1 = Yes (diagnosed high blood pressure)")
    lines.append("  HighChol      | Binary    | 0 = No, 1 = Yes (diagnosed high cholesterol)")
    lines.append("  Stroke        | Binary    | 0 = No, 1 = Yes (diagnosed stroke)")
    lines.append("  HeartDisease  | Binary    | 0 = No, 1 = Yes (CHD or heart attack)")
    lines.append("  DiffWalk      | Binary    | 0 = No, 1 = Yes (difficulty walking or climbing stairs)")
    lines.append("  PhysActivity  | Binary    | 0 = No, 1 = Yes (exercise in past 30 days outside work)")
    lines.append("  Fruits        | Binary    | 0 = No, 1 = Yes (fruit >= 1 time/day)")
    lines.append("  Veggies       | Binary    | 0 = No, 1 = Yes (vegetables >= 1 time/day)")
    lines.append("  Smoker        | Binary    | 0 = No, 1 = Yes (smoked >= 100 cigarettes in lifetime)")
    lines.append("  HvyAlcohol    | Binary    | 0 = No, 1 = Yes (>14 drinks/wk men, >7 drinks/wk women)")
    lines.append("")
    lines.append("C. RISK SCORE & SCREENING TIERS FRAMING:")
    lines.append("-" * 85)
    lines.append("  - The predict_proba output represents an ML-generated risk score / screening indication.")
    lines.append("  - It is NOT a medically confirmed probability of having diabetes.")
    lines.append("  - Prototype Screening Thresholds (Heuristic Student Prototype, NOT Clinical Diagnostic):")
    lines.append("      * Low Risk Indication:                 Risk Score < 30.0%")
    lines.append("      * Moderate / Elevated Risk Indication: 30.0% <= Risk Score <= 60.0%")
    lines.append("      * High Risk Indication:                Risk Score > 60.0%")
    lines.append("  - Clear medical disclaimers must be displayed alongside any screening indication.")
    lines.append("=" * 85)

    doc_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(doc_text)
        
    print(f"[EarlyDia] Successfully generated feature documentation in: {report_path}")


if __name__ == "__main__":
    generate_feature_documentation()
