"""
src/insights.py
---------------
Personalized, rule-based profile observations for EarlyDia Phase 2.
Translates self-reported biometric and lifestyle inputs into transparent,
educational factors for clinical attention and positive lifestyle habits.

IMPORTANT MEDICAL COMMUNICATION PRINCIPLE:
EarlyDia NEVER asserts causality (e.g., "This factor caused your risk").
Observations are strictly described as:
"Based on your provided profile, the following factors may deserve additional attention."
"""

from typing import Dict, Any, List


def generate_profile_insights(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates educational, transparent observations from the user profile.
    Separates factors deserving attention from positive lifestyle factors.
    
    Args:
        profile (Dict[str, Any]): Dictionary containing the 16 model feature values
                                  and biometric metadata (height, weight, bmi, age).
                                  
    Returns:
        Dict[str, Any]: Categorized lists of factors for attention and protective habits.
    """
    factors_for_attention: List[Dict[str, str]] = []
    positive_habits: List[Dict[str, str]] = []

    # 1. BMI Observation
    bmi = profile.get("BMI", 0.0)
    if bmi >= 30.0:
        factors_for_attention.append({
            "title": "Elevated Body Mass Index (Obesity Range)",
            "icon": "⚖️",
            "category": "Anthropometric",
            "observation": f"Your calculated BMI is {bmi:.1f} kg/m², which falls into the obesity category. Elevated adipose tissue is a primary physiological contributor to cellular insulin resistance.",
            "suggestion": "Discuss balanced nutritional strategies and gradual, sustainable weight management with a registered dietitian or physician."
        })
    elif bmi >= 25.0:
        factors_for_attention.append({
            "title": "Overweight BMI Category",
            "icon": "⚖️",
            "category": "Anthropometric",
            "observation": f"Your calculated BMI is {bmi:.1f} kg/m², which falls into the overweight category (25.0–29.9 kg/m²). Modest weight adjustments can substantially enhance metabolic insulin sensitivity.",
            "suggestion": "Pairing regular low-impact physical movement with portion control can help guide your BMI toward a normal range."
        })
    elif 18.5 <= bmi < 25.0:
        positive_habits.append({
            "title": "Healthy Body Mass Index",
            "icon": "✨",
            "observation": f"Your calculated BMI is {bmi:.1f} kg/m², residing comfortably in the recommended standard range (18.5–24.9 kg/m²)."
        })

    # 2. Physical Activity Observation
    phys_act = profile.get("PhysActivity", 0)
    if phys_act == 0:
        factors_for_attention.append({
            "title": "Sedentary Routine / Low Physical Activity",
            "icon": "🚶",
            "category": "Lifestyle",
            "observation": "You reported no physical exercise outside regular work hours during the past 30 days. Regular skeletal muscle contraction directly stimulates GLUT4 glucose uptake independently of insulin.",
            "suggestion": "Gradually incorporate 150 minutes of moderate aerobic activity (e.g., brisk walking, cycling, or swimming) per week into your daily routine."
        })
    else:
        positive_habits.append({
            "title": "Regular Physical Activity",
            "icon": "🏃",
            "observation": "You engaged in exercise or physical movement outside work during the past 30 days, actively supporting glucose disposal and cardiovascular tone."
        })

    # 3. Dietary Habits (Fruits & Vegetables)
    fruits = profile.get("Fruits", 0)
    veggies = profile.get("Veggies", 0)
    if fruits == 0 and veggies == 0:
        factors_for_attention.append({
            "title": "Low Daily Dietary Fiber & Micronutrient Intake",
            "icon": "🥗",
            "category": "Dietary",
            "observation": "You reported not consuming fruits or vegetables at least once per day. Soluble dietary fiber slows postprandial glucose absorption and promotes metabolic satiety.",
            "suggestion": "Aim to include at least one serving of leafy greens, cruciferous vegetables, or whole fruits with each daily meal."
        })
    elif fruits == 1 and veggies == 1:
        positive_habits.append({
            "title": "Daily Fruit & Vegetable Intake",
            "icon": "🥦",
            "observation": "Consuming whole fruits and vegetables daily delivers vital micronutrients, polyphenols, and dietary fiber that stabilize glycemic spikes."
        })

    # 4. Tobacco Smoking
    smoker = profile.get("Smoker", 0)
    if smoker == 1:
        factors_for_attention.append({
            "title": "Lifetime Tobacco Exposure (100+ Cigarettes)",
            "icon": "🚭",
            "category": "Lifestyle",
            "observation": "You reported a history of smoking at least 100 cigarettes in your lifetime. Nicotine and combustion byproducts exacerbate systemic oxidative stress and impair endothelial and beta-cell function.",
            "suggestion": "If currently smoking, explore structured smoking cessation programs and nicotine replacement therapies with your healthcare clinician."
        })
    else:
        positive_habits.append({
            "title": "Minimal or No Tobacco Exposure",
            "icon": "🫁",
            "observation": "You have not accumulated significant lifetime tobacco exposure (<100 cigarettes), protecting vascular endothelium and metabolic regulation."
        })

    # 5. Heavy Alcohol Consumption
    alcohol = profile.get("HvyAlcoholConsump", 0)
    if alcohol == 1:
        factors_for_attention.append({
            "title": "Heavy Alcohol Consumption",
            "icon": "🍷",
            "category": "Lifestyle",
            "observation": "You indicated heavy alcohol consumption (>14 drinks/week for men, >7 drinks/week for women). Chronic excessive alcohol contributes to hepatic steatosis, triglyceride elevation, and metabolic burden.",
            "suggestion": "Consider moderating weekly alcohol intake to align with standard preventive public health guidelines."
        })
    else:
        positive_habits.append({
            "title": "Moderate or Non-Excessive Alcohol Intake",
            "icon": "💧",
            "observation": "Your alcohol intake does not meet the CDC criteria for heavy consumption, supporting healthy hepatic and metabolic processing."
        })

    # 6. High Blood Pressure (Hypertension)
    high_bp = profile.get("HighBP", 0)
    if high_bp == 1:
        factors_for_attention.append({
            "title": "Diagnosed High Blood Pressure (Hypertension)",
            "icon": "🩺",
            "category": "Cardiometabolic",
            "observation": "You reported being told by a healthcare professional that you have high blood pressure. Hypertension is a prominent component of metabolic syndrome and commonly clusters with insulin resistance.",
            "suggestion": "Maintain ongoing blood pressure monitoring and follow up regularly with your primary care provider."
        })

    # 7. High Cholesterol
    high_chol = profile.get("HighChol", 0)
    if high_chol == 1:
        factors_for_attention.append({
            "title": "Diagnosed Elevated Cholesterol (Dyslipidemia)",
            "icon": "🩸",
            "category": "Cardiometabolic",
            "observation": "You reported a history of elevated cholesterol. Altered lipid metabolism (elevated triglycerides, low HDL) often accompanies subclinical glucose dysregulation.",
            "suggestion": "Discuss a comprehensive fasting lipid panel and cardiovascular risk assessment with your clinician."
        })

    # 8. Cardiovascular History (Stroke / Heart Disease)
    stroke = profile.get("Stroke", 0)
    heart_dis = profile.get("HeartDiseaseorAttack", 0)
    if stroke == 1 or heart_dis == 1:
        events = []
        if stroke == 1:
            events.append("stroke")
        if heart_dis == 1:
            events.append("coronary heart disease/heart attack")
        factors_for_attention.append({
            "title": "Cardiovascular / Cerebrovascular History",
            "icon": "❤️",
            "category": "Clinical History",
            "observation": f"You reported a history of {', '.join(events)}. Cardiovascular disease shares major underlying metabolic, atherosclerotic, and inflammatory etiology with diabetes.",
            "suggestion": "Maintain comprehensive cardiovascular management and preventive screenings under active clinical supervision."
        })

    # 9. Mobility Limitations (DiffWalk)
    diff_walk = profile.get("DiffWalk", 0)
    if diff_walk == 1:
        factors_for_attention.append({
            "title": "Serious Difficulty Walking or Climbing Stairs",
            "icon": "♿",
            "category": "Mobility",
            "observation": "You reported serious difficulty walking or climbing stairs. Functional mobility limitations often restrict routine physical activity, compounding metabolic and cardiovascular risks.",
            "suggestion": "Consult a physical therapist or rehabilitation specialist for safe, seated, or assisted low-impact exercise options."
        })

    # 10. General Health Rating
    gen_hlth = profile.get("GenHlth", 1)
    if gen_hlth in [4, 5]:
        rating_label = "Fair" if gen_hlth == 4 else "Poor"
        factors_for_attention.append({
            "title": f"Self-Assessed {rating_label} General Health",
            "icon": "🌡️",
            "category": "Well-Being",
            "observation": f"You rated your general health as '{rating_label}'. In large epidemiological studies, self-assessed fair or poor health strongly correlates with systemic chronic disease burden and metabolic stress.",
            "suggestion": "Schedule a comprehensive health evaluation to investigate any underlying systemic fatigue, chronic pain, or metabolic concerns."
        })

    # 11. Days of Poor Physical or Mental Health
    phys_hlth = profile.get("PhysHlth", 0)
    ment_hlth = profile.get("MentHlth", 0)
    if phys_hlth >= 14 or ment_hlth >= 14:
        details = []
        if phys_hlth >= 14:
            details.append(f"{phys_hlth} days of poor physical health")
        if ment_hlth >= 14:
            details.append(f"{ment_hlth} days of poor mental health")
        factors_for_attention.append({
            "title": "Persistent Functional or Emotional Strain (>= 14 Days)",
            "icon": "🧠",
            "category": "Well-Being",
            "observation": f"You reported frequent impairment during the past 30 days: {', '.join(details)}. Chronic physical or emotional distress elevates cortisol and inflammatory cascades that disrupt glucose homeostasis.",
            "suggestion": "Discuss mental well-being, stress mitigation, and physical pain management strategies with your care team."
        })

    return {
        "factors_for_attention": factors_for_attention,
        "positive_habits": positive_habits
    }
