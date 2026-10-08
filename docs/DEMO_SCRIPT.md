# EarlyDia v2 - 3-Minute Demo Walkthrough Script

This script walks through a concise, 3-minute live demonstration of the EarlyDia v2 screening application.

**Target audience**: Clinical researchers, ML engineers, evaluators.  
**Tone**: Honest, grounded, methodologically precise.  
**Key framing**: EarlyDia is a research/prototype **screening and risk stratification** system built on NFHS-5 survey data for adults aged 18–40. It evaluates statistical resemblance to an **"NFHS-5 diabetes-range status"** research construct. It is **not** a clinical diagnostic test, does **not** predict future diabetes, and is **not** clinically validated.

---

### Minute 0:00 - 0:30 | Landing Page

1. **Open** `http://localhost:5173` in your browser.
2. **Key points to say**:
   - "EarlyDia is a prototype screening classification tool developed on India's NFHS-5 survey data (2019–2021) for young adults aged 18 to 40."
   - "It models women and men separately due to structural differences in the survey (e.g., BMI was not collected for men in NFHS-5)."
   - "Notice the framing throughout: this is screening risk stratification, not medical diagnosis or future disease prediction."
3. **Action**: Click **"Start Screening"**.

---

### Minute 0:30 - 1:00 | Step 1: Profile (with State Selection)

1. **Select Profile Inputs**:
   - **Sex**: Female (or Male).
   - **Age**: 28.
   - **State / Union Territory**: Select e.g. **Maharashtra** (or Tamil Nadu).
   - **Education**: Higher secondary or Graduate.
   - **Residence**: Urban.
2. **Key points to say**:
   - "In EarlyDia v2, State is introduced as a required contextual regional marker capturing geographic baseline prevalence patterns from NFHS-5 sampling clusters."
   - "State is an ecological marker, not an individual biological cause of risk."
   - "If a state code were omitted via the API, the system falls back to code 9 (Uttar Pradesh, the survey mode) with an explicit notice, but the UI enforces conscious selection."
3. **Action**: Click **"Continue"**.

---

### Minute 1:00 - 1:30 | Step 2: Lifestyle & Step 3: Health

1. **Step 2 (Lifestyle)**:
   - Answer tobacco questions (e.g., No).
   - Answer alcohol questions (e.g., No).
   - Complete the 9 diet frequency questions (pulses, dairy, green leafy vegetables, fruits, eggs, etc.).
   - Click **"Continue"**.
2. **Step 3 (Health)**:
   - Enter **Height** (e.g., 160 cm) and **Weight** (e.g., 58 kg) — note real-time BMI display for women.
   - Enter **Blood Pressure** (e.g., 120 / 80 mmHg).
   - *(Optional IDRS comparison fields can be left empty or filled)*.
   - Click **"Continue"**.
3. **Key points to say**:
   - "Blood pressure readings are required; physiological bounds are validated client- and server-side."
   - "All user inputs map deterministically to the backend feature contract."

---

### Minute 1:30 - 2:00 | Step 4: Review

1. **Review Page**:
   - Review each structured card: Profile (including State / UT), Lifestyle, Health.
   - Highlight that any section can be modified directly via its "Edit" link before submission.
2. **Key points to say**:
   - "The review step ensures complete transparency of the data payload sent to the FastAPI backend."
3. **Action**: Click **"Start Screening"**.

---

### Minute 2:00 - 3:00 | Screening Result & Interpretation

1. **Examine Result Elements**:
   - **Probability**: Calibrated percentage output (e.g., `4.8%`).
   - **Status Pill**: "Flagged for follow-up" or "Lower screening likelihood".
   - **Model & Cutoff**: Small meta line: `Model v2 - screening threshold 0.024` (or `0.029` for men).
   - **State Disclosure**: Note stating `State used: <State Name>. State is a regional marker from NFHS-5, not a cause of risk.`
   - **Interactive Technical Details**: Expand the accordion to view held-out test ROC-AUC, sensitivity, specificity, and calibration methodology.
2. **Key points to say (What the result DOES and DOES NOT mean)**:
   - **What it IS**: "The calibrated probability reflects statistical alignment with the research-defined NFHS-5 diabetes-range group (self-reported diagnosis or random blood glucose ≥ 200 mg/dL) at the chosen sensitivity-oriented cutoff (≥ 75% target sensitivity on training data)."
   - **What it is NOT**: "This does NOT mean the user has diabetes. It does not predict lifetime risk. It is not an alternative to clinical tests like HbA1c or fasting plasma glucose."
   - **Why PPV is modest**: "Because population prevalence in this 18–40 cohort is ~3.0–3.3%, any sensitive screening threshold inherently produces a low Positive Predictive Value (~4.7–4.9%) and many false alarms. That trade-off is documented and transparent."
3. **Conclusion**:
   - "EarlyDia demonstrates rigorous ML engineering under clustered survey conditions: cluster-aware splitting, out-of-fold calibration, honest metric reporting, and safe product design."
