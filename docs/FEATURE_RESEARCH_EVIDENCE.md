# Feature research evidence

Each production predictor, the published evidence that it is a *recognised factor in diabetes research*, how to describe it safely,
and what must not be claimed. This table is documentation only; it does not validate the model. Full per-item detail (nine diet items)
is in [RESEARCH_EVIDENCE.md](RESEARCH_EVIDENCE.md); reference keys below are defined there and in [REFERENCES.md](REFERENCES.md).

| Feature | EarlyDia variable | Evidence | Safe interpretation | Causal claim prohibited |
|---|---|---|---|---|
| Age | `age` (v012 / mv012) | Screening guidance and Indian prevalence data show diabetes becomes more common with age [ADA-2], [INDIAB-1]. In EarlyDia, age is the top permutation-importance feature for both sexes. | Age is a predictive signal / risk marker; the model contributed strongly to discrimination through age. | "Ageing causes diabetes"; validity outside 18-40. |
| BMI (women only) | `bmi` (v445 / 100) | Excess weight is a listed screening risk factor, with lower Asian cut-offs [ADA-2], [WHO-BMI]. | BMI is associated with the outcome and contributes to the women's model. The men's model has **no** BMI feature. | "BMI causes diabetes"; any BMI claim for men. |
| Blood pressure | `avg_systolic_bp`, `avg_diastolic_bp` | Higher usual BP is associated with new-onset diabetes in a meta-analysis [EMDIN-2015]; hypertension is a screening risk factor [ADA-2]. Geldsetzer et al. 2018 studied diabetes and hypertension together in India. | A risk marker; a reading does not mean a person has or will have diabetes or hypertension. | "High BP causes diabetes". |
| Tobacco | `tob_*` items | Active smoking is associated with higher type 2 diabetes incidence [WILLI-2007], [PAN-2015]; evidence is mainly for cigarettes. | Associated with higher risk in published studies. | That smokeless products carry the same risk; that quitting prevents diabetes. |
| Alcohol | `alc_*` items | Dose-response associations are non-linear in women and more linear in men [KNOTT-2015], [BALIUNAS-2009]. | Intake pattern is associated with risk in studies. | That any amount is protective. |
| Diet | `diet_*` (nine frequency items) | Fruit/vegetable, dairy, sugar-sweetened beverage and fried-food associations [CARTER-2010], [SCHWING-2017], [GIJSBERS-2016], [IMAMURA-2015], [CAHILL-2014]; evidence is limited for pulses, eggs and fish. | Frequency answers have model contribution; survey items are coarse. | That a food prevents or causes diabetes. |
| Education | `education` (v106 / mv106) | Prevalence varies across socioeconomic groups in India [INDIAB-1]. | A background factor that differs across the population. | That education causes any outcome. |
| Residence | `residence` (v025 / mv025) | Urban/rural prevalence differences in India [INDIAB-1]. | A background factor reflecting lifestyle, socioeconomic and service factors. | That living in a place causes diabetes. |

**Not in the production model (by design, locked):** waist circumference, physical activity, family history, wealth, hypertension and BMI for men.
**Wording rules:** use *associated with*, *predictive signal*, *model contribution*, *risk marker*. Avoid *causes*, *prevents*, *guarantees*.
Permutation importance is not SHAP and is not causal.

## Independent comparator: Indian Diabetes Risk Score (IDRS)

IDRS is **not a model feature**. Its three extra answers (waist circumference, physical activity, family history) are optional,
are collected in Step 3, and are used only by `backend/api/idrs.py` to display a published rule-based score beside the
EarlyDia result. They are never passed to the production models (verified by `tests/test_idrs.py`, which asserts the ML
probability, flag and cutoff are identical with and without them). The models, Target A, thresholds and feature contract are unchanged.

| Input | Scoring in IDRS | Evidence | Safe interpretation | Causal claim prohibited |
|---|---|---|---|---|
| Age | <35 = 0; 35-49 = 20; >=50 = 30 | Mohan 2005 [IDRS-1] | A component of a published screening score. | "Age causes diabetes." |
| Waist circumference | Women: <80 = 0, 80-89 = 10, >=90 = 20. Men: <90 = 0, 90-99 = 10, >=100 = 20 | Mohan 2005 [IDRS-1] | Abdominal obesity is a component of a published screening score. | "Waist size causes diabetes." |
| Physical activity | Regular exercise AND strenuous work = 0; either = 20; no exercise and sedentary work = 30 | Mohan 2005 [IDRS-1] | Lower activity scores higher in the published score. | "Inactivity causes diabetes." |
| Family history | None = 0; one parent = 10; both parents = 20 | Mohan 2005 [IDRS-1] | Family history is a component of a published screening score. | "A parent's diabetes means you will develop it." |

Risk bands: <30 lower, 30-50 moderate, >=60 higher. Possible totals are multiples of 10 (0-100).

**[IDRS-1]** Mohan V, Deepa R, Deepa M, Somannavar S, Datta M. A simplified Indian Diabetes Risk Score for screening for undiagnosed
diabetic subjects. *J Assoc Physicians India* 2005;53:759-63. **Verification status:** the scoring table was cross-checked against
several independent published reproductions of the original table; the original paper itself was not retrieved in this build -
click-check against the journal page before submission. IDRS was developed to detect *undiagnosed diabetes* in adults; it was not
built or validated on the NFHS-5 Target A, so the two tools are shown side by side and **never combined or compared as accuracy**.
