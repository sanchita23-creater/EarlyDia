# EarlyDia - Research support for the model inputs

**Purpose.** Each input the production models use is listed with (1) the research that supports it
as a *recognised diabetes-related factor*, (2) a safe way to describe it, and (3) what must **not**
be claimed. This lives in documentation on purpose - it is not repeated on the landing or results pages.

**Ground rules (apply to every row)**

- EarlyDia learned *statistical associations* in NFHS-5 survey data. It did **not** discover that any
  feature causes diabetes, and no single feature "caused" an individual's result.
- The inputs were chosen because they are non-invasive, self-reportable or easily measured and exist in
  both NFHS-5 questionnaires - the literature below shows each is a *recognised factor in diabetes
  research*; it does not validate the model.
- The outcome is a research-defined NFHS-5 group (README, "Target definition"), not a clinical diagnosis.

---

## Features

### Age
- **Evidence:** ADA Standards of Care recommend screening all adults from age 35 and earlier in people with
  risk factors; diabetes prevalence rises with age in India (ICMR-INDIAB). [ADA-2], [INDIAB-1]
- **Safe explanation:** "Diabetes becomes more common with age, so age is included as an input."
- **Must not claim:** that ageing "causes" diabetes; that results are valid outside 18-40 (the training range).

### Sex (separate women / men models)
- **Evidence:** NFHS-5 uses separate women's and men's questionnaires with different items (e.g. BMI and some
  tobacco/diet items differ), so separate models were built. Prevalence and risk-factor patterns differ by sex
  in Indian population data. [NFHS-5], [INDIAB-1]
- **Safe explanation:** "Separate models are used for women and men because the survey measured them differently."
- **Must not claim:** a biological explanation for any difference between the models.

### Body mass index (women's model only; derived from height and weight)
- **Evidence:** Overweight/obesity is a core type 2 diabetes risk factor in the ADA screening criteria, with a lower
  BMI cut-off (23 kg/m2) for Asian populations; the WHO expert consultation supports lower Asian cut-offs. [ADA-2], [WHO-BMI]
- **Safe explanation:** "BMI is calculated from your height and weight and is one of the inputs for the women's model."
- **Must not claim:** that BMI alone indicates diabetes or health; that the men's model uses BMI (it does not - the
  locked men's model has no BMI feature).

### Blood pressure (systolic / diastolic)
- **Evidence:** Hypertension (>=130/80 mmHg or treated) is a listed diabetes risk factor in the ADA screening criteria;
  a meta-analysis of prospective studies found higher usual blood pressure associated with new-onset diabetes. [ADA-2], [EMDIN-2015]
- **Safe explanation:** "Blood pressure is a commonly measured health indicator that is statistically linked with diabetes."
- **Must not claim:** that a given reading means a person has/will get diabetes or hypertension.
- **Implementation note:** the models were trained with an internal "number of valid BP readings" survey field. The UI does not ask
  for it; the API treats the entered BP as a complete 3-of-3 average (the most common training value). BP is required - when it was
  absent, the locked models produced abnormal probabilities in a spot-check, so such requests are rejected.

### Tobacco use
- **Evidence:** Systematic reviews and meta-analyses link active smoking with higher type 2 diabetes incidence, with risk falling
  among those who quit. [WILLI-2007], [PAN-2015]
- **Safe explanation:** "Tobacco use is associated with higher diabetes risk in published studies."
- **Must not claim:** that smokeless/other tobacco products in the survey carry the same risk as cigarettes (the meta-analyses
  mainly concern cigarette smoking); that quitting will prevent diabetes.

### Alcohol (use, frequency, years, type)
- **Evidence:** Dose-response meta-analyses show a non-linear (J/U-shaped) association in women and a more linear one in men; heavy
  drinking is associated with higher risk. [KNOTT-2015], [BALIUNAS-2009]
- **Safe explanation:** "Alcohol intake pattern is associated with diabetes risk in published studies."
- **Must not claim:** that any amount of alcohol is protective or that anyone should start drinking.

### Diet (nine food-frequency items)
The nine items are the NFHS-5 dietary-frequency questions. Evidence strength differs by item:

| Item | Research support | Safe wording | Must not claim |
|---|---|---|---|
| Fruits, dark green leafy vegetables | Meta-analyses: higher fruit/vegetable intake (esp. green leafy vegetables) associated with lower incidence. [CARTER-2010], [SCHWING-2017] | "Fruit and vegetable intake is associated with diabetes risk in studies." | That eating them prevents diabetes. |
| Milk or curd | Dose-response meta-analysis: dairy (esp. yoghurt/low-fat) associated with lower incidence. [GIJSBERS-2016], [SCHWING-2017] | "Dairy intake is associated with diabetes risk in studies." | That dairy protects; the item does not separate types. |
| Aerated drinks | Meta-analysis: sugar-sweetened beverages associated with higher incidence. [IMAMURA-2015] | "Sugary drinks are associated with higher diabetes risk in studies." | That diet drinks behave the same; the item does not say if sweetened. |
| Fried food | Two large US cohorts: frequent fried-food intake associated with higher incidence, largely mediated by weight and blood pressure/lipids. [CAHILL-2014] | "Frequent fried food is associated with higher diabetes risk in studies." | That frying itself causes diabetes. |
| Chicken or meat | Red and processed meat show the strongest positive association; the survey item is broad. [SCHWING-2017] | Avoid specific claims; "meat intake is studied in diabetes research." | Any claim about chicken specifically. |
| Pulses, eggs, fish | Evidence limited or inconsistent for these food groups in the cited reviews. | "Included because the survey asked about it." | Any protective or harmful effect. |

### Education and residence (urban / rural)
- **Evidence:** ICMR-INDIAB reports diabetes prevalence varies across states and between urban and rural settings; such patterns
  reflect socioeconomic, lifestyle and health-service factors. [INDIAB-1]
- **Safe explanation:** "Education and place of residence are included as background factors that differ across the population."
- **Must not claim:** that education or living in a rural/urban area causes diabetes, or any judgement about a person's group.

---

## References

Verification status: **[CAHILL-2014] was checked against its indexed record while preparing this document.** All other entries are
given from the published bibliographic record as known to the author of this document and **must be confirmed on the publisher/PubMed page
before you submit** (volume/page numbers especially). For ADA/WHO documents, cite the edition you actually consulted.

- **[ADA-2]** American Diabetes Association Professional Practice Committee. *Diagnosis and Classification of Diabetes* (Section 2, Standards of
  Care in Diabetes - use the current annual edition). *Diabetes Care*.
- **[WHO-BMI]** WHO Expert Consultation. Appropriate body-mass index for Asian populations and its implications for policy and intervention
  strategies. *Lancet* 2004;363:157-163.
- **[INDIAB-1]** Anjana RM, et al. Metabolic non-communicable disease health report of India: the ICMR-INDIAB national cross-sectional study
  (ICMR-INDIAB-17). *Lancet Diabetes Endocrinol* 2023;11(7):474-489.
- **[NFHS-5]** International Institute for Population Sciences (IIPS) and ICF. *National Family Health Survey (NFHS-5), 2019-21: India.* Mumbai: IIPS; 2021.
- **[EMDIN-2015]** Emdin CA, Anderson SG, Woodward M, Rahimi K. Usual blood pressure and risk of new-onset diabetes: evidence from 4.1 million adults
  and a meta-analysis of prospective studies. *J Am Coll Cardiol* 2015;66(14):1552-1563.
- **[WILLI-2007]** Willi C, Bodenmann P, Ghali WA, Faris PD, Cornuz J. Active smoking and the risk of type 2 diabetes: a systematic review and
  meta-analysis. *JAMA* 2007;298(22):2654-2664.
- **[PAN-2015]** Pan A, et al. Relation of active, passive, and quitting smoking with incident type 2 diabetes: a systematic review and meta-analysis.
  *Lancet Diabetes Endocrinol* 2015;3(12):958-967.
- **[KNOTT-2015]** Knott C, Bell S, Britton A. Alcohol consumption and the risk of type 2 diabetes: a systematic review and dose-response meta-analysis
  of more than 1.9 million individuals from 38 observational studies. *Diabetes Care* 2015;38(9):1804-1812.
- **[BALIUNAS-2009]** Baliunas DO, et al. Alcohol as a risk factor for type 2 diabetes: a systematic review and meta-analysis. *Diabetes Care* 2009;32(11):2123-2132.
- **[CARTER-2010]** Carter P, et al. Fruit and vegetable intake and incidence of type 2 diabetes mellitus: systematic review and meta-analysis. *BMJ* 2010;341:c4229.
- **[SCHWING-2017]** Schwingshackl L, et al. Food groups and risk of type 2 diabetes mellitus: a systematic review and meta-analysis of prospective studies.
  *Eur J Epidemiol* 2017;32(5):363-375.
- **[GIJSBERS-2016]** Gijsbers L, et al. Consumption of dairy foods and diabetes incidence: a dose-response meta-analysis of observational studies.
  *Am J Clin Nutr* 2016;103(4):1111-1124.
- **[IMAMURA-2015]** Imamura F, et al. Consumption of sugar sweetened beverages, artificially sweetened beverages, and fruit juice and incidence of type 2
  diabetes: systematic review, meta-analysis, and estimation of population attributable fraction. *BMJ* 2015;351:h3576.
- **[CAHILL-2014]** Cahill LE, Pan A, Chiuve SE, Sun Q, Willett WC, Hu FB, Rimm EB. Fried-food consumption and risk of type 2 diabetes and coronary artery
  disease: a prospective study in 2 cohorts of US women and men. *Am J Clin Nutr* 2014;100(2):667-675.
