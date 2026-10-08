# References

Status key: **Verified** = identity and link confirmed from the open-access page retrieved while preparing this build;
**Unverified details** = cited from the supplied brief / bibliographic knowledge; confirm volume, pages and DOI on the publisher page before submission.
No reference below was invented; where a DOI could not be confirmed it is omitted.

## NFHS / Indian diabetes
1. Sahadevan P, Kamal VK, Sasidharan A, Bagepally BS, Kumari D, Pal A. Prevalence and risk factors associated with undiagnosed diabetes in India: Insights from NFHS-5 national survey. *J Glob Health* 2023. https://pmc.ncbi.nlm.nih.gov/articles/PMC10704946/ - **Verified** (definition: 'no'/'don't know' to diagnosed diabetes plus fasting >=126 mg/dL with >=8 h fasting or random >=200 mg/dL; ages 15-49).
2. Geldsetzer P, et al. Diabetes and hypertension in India: a nationally representative study of 1.3 million adults. *JAMA Intern Med* 2018. - **Unverified details** (DOI not confirmed in this build).
3. Barman et al. Interpretable machine learning for diabetes risk prediction: a large-scale analysis of Indian national survey data. *Discover Public Health* 2025. - **Not located in this build**; no methodological details are asserted (see `reports/research/literature_benchmark.md`).
4. Mohan V, et al. A simplified Indian Diabetes Risk Score for screening for undiagnosed diabetic subjects. *J Assoc Physicians India* 2005;53:759-63. - **Scoring table verified** against several independent published reproductions (age 0/20/30; waist by sex 0/10/20; activity 0/20/30; family history 0/10/20; bands <30, 30-50, >=60); the original paper was not retrieved - click-check the journal page. Used by `backend/api/idrs.py` as an independent comparator, not a model feature.

## Methodology
5. Roberts DR, et al. Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography* 2017. doi:10.1111/ecog.02881
6. Van Calster B, et al. Calibration: the Achilles heel of predictive analytics. *BMC Med* 2019. doi:10.1186/s12916-019-1466-7
7. Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. *Med Decis Making* 2006. doi:10.1177/0272989X06295361
8. Grinsztajn L, Oyallon E, Varoquaux G. Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS* 2022 (Datasets and Benchmarks). arXiv:2207.08815
9. Collins GS, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ* 2024. doi:10.1136/bmj-2023-078378

*DOIs in items 5-9 are given from memory of the published records and should still be click-checked.*

## Heterogeneity / topology
10. Ahlqvist E, et al. Novel subgroups of adult-onset diabetes and their association with outcomes. *Lancet Diabetes Endocrinol* 2018. doi:10.1016/S2213-8587(18)30051-2 - conceptual motivation only.
11. Li L, et al. Identification of type 2 diabetes subgroups through topological analysis of patient similarity. *Sci Transl Med* 2015. - motivation for patient-similarity analysis only; **unverified details**.
12. Topaz, 2026. "A Null Model for Mapper Subtype Claims." - supplied in the brief; **not located or verified in this build**; used only as a methodological warning (covariance-matched null model).

## Feature evidence (defined in docs/RESEARCH_EVIDENCE.md)
[ADA-2], [WHO-BMI], [INDIAB-1], [NFHS-5], [EMDIN-2015], [WILLI-2007], [PAN-2015], [KNOTT-2015], [BALIUNAS-2009], [CARTER-2010], [SCHWING-2017], [GIJSBERS-2016], [IMAMURA-2015], [CAHILL-2014]: full citations in `docs/RESEARCH_EVIDENCE.md` (same verification caveat).
