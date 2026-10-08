# Target definition comparison

**Alternative literature-aligned target - NOT used by the EarlyDia production model.**

| Aspect | EarlyDia Target A (production, locked) | Sahadevan et al. 2023 (J Glob Health) |
|---|---|---|
| Population | NFHS-5, ages 18-40, women non-pregnant + men | NFHS-5, ages 15-49, women and men |
| Known diabetes | any of sb56/sb57/s728a (women), smb56/smb57/sm627a (men) = Yes -> positive | 'diagnosed' variable; undiagnosed = answered 'no' or 'don't know' |
| Positive (glucose) | valid glucose >= 200 mg/dL, OR any known-diabetes flag | fasting >=126 mg/dL with >=8 h fasting, OR random >=200 mg/dL (among those not diagnosed) |
| Negative | all flags explicit No, valid glucose, glucose <= 140 | not diagnosed and below the above glucose criteria |
| Ambiguous handling | 141-199 mg/dL without known diabetes -> excluded | not excluded by glucose band |
| Includes already-diagnosed people | Yes (known diabetes = positive) | No (outcome is *undiagnosed* diabetes) |

## Replication status

Full replication was **not possible** from the retained extraction: Fasting status/duration variables cannot be verified from the retained extraction documentation, and no row-level data is present to inspect their codes. No alternative target was constructed and no alternative prevalence is reported.

Consequently no alternative-target prevalence is reported here. Prevalence from the two definitions must not be compared as like-for-like: the populations, age ranges, treatment of already-diagnosed people and glucose rules differ.

This alternative target may only ever be used for prevalence sensitivity, target comparison and robustness analysis. It must never be used to retrain, rethreshold, recalibrate or alter inference.
