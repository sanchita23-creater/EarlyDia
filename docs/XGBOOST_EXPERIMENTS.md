# XGBoost Implementation and Experiments

## 1. Candidate Tuning
The XGBoost candidates were integrated cleanly into the `src/v2/features_v2.py` pipeline definition. The hyperparameter search space used `scale_pos_weight` to address the severe class imbalance (~3.5% positive prevalence) and tested combinations of depth, tree counts, and regularization.

### Tested Configurations
* **`xgb_default`**: Baseline XGBoost with `scale_pos_weight=20.0`.
* **`xgb_A`**: `max_depth=4`, `learning_rate=0.05`, `n_estimators=200`, `scale_pos_weight=15.0`.
* **`xgb_B`**: `max_depth=3`, `learning_rate=0.05`, `n_estimators=300`, `scale_pos_weight=25.0`.
* **`xgb_C`**: `max_depth=4`, `learning_rate=0.03`, `n_estimators=300`, `scale_pos_weight=30.0`.

## 2. Evaluation Results (OOF CV)

All experiments were evaluated using 5-fold PSU-grouped cross-validation on the development data, with cross-fitted calibration.

### Men's Results
| Model | ROC-AUC | PR-AUC | Sensitivity | Specificity | Calibration |
|-------|---------|--------|-------------|-------------|-------------|
| `xgb_B` | **0.6948** | **0.0977** | 0.7586 | 0.5026 | Sigmoid |
| `xgb_A` | 0.6941 | 0.0985 | 0.7659 | 0.5080 | Sigmoid |
| `lr_state` (v2 baseline) | 0.6915 | 0.0849 | 0.7586 | 0.5045 | Sigmoid |

**Analysis**: Both `xgb_A` and `xgb_B` outperform the current best men's candidate (`lr_state`) on both ROC-AUC and PR-AUC. `xgb_B` is chosen as the strongest robust candidate.

### Women's Results
| Model | ROC-AUC | PR-AUC | Sensitivity | Specificity | Calibration |
|-------|---------|--------|-------------|-------------|-------------|
| `state_native_B` (v2 baseline) | **0.7011** | 0.0969 | 0.7513 | 0.5215 | Isotonic |
| `xgb_A` | 0.6996 | **0.0986** | 0.7618 | 0.5028 | Isotonic |
| `xgb_B` | 0.6996 | 0.0965 | 0.7667 | 0.5005 | Isotonic |

**Analysis**: While `state_native_B` has a marginal edge in ROC-AUC (+0.0015), `xgb_A` demonstrates superior PR-AUC (+0.0017). Given the rare positive class (~3.1% in women), precision-recall (PR-AUC) is the more informative metric per our evaluation protocol.

## 3. Promotion Decision
Based on the predefined evaluation policy which prioritizes properly calibrated, leakage-free models evaluated on informative metrics (PR-AUC over pure ROC-AUC for severe imbalance):
- **Men**: Promote `xgb_B`.
- **Women**: Promote `xgb_A`.

These candidates will proceed to the single-shot final evaluation on the held-out test split, after which their production artifacts will be serialized to `models_v2/` and registered in `configs/model_registry.json`.
