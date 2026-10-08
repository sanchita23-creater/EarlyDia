import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from src.v2 import evaluation_v2 as E

oof_dir = _ROOT / "data" / "processed" / "v2"

ref = pd.read_pickle(oof_dir / "men_oof_v1_repro.pkl")
y = ref["y"].values
folds = ref["fold"].values
psu = ref["psu"].values
age = ref["age"].values

candidates = {
    "v1_repro": ref["p_raw"].values,
    "lr_state": pd.read_pickle(oof_dir / "men_oof_lr_state.pkl")["p_raw"].values,
    "state": pd.read_pickle(oof_dir / "men_oof_state.pkl")["p_raw"].values,
    "state_native_C": pd.read_pickle(oof_dir / "men_oof_state_native_C.pkl")["p_raw"].values,
    "blend_lr_state (50:50)": pd.read_pickle(oof_dir / "men_oof_blend_men_lr_state.pkl")["p_raw"].values,
    "blend_lr_stateC (50:50)": pd.read_pickle(oof_dir / "men_oof_blend_men_lr_stateC.pkl")["p_raw"].values,
}

# Also test weighted logit blends: 60/40 and 70/30
z_lr = E._logit(candidates["lr_state"]).ravel()
z_state = E._logit(candidates["state"]).ravel()
z_stateC = E._logit(candidates["state_native_C"]).ravel()

for w_lr, w_tree, tag, z_tree in [
    (0.6, 0.4, "blend_60lr_40state", z_state),
    (0.7, 0.3, "blend_70lr_30state", z_state),
    (0.6, 0.4, "blend_60lr_40stateC", z_stateC),
    (0.7, 0.3, "blend_70lr_30stateC", z_stateC),
]:
    z = w_lr * z_lr + w_tree * z_tree
    candidates[tag] = 1.0 / (1.0 + np.exp(-z))

rows = []
for name, p in candidates.items():
    s, p_cal, _ = E.evaluate_oof(y, p, folds, age, floor=0.75)
    f_auc, f_ap = E.fold_consistency(y, p, candidates["v1_repro"], folds)
    b = E.psu_bootstrap_delta(y, p, candidates["v1_repro"], psu, n_boot=200)
    rows.append({
        "name": name,
        "roc_auc": s["roc_auc"],
        "d_auc": b["d_auc"],
        "d_auc_ci": f"[{b['d_auc_lo']:+.4f}, {b['d_auc_hi']:+.4f}]",
        "pr_auc": s["pr_auc"],
        "d_ap": b["d_ap"],
        "d_ap_ci": f"[{b['d_ap_lo']:+.4f}, {b['d_ap_hi']:+.4f}]",
        "fold_wins": f"{f_auc}/5 AUC, {f_ap}/5 AP",
        "threshold": s["threshold"],
        "sens": s["sensitivity"],
        "spec": s["specificity"],
        "ppv": s["ppv"],
        "flag_rate": s["flag_rate"],
        "top10_lift": s["top10_lift"],
        "cal_slope": s["cal_slope"],
        "ece": s["ece"],
    })

res_df = pd.DataFrame(rows).set_index("name")
with pd.option_context("display.width", 260, "display.max_columns", 20):
    print(res_df.round(4).to_string())
