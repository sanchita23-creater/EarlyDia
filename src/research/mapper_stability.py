"""Mapper stability: bootstrap + lens/resolution/overlap/seed variation.

Per-run labelings are compared on the points both runs cover using adjusted Rand index and
normalised mutual information; each reference community gets a best-match Jaccard against every
other run. A community is called STABLE if its mean best-match Jaccard >= STABLE_JACCARD.
Only stable communities may be discussed substantively.
"""
from dataclasses import replace

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.research.mapper_analysis import MapperConfig, build_mapper

STABLE_JACCARD = 0.6   # documented a-priori rule, not tuned


def variations(base: MapperConfig):
    out = [replace(base, lens=l) for l in ("pca1", "pca2", "density") if l != base.lens]
    out += [replace(base, n_intervals=max(4, base.n_intervals - 2)), replace(base, n_intervals=base.n_intervals + 2)]
    out += [replace(base, overlap=max(0.2, base.overlap - 0.1)), replace(base, overlap=min(0.6, base.overlap + 0.1))]
    return out


def _jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0


def assess(X, base=MapperConfig(), n_boot=5, seed=42):
    ref = build_mapper(X, base)
    ref_sets = [set(np.where(ref["point_community"] == c)[0].tolist()) for c in range(len(ref["communities"]))]
    rng = np.random.default_rng(seed)
    runs = []
    for cfg in variations(base):
        runs.append(("variation", cfg, np.arange(len(X)), build_mapper(X, cfg)))
    for b in range(n_boot):
        idx = np.sort(rng.choice(len(X), len(X), replace=True)); idx = np.unique(idx)
        runs.append(("bootstrap", replace(base, seed=base.seed + b + 1), idx, build_mapper(X[idx], replace(base, seed=base.seed + b + 1))))
    ari, nmi = [], []
    best = [[] for _ in ref_sets]
    for kind, cfg, idx, res in runs:
        lab_full = np.full(len(X), -1); lab_full[idx] = res["point_community"]
        both = (ref["point_community"] >= 0) & (lab_full >= 0)
        if both.sum() > 10:
            ari.append(adjusted_rand_score(ref["point_community"][both], lab_full[both]))
            nmi.append(normalized_mutual_info_score(ref["point_community"][both], lab_full[both]))
        run_sets = [set(idx[np.where(res["point_community"] == c)[0]].tolist()) for c in range(len(res["communities"]))]
        for i, s in enumerate(ref_sets):
            sub = s & set(idx.tolist())          # compare within the points this run could see
            best[i].append(max([_jaccard(sub, r) for r in run_sets] or [0.0]))
    stab = [float(np.mean(b)) if b else 0.0 for b in best]
    return dict(reference=ref, n_runs=len(runs), mean_ari=float(np.mean(ari)) if ari else float("nan"),
                mean_nmi=float(np.mean(nmi)) if nmi else float("nan"),
                community_jaccard=stab, stable=[s >= STABLE_JACCARD for s in stab], rule=f"stable if mean best-match Jaccard >= {STABLE_JACCARD}")
