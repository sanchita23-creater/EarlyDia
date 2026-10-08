"""Exploratory Mapper / TDA on the production predictor space (research-only).

Outputs are 'exploratory feature-space communities' / 'data-driven population subgroups' -
NOT biological diabetes subtypes. Target A is never an input to the topology; the topology is
built on TRAINING-set predictors only, and the final test set is never used to construct or
tune it. Labels are used only AFTER construction to characterise communities.

Pipeline: encode predictors (median-impute, one-hot categoricals, standardise) -> lens (1st PCA
component by default) -> overlapping interval cover -> single-linkage clustering inside each
preimage -> nodes/edges (shared points) -> communities (greedy modularity).
Implemented with numpy/sklearn/networkx only (no kmapper dependency).
"""
from dataclasses import dataclass

import networkx as nx
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class MapperConfig:
    n_intervals: int = 8
    overlap: float = 0.4
    lens: str = "pca1"          # 'pca1' | 'pca2' | 'density'
    cluster_scale: float = 1.5  # single-linkage cut = scale * median kNN distance (label-free)
    min_node_size: int = 30
    seed: int = 42


def encode_predictors(df, num_cols, cat_cols, ref=None):
    """Impute + one-hot + standardise. Pass `ref` (stats from the training set) to encode other data
    identically. Never touches the target."""
    import pandas as pd
    if ref is None:
        med = {c: float(np.nanmedian(df[c])) for c in num_cols}
        cats = {c: sorted(pd.Series(df[c]).dropna().unique().tolist()) for c in cat_cols}
        ref = {"med": med, "cats": cats}
    parts = [np.column_stack([np.nan_to_num(df[c].to_numpy(float), nan=ref["med"][c]) for c in num_cols])]
    names = list(num_cols)
    for c in cat_cols:
        for v in ref["cats"][c]:
            parts.append((df[c].to_numpy() == v).astype(float)[:, None]); names.append(f"{c}={v}")
    X = np.hstack(parts)
    if "mu" not in ref:
        ref["mu"], ref["sd"] = X.mean(0), np.where(X.std(0) == 0, 1.0, X.std(0))
    return (X - ref["mu"]) / ref["sd"], names, ref


def _lens(X, cfg):
    if cfg.lens == "density":
        k = min(15, len(X) - 1)
        d, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
        return -d[:, -1]
    n = 2 if cfg.lens == "pca2" else 1
    comp = PCA(n_components=n, random_state=cfg.seed).fit_transform(X)
    return comp[:, -1]


def build_mapper(X, cfg=MapperConfig()):
    """Returns dict(graph, node_members, communities(list of sets of node ids), point_community(array, -1 = uncovered))."""
    n = len(X)
    f = _lens(X, cfg)
    lo, hi = f.min(), f.max()
    width = (hi - lo) / (cfg.n_intervals - (cfg.n_intervals - 1) * cfg.overlap)
    step = width * (1 - cfg.overlap)
    k = min(5, n - 1)
    dist, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    cut = cfg.cluster_scale * float(np.median(dist[:, -1]))
    nodes = []
    for i in range(cfg.n_intervals):
        a = lo + i * step; b = a + width
        idx = np.where((f >= a) & (f <= b + (1e-9 if i == cfg.n_intervals - 1 else 0)))[0]
        if len(idx) < cfg.min_node_size:
            continue
        lab = AgglomerativeClustering(n_clusters=None, distance_threshold=cut, linkage="single").fit_predict(X[idx])
        for c in np.unique(lab):
            mem = idx[lab == c]
            if len(mem) >= cfg.min_node_size:
                nodes.append(set(mem.tolist()))
    G = nx.Graph(); G.add_nodes_from(range(len(nodes)))
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if nodes[i] & nodes[j]:
                G.add_edge(i, j, weight=len(nodes[i] & nodes[j]))
    comms = [set(c) for c in nx.community.greedy_modularity_communities(G, weight="weight")] if len(nodes) else []
    pc = np.full(n, -1)
    best = np.zeros(n)
    for ci, cm in enumerate(comms):          # point -> community where it appears in the most nodes
        counts = np.zeros(n)
        for nid in cm:
            counts[list(nodes[nid])] += 1
        upd = counts > best
        pc[upd] = ci; best[upd] = counts[upd]
    return dict(graph=G, node_members=nodes, communities=comms, point_community=pc, config=cfg)


def community_separation(X, point_community):
    """Label-free separation: mean eta^2 (between-community variance share) over encoded features,
    on covered points. Used for observed-vs-null comparison."""
    m = point_community >= 0
    if m.sum() < 10 or len(np.unique(point_community[m])) < 2:
        return 0.0
    Z, g = X[m], point_community[m]
    gm = Z.mean(0); tot = ((Z - gm) ** 2).sum(0); tot[tot == 0] = np.inf
    bet = np.zeros(Z.shape[1])
    for c in np.unique(g):
        zc = Z[g == c]; bet += len(zc) * (zc.mean(0) - gm) ** 2
    return float((bet / tot).mean())


def characterise(point_community, y=None, p=None, thr=None, features=None):
    """Per-community summary. y/p are used ONLY for characterisation after the topology is fixed."""
    import pandas as pd
    rows = []
    for c in sorted(set(point_community[point_community >= 0])):
        m = point_community == c
        r = dict(community=int(c), n=int(m.sum()))
        if features is not None:
            for col in features.columns:
                r[f"mean_{col}"] = float(np.nanmean(features.loc[m, col]))
        if y is not None:
            r["observed_prevalence"] = float(np.mean(y[m]))
        if p is not None:
            r["mean_predicted"] = float(np.mean(p[m]))
            if thr is not None:
                r["flagged_rate"] = float(np.mean(p[m] >= thr))
        rows.append(r)
    return pd.DataFrame(rows)
