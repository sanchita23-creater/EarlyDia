"""Covariance-preserving null model for Mapper (methodological caution after Topaz 2026).

A Mapper graph can look 'structured' purely from covariance geometry. We draw B datasets from a
multivariate normal with the OBSERVED feature mean and covariance (same n), apply the IDENTICAL Mapper
configuration, and compare the label-free community-separation statistic with the observed one.
Limitation: Gaussian nulls do not reproduce binary/one-hot marginals exactly.
Finding the observed structure is not clearly stronger than null => no distinct-subtype claim.
"""
import numpy as np

from src.research.mapper_analysis import MapperConfig, build_mapper, community_separation

VERDICT_NOT_SUPPORTED = "The observed Mapper structure is not sufficient to support a distinct subtype claim."


def run_null(X, cfg=MapperConfig(), B=30, seed=42, alpha=0.05):
    obs_res = build_mapper(X, cfg)
    obs = community_separation(X, obs_res["point_community"])
    mu, cov = X.mean(0), np.cov(X, rowvar=False)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(B):
        Z = rng.multivariate_normal(mu, cov, size=len(X), method="cholesky" if np.all(np.linalg.eigvalsh(cov) > 1e-10) else "svd")
        null.append(community_separation(Z, build_mapper(Z, cfg)["point_community"]))
    null = np.array(null)
    p = float((1 + (null >= obs).sum()) / (1 + B))
    supported = p < alpha
    return dict(observed=obs, null_mean=float(null.mean()), null_sd=float(null.std()), null_values=null.tolist(),
                B=B, empirical_p=p, alpha=alpha, exceeds_null=bool(supported),
                verdict=("Observed separation exceeds the covariance-matched null (exploratory; still not evidence of biological subtypes)."
                         if supported else VERDICT_NOT_SUPPORTED))
