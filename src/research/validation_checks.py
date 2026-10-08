"""Cluster-aware validation guards (PSU overlap, CV-fold overlap, accidental random CV)."""
import re
from pathlib import Path

from src.research.common import ROOT


def psu_overlap(train_psu, test_psu):
    return set(train_psu) & set(test_psu)


def assert_no_psu_overlap(train_psu, test_psu):
    ov = psu_overlap(train_psu, test_psu)
    assert not ov, f"{len(ov)} PSUs appear in both train and test"
    return True


def cv_fold_psu_overlaps(groups, folds):
    """folds: iterable of (train_idx, val_idx). Returns number of PSUs shared by train/val per fold."""
    import numpy as np
    g = np.asarray(groups)
    return [len(set(g[tr]) & set(g[va])) for tr, va in folds]


_RANDOM_CV = re.compile(r"\b(KFold|StratifiedKFold|ShuffleSplit|StratifiedShuffleSplit|train_test_split)\s*\(")


def scan_for_random_cv(paths=("src/models", "src/validation", "src/calibration", "src/evaluation")):
    """Static scan: flag non-grouped splitters in the production pipeline source. (Group* variants
    do not match because the regex requires the bare name.)"""
    hits = []
    for rel in paths:
        for f in (ROOT / rel).rglob("*.py"):
            for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                s = line.strip()
                if s.startswith("#") or s.startswith('"""'):
                    continue
                m = _RANDOM_CV.search(line)
                if m and not re.search(r"(Group|StratifiedGroup)" + m.group(1), line):
                    hits.append(f"{f.relative_to(ROOT)}:{i}: {s}")
    return hits
