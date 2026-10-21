# Group k-fold split that gives the same folds on every machine.
#
# sklearn's GroupKFold orders the groups by size with np.argsort, which does not fix the order of
# groups that have the same size. That order, and so the folds, can differ between CPUs. Here the
# groups are ordered by size (largest first) and then by group id, and each group goes to the fold
# with the fewest samples so far (the lowest fold number on a tie). Nothing is random.
import numpy as np
import pandas as pd


def assign_group_folds(groups, n_splits=5):
    # Returns one row per group: group id, number of samples, fold number (0 .. n_splits - 1).
    groups = pd.Series(np.asarray(groups)).astype(str)
    sizes = groups.value_counts()
    if len(sizes) < n_splits:
        raise ValueError(f"need at least {n_splits} groups, got {len(sizes)}")
    order = sorted(sizes.index, key=lambda g: (-sizes[g], g))
    fold_samples = np.zeros(n_splits, dtype=np.int64)
    folds = []
    for g in order:
        fold = int(np.argmin(fold_samples))
        folds.append(fold)
        fold_samples[fold] += sizes[g]
    return pd.DataFrame({"group": order, "n_samples": [int(sizes[g]) for g in order], "fold": folds})


class StableGroupKFold:
    # Use it like sklearn's GroupKFold: cv.split(X, y, groups), or cv=... in GridSearchCV and RandomizedSearchCV.

    def __init__(self, n_splits=5):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("groups must be given")
        groups = np.asarray(groups).astype(str)
        table = assign_group_folds(groups, self.n_splits)
        fold = pd.Series(table["fold"].to_numpy(), index=table["group"]).loc[groups].to_numpy()
        index = np.arange(len(groups))
        for k in range(self.n_splits):
            yield index[fold != k], index[fold == k]
