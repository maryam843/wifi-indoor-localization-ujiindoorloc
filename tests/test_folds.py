import numpy as np
import pytest

from src.folds import StableGroupKFold, assign_group_folds


def make_groups(seed=0, n_groups=200):
    # Many groups with the same size, like the 735 location groups in the training data.
    rng = np.random.default_rng(seed)
    sizes = rng.choice([10, 20, 20, 30, 30, 30], size=n_groups)
    groups = np.repeat([f"g{i:03d}" for i in range(n_groups)], sizes)
    return rng.permutation(groups)


def test_same_folds_on_two_calls():
    groups = make_groups()
    first = assign_group_folds(groups, 5)
    second = assign_group_folds(groups, 5)
    assert first.equals(second)


def test_same_folds_whatever_the_row_order():
    groups = make_groups()
    shuffled = np.random.default_rng(1).permutation(groups)
    a = assign_group_folds(groups, 5).sort_values("group").reset_index(drop=True)
    b = assign_group_folds(shuffled, 5).sort_values("group").reset_index(drop=True)
    assert a.equals(b)


def test_every_group_in_exactly_one_fold():
    groups = make_groups()
    table = assign_group_folds(groups, 5)
    assert sorted(table["group"]) == sorted(set(groups))
    assert table["group"].is_unique
    assert set(table["fold"]) == set(range(5))


def test_folds_are_balanced():
    # Tolerance: fold sizes differ by at most the size of the largest group.
    groups = make_groups()
    table = assign_group_folds(groups, 5)
    per_fold = table.groupby("fold")["n_samples"].sum()
    assert per_fold.sum() == len(groups)
    assert per_fold.max() - per_fold.min() <= table["n_samples"].max()


def test_split_never_splits_a_group():
    groups = make_groups()
    seen = set()
    for train_idx, test_idx in StableGroupKFold(5).split(np.zeros(len(groups)), groups=groups):
        test_groups = set(groups[test_idx])
        assert test_groups.isdisjoint(groups[train_idx])
        assert test_groups.isdisjoint(seen)
        seen |= test_groups
        assert len(train_idx) + len(test_idx) == len(groups)
    assert seen == set(groups)


def test_split_needs_groups():
    with pytest.raises(ValueError):
        next(StableGroupKFold(5).split(np.zeros(10)))


def test_too_few_groups():
    with pytest.raises(ValueError):
        assign_group_folds(["a", "a", "b"], 5)
