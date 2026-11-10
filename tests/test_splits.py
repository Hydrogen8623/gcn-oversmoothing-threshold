
import numpy as np
import pytest

from src.data.splits import stratified_split


def test_stratified_split_has_correct_counts():
    labels = np.repeat(np.arange(4), 100)

    train, val, test = stratified_split(
        labels,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
    )

    assert train.sum() == 240
    assert val.sum() == 80
    assert test.sum() == 80

    for class_label in range(4):
        class_mask = labels == class_label

        assert train[class_mask].sum() == 60
        assert val[class_mask].sum() == 20
        assert test[class_mask].sum() == 20


def test_stratified_split_has_no_overlap_and_covers_all_nodes():
    labels = np.repeat(np.arange(4), 100)

    train, val, test = stratified_split(
        labels, 0.6, 0.2, 0.2, seed=42
    )

    assert not np.any(train & val)
    assert not np.any(train & test)
    assert not np.any(val & test)

    assert np.all(train | val | test)


def test_stratified_split_is_reproducible():
    labels = np.repeat(np.arange(4), 100)

    result1 = stratified_split(
        labels, 0.6, 0.2, 0.2, seed=42
    )
    result2 = stratified_split(
        labels, 0.6, 0.2, 0.2, seed=42
    )

    for a, b in zip(result1, result2):
        assert np.array_equal(a, b)


def test_invalid_split_ratios():
    labels = np.repeat(np.arange(4), 100)

    with pytest.raises(ValueError):
        stratified_split(labels, 0.6, 0.3, 0.2, seed=42)

    with pytest.raises(ValueError):
        stratified_split(labels, 0.6, -0.1, 0.5, seed=42)