
"""Data splitting utilities for graph experiments."""

import numpy as np
import numpy.typing as npt


def stratified_split(
    labels: npt.NDArray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[
    npt.NDArray[np.bool_],
    npt.NDArray[np.bool_],
    npt.NDArray[np.bool_],
]:
    """Split nodes into train, validation, and test masks by class."""
    labels = np.asarray(labels)

    if labels.ndim != 1:
        raise ValueError("labels must be 1-D.")

    if labels.size == 0:
        raise ValueError("labels must not be empty.")

    ratios = np.array(
        [train_ratio, val_ratio, test_ratio],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(ratios)):
        raise ValueError("Split ratios must be finite.")

    if np.any(ratios <= 0):
        raise ValueError("All split ratios must be positive.")

    if not np.isclose(ratios.sum(), 1.0):
        raise ValueError("Split ratios must sum to 1.")

    train_mask = np.zeros(labels.size, dtype=bool)
    val_mask = np.zeros(labels.size, dtype=bool)
    test_mask = np.zeros(labels.size, dtype=bool)

    rng = np.random.default_rng(seed)

    for class_label in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_label)
        shuffled = rng.permutation(class_indices)

        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        if min(n_train, n_val, n_test) < 1:
            raise ValueError(
                "Each class must have at least one node in every split."
            )

        train_indices = shuffled[:n_train]
        val_indices = shuffled[n_train:n_train + n_val]
        test_indices = shuffled[n_train + n_val:]

        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True

    return train_mask, val_mask, test_mask