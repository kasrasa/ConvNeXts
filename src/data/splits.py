"""Reproducible train/validation splits for PyTorch datasets."""

from collections.abc import Sequence

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset


def get_train_val_split(
    dataset: Dataset,
    ratio: float = 0.2,
    seed: int = 42,
    labels: Sequence[int] | None = None,
    val_dataset: Dataset | None = None,
) -> tuple[Subset, Subset]:
    """Split one dataset into train and validation subsets.

    Pass labels for a stratified classification split. If training and
    validation require different transforms, pass a second dataset instance
    with identical sample ordering as val_dataset.
    """
    if not 0 < ratio < 1:
        raise ValueError("ratio must be between 0 and 1.")
    if len(dataset) < 2:
        raise ValueError("The dataset must contain at least two samples.")
    if labels is not None and len(labels) != len(dataset):
        raise ValueError("labels must have one entry per dataset sample.")
    if val_dataset is not None and len(val_dataset) != len(dataset):
        raise ValueError("val_dataset must have the same length as dataset.")

    train_indices, val_indices = train_test_split(
        range(len(dataset)),
        test_size=ratio,
        random_state=seed,
        stratify=labels,
    )

    train_split = Subset(dataset, train_indices)
    val_split = Subset(
        val_dataset if val_dataset is not None else dataset,
        val_indices,
    )
    return train_split, val_split
