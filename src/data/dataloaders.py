"""DataLoader configuration shared across experiments."""

import torch
from torch.utils.data import DataLoader, Dataset


def get_dataloader(
    dataset: Dataset,
    split: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    collate_fn=None,
    pin_memory: bool | None = None,
) -> DataLoader:
    """Build a DataLoader for a prepared dataset.

    Validation and test loaders never shuffle. Images must be transformed to
    tensors before default batching. For detection datasets with variable
    numbers of boxes, supply an appropriate collate_fn.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unknown split: {split}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if split == "train" else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
