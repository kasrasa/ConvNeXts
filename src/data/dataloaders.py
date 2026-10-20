import torch


def get_dataloader(dataset, split, batch_size, shuffle=True, num_workers=4):
    """
    Get the dataloader for the given dataset and split.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to load.
        split (str): The split of the dataset ('train' or 'val').
        batch_size (int): The batch size for the dataloader.
        shuffle (bool): Whether to shuffle the dataset. Default is True.
        num_workers (int): Number of subprocesses to use for data loading. Default is 4.
    """
    if split not in ["train", "val"]:
        raise ValueError(f"Unknown split: {split}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if split == "train" else False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader