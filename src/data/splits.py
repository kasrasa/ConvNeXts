import torch
import torchvision
import Path

def get_train_val_split(path: Path, ratio = 0.2, seed = 42):
    """
    Get the train split of the dataset.

    Args:
        path (Path): Path to the dataset.
        ratio (float): Ratio of the validation set. Default is 0.2.
        seed (int): Random seed for shuffling. Default is 42.
        shuffle (bool): Whether to shuffle the dataset before splitting. Default is True.

    Returns:
        train_split (torch.utils.data.Subset): Train split of the dataset.
        val_split (torch.utils.data.Subset): Validation split of the dataset.
    """
    full_dataset = torchvision.datasets.ImageFolder(root=path)

    num_samples = len(full_dataset)
    num_val_samples = int(num_samples * ratio)

    indices = list(range(num_samples))

    train_indices = indices[num_val_samples:]
    val_indices = indices[:num_val_samples]

    train_split = torch.utils.data.Subset(full_dataset, train_indices)
    val_split = torch.utils.data.Subset(full_dataset, val_indices)

    return train_split, val_split

