"""Dataset loading for the classification, detection, and segmentation experiments."""

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection, OxfordIIITPet


class ADE20KDataset(Dataset):
    """Read ADE20K scene-parsing images and segmentation masks from disk.

    Expects the extracted ADEChallengeData2016/images and annotations folders.
    The original mask IDs are 1..150, with 0 denoting ignored pixels; returned
    masks use 0..149 and 255 for ignored pixels.
    """

    def __init__(self, root: str | Path, split: str = "validation", transforms=None):
        if split not in ("training", "validation"):
            raise ValueError(f"Unknown ADE20K split: {split}")

        root = Path(root)
        if (root / "ADEChallengeData2016").is_dir():
            root = root / "ADEChallengeData2016"

        images_dir = root / "images" / split
        masks_dir = root / "annotations" / split

        images = sorted(images_dir.glob("*.jpg"))
        self.samples = [(image, masks_dir / f"{image.stem}.png") for image in images]
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        with Image.open(mask_path) as mask_file:
            raw_mask = np.asarray(mask_file, dtype=np.uint8)

        mapped_mask = np.where(raw_mask == 0, 255, raw_mask.astype(np.int16) - 1)
        mask = Image.fromarray(mapped_mask.astype(np.uint8))

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        return image, mask


def get_dataset(
    split: str,
    dataset: str,
    data_dir: str | Path,
    download: bool = False,
    transform=None,
    transforms=None,
):
    """Load a dataset without inventing train/validation partitions.

    Oxford-IIIT Pet supports 'trainval' and 'test'. Split its trainval portion
    using get_train_val_split, keeping the official test set untouched.
    COCO and ADE20K are loaded from manually downloaded files. Use 'transforms'
    for synchronized image/target transforms (particularly segmentation masks).
    """
    if dataset == "Oxford-IIIT Pet classification":
        if split not in ("trainval", "test"):
            raise ValueError("Oxford-IIIT Pet supports 'trainval' and 'test'.")
        return OxfordIIITPet(
            root=data_dir,
            split=split,
            target_types="category",
            download=download,
            transform=transform,
            transforms=transforms,
        )

    if dataset == "COCO 2017 validation":
        if split != "val":
            raise ValueError("COCO 2017 validation supports only 'val'.")
        if download:
            raise ValueError("COCO must be downloaded and extracted manually.")
        data_dir = Path(data_dir)
        return CocoDetection(
            root=str(data_dir / "val2017"),
            annFile=str(data_dir / "annotations" / "instances_val2017.json"),
            transform=transform,
            transforms=transforms,
        )

    if dataset == "ADE20K validation":
        if split != "val":
            raise ValueError("ADE20K validation supports only 'val'.")
        if download:
            raise ValueError("ADE20K must be downloaded and extracted manually.")
        if transform is not None:
            raise ValueError("Use 'transforms' to transform ADE20K images and masks together.")
        return ADE20KDataset(
            root=data_dir,
            split="validation",
            transforms=transforms,
        )

    raise ValueError(f"Unknown dataset: {dataset}")
