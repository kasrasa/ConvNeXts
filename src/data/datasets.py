import torch
import torchvision

def get_dataset(split: str, dataset: str, data_dir: str, download: bool = True):
    if dataset == "Oxford-IIIT Pet classification":
        if split == "train":
            return torchvision.datasets.OxfordIIITPet(
                root=data_dir, split="trainval", target_types="category", download=download
            )
        elif split == "val":
            return torchvision.datasets.OxfordIIITPet(
                root=data_dir, split="test", target_types="category", download=download
            )
        else:
            raise ValueError(f"Unknown split: {split}")
    if dataset == "COCO 2017 validation":
        if split == "val":
            return torchvision.datasets.CocoDetection(
                root=f"{data_dir}/val2017",
                annFile=f"{data_dir}/annotations/instances_val2017.json",
                download=download,
            )
        else:
            raise ValueError(f"Unknown split: {split}")
    if dataset == "ADE20K validation":
        if split == "val":
            return torchvision.datasets.ADE20K(
                root=data_dir, split="validation", download=download
            )
        else:
            raise ValueError(f"Unknown split: {split}")