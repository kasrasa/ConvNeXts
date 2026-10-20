"""Paired image and semantic segmentation mask transforms."""

from PIL import Image
import torch
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


def transforms(
    image: Image.Image,
    mask: Image.Image,
    *,
    train: bool = False,
    size: tuple[int, int] = (256, 256),
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform an image and semantic mask using the same spatial operations.

    Training uses a shared random crop and horizontal flip, with color jitter
    applied only to the image. Validation applies deterministic resizing.
    Mask category IDs (including ignore label 255) are preserved.
    Provide mean/std only if the selected model requires normalization here.
    """
    if not isinstance(image, Image.Image):
        image = F.to_pil_image(image)
    image = image.convert("RGB")
    if not isinstance(mask, Image.Image):
        mask = F.to_pil_image(mask)
    if mask.mode not in ("L", "P", "I", "I;16"):
        raise ValueError("The segmentation mask must contain single-channel class IDs.")
    if image.size != mask.size:
        raise ValueError("The image and mask must have matching spatial dimensions.")
    if (mean is None) != (std is None):
        raise ValueError("Supply both mean and std, or neither.")

    if train:
        top, left, height, width = T.RandomResizedCrop.get_params(
            image, scale=(0.8, 1.0), ratio=(3 / 4, 4 / 3)
        )
        image = F.resized_crop(
            image, top, left, height, width, size, InterpolationMode.BILINEAR
        )
        mask = F.resized_crop(
            mask, top, left, height, width, size, InterpolationMode.NEAREST
        )
        if torch.rand(()).item() < 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)
        image = T.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        )(image)
    else:
        image = F.resize(image, size, interpolation=InterpolationMode.BILINEAR)
        mask = F.resize(mask, size, interpolation=InterpolationMode.NEAREST)

    image_tensor = F.to_tensor(image)
    if mean is not None:
        image_tensor = F.normalize(image_tensor, mean, std)
    mask_tensor = F.pil_to_tensor(mask).squeeze(0).long()
    return image_tensor, mask_tensor
