"""Image-only and synchronized image/segmentation-mask transformations."""

from collections.abc import Sequence

from PIL import Image
import torch
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


def image_transform(
    image: Image.Image,
    *,
    mean: Sequence[float],
    std: Sequence[float],
    train: bool = False,
    size: tuple[int, int] = (224, 224),
    crop_params: tuple[int, int, int, int] | None = None,
    horizontal_flip: bool | None = None,
) -> torch.Tensor:
    """Apply image augmentation and normalize using training-dataset statistics.

    For segmentation, supply the same crop_params/horizontal_flip to
    mask_transform. Otherwise, training augmentation is sampled independently.
    """
    if not isinstance(image, Image.Image):
        image = F.to_pil_image(image)
    image = image.convert("RGB")

    if train:
        if crop_params is None:
            crop_params = T.RandomResizedCrop.get_params(
                image, scale=(0.8, 1.0), ratio=(3 / 4, 4 / 3)
            )
        image = F.resized_crop(
            image, *crop_params, size, interpolation=InterpolationMode.BILINEAR
        )
        if horizontal_flip is None:
            horizontal_flip = bool(torch.rand(()) < 0.5)
        if horizontal_flip:
            image = F.hflip(image)
        image = T.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        )(image)
    else:
        image = F.resize(image, size, interpolation=InterpolationMode.BILINEAR)

    return F.normalize(F.to_tensor(image), mean=mean, std=std)


def mask_transform(
    mask: Image.Image,
    *,
    size: tuple[int, int] = (224, 224),
    crop_params: tuple[int, int, int, int] | None = None,
    horizontal_flip: bool = False,
) -> torch.Tensor:
    """Resize a semantic mask without interpolating or normalizing class IDs."""
    if not isinstance(mask, Image.Image):
        mask = F.to_pil_image(mask)
    if mask.mode not in ("L", "P", "I", "I;16"):
        raise ValueError("Segmentation masks must contain single-channel class IDs.")

    if crop_params is None:
        mask = F.resize(mask, size, interpolation=InterpolationMode.NEAREST)
    else:
        mask = F.resized_crop(
            mask, *crop_params, size, interpolation=InterpolationMode.NEAREST
        )
    if horizontal_flip:
        mask = F.hflip(mask)
    return F.pil_to_tensor(mask).squeeze(0).long()


def transforms(
    image: Image.Image,
    mask: Image.Image | None = None,
    *,
    mean: Sequence[float],
    std: Sequence[float],
    train: bool = False,
    size: tuple[int, int] = (224, 224),
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Preprocess an image, optionally pairing it with a segmentation mask.

    When a mask is present, both paths use identical random crop/flip settings.
    With mask=None, only an image tensor is returned.
    """
    if mask is None:
        return image_transform(image, mean=mean, std=std, train=train, size=size)

    if not isinstance(image, Image.Image):
        image = F.to_pil_image(image)
    if not isinstance(mask, Image.Image):
        mask = F.to_pil_image(mask)
    if image.size != mask.size:
        raise ValueError("Image and mask sizes must match before transformation.")

    crop_params = None
    horizontal_flip = False
    if train:
        crop_params = T.RandomResizedCrop.get_params(
            image, scale=(0.8, 1.0), ratio=(3 / 4, 4 / 3)
        )
        horizontal_flip = bool(torch.rand(()) < 0.5)

    transformed_image = image_transform(
        image,
        mean=mean,
        std=std,
        train=train,
        size=size,
        crop_params=crop_params,
        horizontal_flip=horizontal_flip,
    )
    transformed_mask = mask_transform(
        mask, size=size, crop_params=crop_params, horizontal_flip=horizontal_flip
    )
    return transformed_image, transformed_mask
