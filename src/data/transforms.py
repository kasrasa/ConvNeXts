import torchvision.transforms as T


def transforms(image, mask):
    """Apply synchronized transforms to an image and its segmentation mask.
    This is useful for segmentation datasets where the image and mask must be
    transformed in the same way (e.g., random crops, flips, etc.).
    """
    if not isinstance(image, T.functional.Image.Image):
        image = T.functional.to_pil_image(image)
    if not isinstance(mask, T.functional.Image.Image):
        mask = T.functional.to_pil_image(mask)

    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
        T.ToTensor(),
    ])

    image = transform(image)
    mask = transform(mask)

    return image, mask
