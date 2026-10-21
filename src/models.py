import timm

def get_classification_model(model_name: str, pretrained: bool = True, num_classes: int = 1000):
    """Load a ConvNeXt or Swin model from timm."""
    if model_name not in timm.list_models():
        raise ValueError(f"Unknown model name: {model_name}")
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)