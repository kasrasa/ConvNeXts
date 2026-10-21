"""Save and load reproducible experiment results."""

import json
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(path, model, optimizer, epoch, validation_metrics, metadata=None):
    """Save model and optimizer state when the notebook selects a best epoch.

    Checkpoint selection stays in the experiment notebook, not in this helper.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "validation_metrics": validation_metrics,
        "metadata": metadata or {},
    }
    torch.save(checkpoint, path)
    return path


def load_checkpoint(path, model, device="cpu", optimizer=None):
    """Load a saved checkpoint into an existing model and optionally optimizer."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    if optimizer is not None and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def save_json(data, path):
    """Write scalar metrics, arrays, or epoch history to a readable JSON file."""
    def encode(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Cannot serialize {type(value).__name__}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=encode, allow_nan=False)
    return path


def save_predictions(predictions, path):
    """Save labels, hard predictions, and probabilities for ROC/PR plots."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        labels=predictions["labels"],
        predictions=predictions["predictions"],
        probabilities=predictions["probabilities"],
    )
    return path
