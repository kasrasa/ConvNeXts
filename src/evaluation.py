"""Training loops and metrics for classification experiments."""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm.auto import tqdm


def train_one_epoch(model, dataloader, criterion, optimizer, device, head_only=False):
    """Train for one epoch; return sample-weighted loss and accuracy.

    For head_only=True, keep the frozen backbone in evaluation mode.
    For partial fine-tuning, frozen BatchNorm layers also keep their
    pretrained running statistics rather than updating them.
    """
    model.train()
    if head_only:
        model.eval()
        model.get_classifier().train()
    else:
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                if not any(param.requires_grad for param in module.parameters()):
                    module.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        correct_predictions += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    if total_samples == 0:
        raise ValueError("Training DataLoader is empty.")
    return running_loss / total_samples, correct_predictions / total_samples


def validate(model, dataloader, criterion, device):
    """Measure validation loss and accuracy without updating the model."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            correct_predictions += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

    if total_samples == 0:
        raise ValueError("Validation DataLoader is empty.")
    return running_loss / total_samples, correct_predictions / total_samples


def evaluate(model, dataloader, criterion, device, return_predictions=False):
    """Evaluate multiclass classification on a held-out dataset.

    ROC-AUC is the macro mean of valid one-vs-rest ROC AUCs.
    PR-AUC is reported as macro average precision (AP), not trapezoidal AUC.
    Classes absent from evaluation labels are excluded from these averages;
    if none is valid, the metric is None.

    Set return_predictions=True to additionally get arrays for plotting.
    """
    model.eval()
    label_batches = []
    probability_batches = []
    running_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            probabilities = F.softmax(logits, dim=1)

            running_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            label_batches.append(labels.cpu().numpy())
            probability_batches.append(probabilities.cpu().numpy())

    if total_samples == 0:
        raise ValueError("Evaluation DataLoader is empty.")

    y_true = np.concatenate(label_batches)
    y_probs = np.concatenate(probability_batches)
    y_pred = y_probs.argmax(axis=1)
    num_classes = y_probs.shape[1]
    class_ids = np.arange(num_classes)

    roc_scores = []
    ap_scores = []
    for class_id in class_ids:
        binary_labels = (y_true == class_id).astype(int)
        # Both positive and negative examples are required for ROC-AUC.
        if 0 < binary_labels.sum() < total_samples:
            roc_scores.append(roc_auc_score(binary_labels, y_probs[:, class_id]))
            ap_scores.append(average_precision_score(binary_labels, y_probs[:, class_id]))

    metrics = {
        "loss": running_loss / total_samples,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, labels=class_ids, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, labels=class_ids, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, y_pred, labels=class_ids, average="macro", zero_division=0
        ),
        "precision_weighted": precision_score(
            y_true, y_pred, labels=class_ids, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_true, y_pred, labels=class_ids, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_true, y_pred, labels=class_ids, average="weighted", zero_division=0
        ),
        "roc_auc_macro_ovr": float(np.mean(roc_scores)) if roc_scores else None,
        "average_precision_macro_ovr": float(np.mean(ap_scores)) if ap_scores else None,
        "ece": expected_calibration_error(y_probs, y_true),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=class_ids),
        "num_classes_evaluated_for_auc": len(roc_scores),
    }

    if return_predictions:
        predictions = {
            "labels": y_true,
            "predictions": y_pred,
            "probabilities": y_probs,
        }
        return metrics, predictions
    return metrics


def expected_calibration_error(probs, labels, n_bins=15):
    """Top-label expected calibration error (ECE), from 0 to 1."""
    if n_bins < 1:
        raise ValueError("n_bins must be positive.")
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)

    bin_ids = np.minimum((confidences * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for bin_index in range(n_bins):
        members = bin_ids == bin_index
        if members.any():
            ece += members.mean() * abs(
                correct[members].mean() - confidences[members].mean()
            )
    return float(ece)
