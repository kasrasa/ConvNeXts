import torch
import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
import torch.nn.functional as F
import numpy as np

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples
    return epoch_loss, epoch_accuracy

def validate(model, dataloader, criterion, device):
    """validate the model on the validation or test set."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples
    return epoch_loss, epoch_accuracy


def evaluate(model, test_dataloader, device):
    """Evaluate the model with different metrics on the held out test set.
    metrics include:
      accuracy, precision, recall, F1 score, ECE, AUC_ROC, AUC_PR, confusion matrix
    """
    model.eval()
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm.tqdm(test_dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    ece = expected_calibration_error(all_probs, all_labels)
    
    # For AUC-ROC and AUC-PR, we need to handle multi-class case
    try:
        auc_roc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        auc_pr = average_precision_score(all_labels, all_probs, average='weighted')
    except ValueError:
        auc_roc = None
        auc_pr = None

    conf_matrix = confusion_matrix(all_labels, all_predictions)

    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'ece': ece,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'confusion_matrix': conf_matrix
    }

    return metrics_dict

def expected_calibration_error(probs, labels, n_bins=15):
    """Calculate the Expected Calibration Error (ECE) of the model."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find the indices of the samples that fall into the current bin
        in_bin = np.logical_and(probs.max(axis=1) > bin_lower, probs.max(axis=1) <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin] == np.argmax(probs[in_bin], axis=1))
            avg_confidence_in_bin = np.mean(probs[in_bin].max(axis=1))
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece