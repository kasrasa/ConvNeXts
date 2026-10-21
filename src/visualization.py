"""Tables and plots for classification experiments."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    precision_recall_curve,
    roc_curve,
)


def display_metrics(metrics_by_model):
    """Display all scalar test metrics in one model-comparison table.

    Pass {"Model name": evaluate(...) result}. Confusion matrices are shown
    separately with plot_confusion_matrix, not embedded inside table cells.
    Returns the DataFrame for optional use in the notebook.
    """
    columns = {
        "loss": "Loss",
        "accuracy": "Accuracy",
        "precision_macro": "Precision (macro)",
        "recall_macro": "Recall (macro)",
        "f1_macro": "F1 (macro)",
        "precision_weighted": "Precision (weighted)",
        "recall_weighted": "Recall (weighted)",
        "f1_weighted": "F1 (weighted)",
        "roc_auc_macro_ovr": "ROC-AUC (macro OvR)",
        "average_precision_macro_ovr": "PR-AP (macro OvR)",
        "ece": "ECE",
        "num_classes_evaluated_for_auc": "Classes in AUC",
    }
    rows = {
        name: {title: metrics.get(key) for key, title in columns.items()}
        for name, metrics in metrics_by_model.items()
    }
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "Model"
    display(df.style.format(precision=4, na_rep="—"))
    return df


def plot_training_history(history, model_name):
    """Plot training/validation loss and accuracy for one training run."""
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, label in (
        (axes[0], "loss", "Loss"),
        (axes[1], "accuracy", "Accuracy"),
    ):
        ax.plot(epochs, [row[f"train_{metric}"] for row in history], label="Train")
        ax.plot(epochs, [row[f"val_{metric}"] for row in history], label="Validation")
        ax.set(xlabel="Epoch", ylabel=label, title=f"{model_name} — {label}")
        ax.legend()
        ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(confusion_matrix, model_name, class_names=None):
    """Visualize an evaluation confusion matrix without recalculating it."""
    labels = class_names if class_names is not None else None
    fig, ax = plt.subplots(figsize=(13, 11))
    ConfusionMatrixDisplay(confusion_matrix, display_labels=labels).plot(
        ax=ax, cmap="Blues", values_format="d", xticks_rotation=90,
        colorbar=False, include_values=False,
    )
    ax.set_title(f"{model_name} — Confusion matrix")
    plt.tight_layout()
    plt.show()


def plot_AUC_ROC(y_true, y_scores, model_name, class_index=None):
    """Plot micro-average multiclass ROC, or a single class one-vs-rest ROC.

    The micro-average curve differs from the macro OvR value in display_metrics.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    if y_scores.ndim != 2:
        raise ValueError("y_scores must be shaped [samples, classes].")
    if class_index is None:
        targets = np.eye(y_scores.shape[1], dtype=int)[y_true].ravel()
        scores = y_scores.ravel()
        name = "micro-average"
    else:
        targets = (y_true == class_index).astype(int)
        scores = y_scores[:, class_index]
        name = f"class {class_index} vs rest"
    if len(np.unique(targets)) != 2:
        raise ValueError("ROC curve requires positive and negative samples.")
    fpr, tpr, _ = roc_curve(targets, scores)
    score = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"{name} ROC-AUC = {score:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title=f"{model_name} — ROC")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_AUC_PR(y_true, y_scores, model_name, class_index=None):
    """Plot micro-average multiclass PR, or single class one-vs-rest PR.

    Area under this plotted curve is trapezoidal PR-AUC; the table's PR-AP
    instead uses average precision per class. They are different definitions.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    if y_scores.ndim != 2:
        raise ValueError("y_scores must be shaped [samples, classes].")
    if class_index is None:
        targets = np.eye(y_scores.shape[1], dtype=int)[y_true].ravel()
        scores = y_scores.ravel()
        name = "micro-average"
    else:
        targets = (y_true == class_index).astype(int)
        scores = y_scores[:, class_index]
        name = f"class {class_index} vs rest"
    if len(np.unique(targets)) != 2:
        raise ValueError("PR curve requires positive and negative samples.")
    precision, recall, _ = precision_recall_curve(targets, scores)
    score = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"{name} PR-AUC = {score:.4f}")
    ax.set(xlabel="Recall", ylabel="Precision", title=f"{model_name} — Precision–recall")
    ax.legend()
    plt.tight_layout()
    plt.show()
