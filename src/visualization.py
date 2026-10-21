import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt

def display_metrics(metrics_dict, model_name):
    """
    Displays the evaluation metrics for a given model in a datafram table format.

    Args:
        metrics_dict (dict): A dictionary containing evaluation metrics.
        model_name (str): The name of the model.
    """
    df = pd.DataFrame(metrics_dict, index=[model_name])
    print(df)


def plot_AUC_ROC(y_true, y_scores, model_name):
    """
    Plots the AUC-ROC curve for a given model.

    Args:
        y_true (array-like): True binary labels.
        y_scores (array-like): Target scores, can either be probability estimates of the positive class or confidence values.
        model_name (str): The name of the model.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    plt.show()


def plot_AUC_PR(y_true, y_scores, model_name):
    """
    Plots the AUC-PR curve for a given model.

    Args:
        y_true (array-like): True binary labels.
        y_scores (array-like): Target scores, can either be probability estimates of the positive class or confidence values.
        model_name (str): The name of the model.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="lower left")
    plt.show()