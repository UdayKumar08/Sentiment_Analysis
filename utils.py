"""
Common utility functions for evaluating sentiment classification models.
Includes metric computation, confusion matrix plotting, and label decoding.
"""

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(y_true, y_pred):
    """
    Computes and returns key classification metrics:
    - Accuracy
    - Precision
    - Recall
    - F1-score

    Uses weighted averaging to handle class imbalance.
    Rounds results to 4 decimal places.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        dict: Dictionary containing all four metrics
    """
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    return {
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4)
    }


def plot_confusion_matrix(y_true, y_pred, labels):
    """
    Plots a labeled confusion matrix using seaborn's heatmap.
    Helps visualize how predictions match actual labels across classes.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of class names in order (e.g., ["negative", "neutral", "positive"])
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def decode_predictions(predictions):
    """
    Decodes integer sentiment predictions into human-readable labels.

    Args:
        predictions: List or array of integer predictions (0, 1, or 2)

    Returns:
        list: Corresponding sentiment labels as strings
    """
    mapping = {0: "negative", 1: "neutral", 2: "positive"}
    return [mapping[p] for p in predictions]
