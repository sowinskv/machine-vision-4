import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def calculate_confusion_matrix(y_true, y_pred):
    """
    Calculate confusion matrix from predictions.

    Args:
        y_true: True labels (one-hot or integer)
        y_pred: Predicted labels (one-hot or probabilities)

    Returns:
        numpy.ndarray: Confusion matrix
    """
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    return confusion_matrix(y_true, y_pred)


def generate_classification_report(y_true, y_pred, class_names):
    """
    Generate detailed classification report with per-class metrics.

    Args:
        y_true: True labels (one-hot or integer)
        y_pred: Predicted labels (one-hot or probabilities)
        class_names: List of class names

    Returns:
        dict: Classification report with precision, recall, f1-score per class
    """
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True
    )
    return report


def save_metrics_to_json(metrics, save_path):
    """
    Save metrics dictionary to JSON file.

    Args:
        metrics: Dictionary containing metrics
        save_path: Path to save JSON file
    """
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"✓ Metrics saved to {save_path}")
