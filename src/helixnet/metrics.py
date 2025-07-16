import mygrad as mg
import numpy as np


def precision(y_pred, y_true):
    """Calculates precision for binary classification."""
    true_positives = mg.sum((y_pred == 1) & (y_true == 1))
    false_positives = mg.sum((y_pred == 1) & (y_true == 0))
    return true_positives / (true_positives + false_positives + 1e-7)  # Epsilon for stability


def recall(y_pred, y_true):
    """Calculates recall for binary classification."""
    true_positives = mg.sum((y_pred == 1) & (y_true == 1))
    false_negatives = mg.sum((y_pred == 0) & (y_true == 1))
    return true_positives / (true_positives + false_negatives + 1e-7)


def f1_score(y_pred, y_true):
    """Calculates the F1-score."""
    p = precision(y_pred, y_true)
    r = recall(y_pred, y_true)
    return 2 * (p * r) / (p + r + 1e-7)


def top_k_accuracy(y_pred_logits, y_true, k=5):
    """Calculates top-k accuracy.
    
    Args:
        y_pred_logits (np.ndarray): The raw logit outputs from the model.
        y_true (np.ndarray): The true integer labels.
        k (int): The number of top predictions to consider.
    """
    # Get the indices of the top k predictions for each sample
    top_k_preds = np.argsort(y_pred_logits, axis=1)[:, -k:]

    # Check if the true label is in the top k predictions for each sample
    matches = mg.any(top_k_preds == y_true.reshape(-1, 1), axis=1)

    return mg.mean(matches)


def rmse(y_pred, y_true):
    """
    It's just the square root of MSE. Its advantage is that the units are the same as the target
    """
    return np.sqrt(np.mean((y_pred - y_true)**2))


def r_squared(y_pred, y_true):
    """
    A standard statistical measure that represents
    the proportion of the variance in the dependent variable that is predictable from
    the independent variables. An R² of 1 indicates that the model explains
    all the variability of the response data around its mean.
    """
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    return 1 - (ss_residual / (ss_total + 1e-7))


def accuracy(y_pred, y_true):
    """Calculates the accuracy of the models"""
    # Assumes y_pred are logits or probabilities
    predictions = np.argmax(y_pred, axis=1)
    return np.mean(predictions == y_true)
