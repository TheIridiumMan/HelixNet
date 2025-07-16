# tests/metrics_test.py
import pytest
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, r2_score

from helixnet import metrics

# --- Fixtures for test data ---


@pytest.fixture
def binary_metrics_data():
    """Provides labels, predictions, and probabilities for binary classification."""
    y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1]) # TP=3, FP=1, FN=1, TN=3
    y_pred_proba = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.6, 0.3, 0.7])
    return y_true, y_pred, y_pred_proba


@pytest.fixture
def multiclass_metrics_data():
    """Provides logits and true labels for top-k accuracy."""
    y_true = np.array([0, 1, 2, 3])
    y_pred_logits = np.array([
        [0.9, 0.1, 0.0, 0.0],  # Correct
        [0.1, 0.2, 0.7, 0.0],  # Incorrect (2 is highest)
        [0.1, 0.2, 0.3, 0.4],  # Correct
        [0.9, 0.0, 0.1, 0.0],  # Incorrect (0 is highest)
    ])
    return y_true, y_pred_logits


@pytest.fixture
def regression_metrics_data():
    """Provides predictions and true values for regression metrics."""
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    return y_true, y_pred

# --- Classification Metrics Tests ---


def test_accuracy(binary_metrics_data):
    y_true, y_pred, _ = binary_metrics_data
    y_true = np.array(y_true)
    y_pred = np.array([y_pred])
    assert np.isclose(metrics.accuracy(y_pred, y_true), 0.5)


def test_precision(binary_metrics_data):
    y_true, y_pred, _ = binary_metrics_data
    # TP=3, FP=1 -> Precision = 3 / (3+1) = 0.75
    expected = precision_score(y_true, y_pred)
    assert np.isclose(metrics.precision(y_pred, y_true), expected)


def test_recall(binary_metrics_data):
    y_true, y_pred, _ = binary_metrics_data
    # TP=3, FN=1 -> Recall = 3 / (3+1) = 0.75
    expected = recall_score(y_true, y_pred)
    assert np.isclose(metrics.recall(y_pred, y_true), expected)


def test_f1_score(binary_metrics_data):
    y_true, y_pred, _ = binary_metrics_data
    expected = f1_score(y_true, y_pred)
    assert np.isclose(metrics.f1_score(y_pred, y_true), expected)


def test_top_k_accuracy(multiclass_metrics_data):
    y_true, y_pred_logits = multiclass_metrics_data
    # Top-1: [0, 2, 3, 0] -> Correct: [T, F, F, F] -> Acc = 0.25
    acc_k1 = metrics.top_k_accuracy(y_pred_logits, y_true, k=1)
    assert np.isclose(acc_k1, 0.25)

    # Top-2: [[0,1], [1,2], [2,3], [0,2]]
    # True in Top-2: [T, T, T, F] -> Acc = 0.75
    acc_k2 = metrics.top_k_accuracy(y_pred_logits, y_true, k=2)
    assert np.isclose(acc_k2, 0.75)

# --- Regression Metrics Tests ---


def test_rmse(regression_metrics_data):
    y_true, y_pred = regression_metrics_data
    expected = np.sqrt(mean_squared_error(y_true, y_pred))
    assert np.isclose(metrics.rmse(y_pred, y_true), expected)


def test_r_squared(regression_metrics_data):
    y_true, y_pred = regression_metrics_data
    expected = r2_score(y_true, y_pred)
    assert np.isclose(metrics.r_squared(y_pred, y_true), expected)
