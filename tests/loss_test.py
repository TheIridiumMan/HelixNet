# tests/loss_test.py
import pytest
import numpy as np
import mygrad as mg

# Assuming your new loss functions are in `helixnet.loss`
from helixnet import loss

# Fixtures for common test data


@pytest.fixture
def regression_data():
    """Provides y_pred and y_true tensors for regression tests."""
    y_pred = mg.tensor([1.0, 2.5, -0.5, 4.0])
    y_true = mg.tensor([1.2, 2.8, -0.7, 3.5])
    return y_pred, y_true


@pytest.fixture
def binary_clf_data():
    """Provides sigmoid predictions and true binary labels."""
    y_pred_sigmoid = mg.tensor([0.1, 0.9, 0.6, 0.4])
    y_true = mg.tensor([0, 1, 1, 0])
    return y_pred_sigmoid, y_true

# --- Tests for Regression Losses ---


def test_huber_loss(regression_data):
    """Tests the Huber loss function."""
    y_pred, y_true = regression_data
    delta = 1.0

    error = y_true.data - y_pred.data # [-0.2, 0.3, -0.2, -0.5]

    # With delta = 1.0, all errors are "small"
    expected_loss_values = 0.5 * error**2
    expected_total_loss = np.mean(expected_loss_values)
    huber_loss = mg.mean(loss.HuberLoss(y_pred, y_true, delta=delta))
    assert np.allclose(huber_loss.data, expected_total_loss)

    # With delta = 0.25, some errors are "large"
    delta_small = 0.25
    huber_loss_small_delta = mg.mean(loss.HuberLoss(y_pred, y_true, delta=delta_small))

    small_mask = np.abs(error) <= delta_small
    large_mask = ~small_mask

    expected_small = 0.5 * error[small_mask]**2
    expected_large = delta_small * (np.abs(error[large_mask]) - 0.5 * delta_small)
    expected_loss_small_delta = np.mean(np.concatenate([expected_small, expected_large]))

    assert np.allclose(huber_loss_small_delta.data, expected_loss_small_delta)


def test_log_cosh_loss(regression_data):
    """Tests the Log-Cosh loss function."""
    y_pred, y_true = regression_data
    error = y_true.data - y_pred.data
    expected_loss = np.mean(np.log(np.cosh(error)))
    log_cosh_loss = mg.mean(loss.LogCoshLoss(y_pred, y_true))
    assert np.allclose(log_cosh_loss.data, expected_loss)

# --- Tests for Classification Losses ---


def test_hinge_loss():
    """Tests the Hinge loss for max-margin classification."""
    y_pred = mg.tensor([-0.5, 0.8, 1.2, -1.5])
    y_true = mg.tensor([-1, 1, 1, -1])  # Labels must be -1 or 1
    # Expected: max(0, 1 - y_true * y_pred)
    # [max(0, 1-0.5), max(0, 1-0.8), max(0, 1-1.2), max(0, 1-1.5)] -> [0.5, 0.2, 0, 0]
    expected_loss = np.mean([0.5, 0.2, 0.0, 0.0])
    hinge_loss = mg.mean(loss.HingeLoss(y_pred, y_true))
    assert np.allclose(hinge_loss.data, expected_loss)


def test_focal_loss(binary_clf_data):
    """Tests the Focal loss for imbalanced datasets."""
    y_pred, y_true = binary_clf_data
    gamma = 2.0
    alpha = 0.25
    p_t = y_pred.data * y_true.data + (1 - y_pred.data) * (1 - y_true.data)
    alpha_t = alpha * y_true.data + (1 - alpha) * (1 - y_true.data)
    focal_modulator = (1 - p_t)**gamma
    bce = -np.log(p_t + 1e-7)
    expected_loss = np.mean(alpha_t * focal_modulator * bce)
    focal_loss = mg.mean(loss.FocalLoss(y_pred, y_true, alpha=alpha, gamma=gamma))
    assert np.allclose(focal_loss.data, expected_loss)

# --- Tests for Metric Learning Losses ---


def test_cosine_similarity_loss():
    """Tests the Cosine Similarity loss."""
    # Same direction -> sim=1, loss=0
    loss1 = mg.mean(loss.CosineSimilarityLoss(mg.tensor([[2., 2.]]), mg.tensor([[4., 4.]])))
    assert np.allclose(loss1.data, 0.0)

    # Orthogonal -> sim=0, loss=1
    loss2 = mg.mean(loss.CosineSimilarityLoss(mg.tensor([[1., 0.]]), mg.tensor([[0., 1.]])))
    assert np.allclose(loss2.data, 1.0)

    # Opposite direction -> sim=-1, loss=2
    loss3 = mg.mean(loss.CosineSimilarityLoss(mg.tensor([[1., 1.]]), mg.tensor([[-1., -1.]])))
    assert np.allclose(loss3.data, 2.0)
