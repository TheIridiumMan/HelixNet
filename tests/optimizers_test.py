import pytest
import numpy as np
import mygrad as mg
from helixnet import optimizers
from unittest.mock import Mock

# --- A pytest fixture to provide a clean parameter for each test ---


@pytest.fixture
def param_with_grad():
    """Provides a real mygrad tensor with a known, predictable gradient."""
    param = mg.tensor([10.0, -20.0], constant=False)

    # The gradient we want to "set" on the parameter
    known_gradient = np.array([2.0, 4.0])

    # Use the dummy loss trick to apply this gradient
    dummy_loss = mg.sum(param * known_gradient)
    dummy_loss.backward()
    dummy_loss.null_grad() # Clean up the dummy loss's own grad

    return param

# --- Tests for your Optimizers ---


def test_nesterov_sgd_one_step(param_with_grad):
    """Tests NesterovSGD for a single, deterministic step."""
    param = param_with_grad
    initial_data = param.data.copy()
    grad = param.grad.copy()

    lr = 0.1
    momentum = 0.9
    optim = optimizers.NesterovSGD(lr, momentum=momentum)

    # --- Manually Calculate Expected Result ---
    prev_momentum = 0 # Starts at 0
    current_momentum = momentum * prev_momentum - lr * grad # [-0.2, -0.4]
    update = -momentum * prev_momentum + (1 + momentum) * current_momentum # 1.9 * [-0.2, -0.4] = [-0.38, -0.76]
    expected_data = initial_data + update # [10.0 - 0.38, -20.0 - 0.76] = [9.62, -20.76]

    # --- Action ---
    optim.optimize_param(param)

    # --- Assert ---
    assert np.allclose(param.data, expected_data)


def test_adam_step_usage(param_with_grad):
    """Tests that Adam correctly uses the `self.step` counter for bias correction."""
    param = param_with_grad
    initial_data = param.data.copy()
    grad = param.grad.copy()

    lr = 0.001
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-7
    optim = optimizers.Adam(lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

    # --- Step 1 ---
    # Manually calculate using step = 1
    m1 = (1 - beta_1) * grad
    v1 = (1 - beta_2) * grad**2
    m1_hat = m1 / (1 - beta_1**1)
    v1_hat = v1 / (1 - beta_2**1)
    expected_data_1 = initial_data - lr * m1_hat / (np.sqrt(v1_hat) + epsilon)

    optim.optimize_param(param)
    optim.epoch_done() # Increment step to 2

    assert np.allclose(param.data, expected_data_1, atol=1e-5)

    # --- Step 2 ---
    # Simulate a new gradient
    param.null_grad() # Clear the old gradient
    new_grad = np.array([-1.0, 1.0])
    dummy_loss_2 = mg.sum(param * new_grad)
    dummy_loss_2.backward()

    # Manually calculate using step = 2
    m2 = beta_1 * m1 + (1 - beta_1) * new_grad
    v2 = beta_2 * v1 + (1 - beta_2) * new_grad**2
    m2_hat = m2 / (1 - beta_1**2)
    v2_hat = v2 / (1 - beta_2**2)
    expected_data_2 = expected_data_1 - lr * m2_hat / (np.sqrt(v2_hat) + epsilon)

    optim.optimize_param(param)

    assert np.allclose(param.data, expected_data_2, atol=1e-6)
