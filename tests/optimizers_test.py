# tests/optimizers_test.py

import pytest
import numpy as np
import mygrad as mg

# Import all necessary components from your framework
from helixnet import optimizers, models, layers, activations

# --- Test Fixtures ---


@pytest.fixture
def param_with_grad():
    """Provides a standard mygrad tensor with a known gradient."""
    param = mg.tensor([10.0, -20.0], constant=False)
    known_gradient = np.array([2.0, 4.0])
    # Use the dummy loss trick to apply a predictable gradient
    dummy_loss = mg.sum(param * known_gradient)
    dummy_loss.backward()
    return param


@pytest.fixture
def mock_model(param_with_grad):
    """Provides a simple Sequential model with one layer and one parameter."""
    # Create a mock layer that will hold our parameter
    mock_layer = layers.Layer(trainable_params=[param_with_grad])
    # The forward pass can be a simple identity function for testing optimizers
    mock_layer.forward = lambda x: x
    return models.Sequential([mock_layer])

# --- Tests for Base Optimizer Logic ---


def test_optimizer_regularization_logic(mock_model):
    """Ensures the base optimizer correctly adds regularization loss."""
    optim = optimizers.SGD(lr=0.1, regularizers=[optimizers.L2(lambda_=0.5)])

    # The loss is a tensor, so we can check if its value increases
    initial_loss = mg.tensor(100.0)
    param = mock_model.layers[0].trainable_params[0] # The parameter is [10, -20]

    # L2 loss should be 0.5 * sum(10^2 + (-20)^2) = 0.5 * 500 = 250
    expected_reg_loss = 250.0

    # The optimize method should modify the loss in-place before backprop
    optim.optimize(mock_model, initial_loss)

    # The final loss value should be the initial loss + regularization loss
    assert np.isclose(initial_loss.data, 100.0 + expected_reg_loss)
    assert param.grad is None # Ensure backward was called


def test_optimizer_no_regularizers(mock_model):
    """Ensures the optimizer runs without error when no regularizers are provided."""
    optim = optimizers.SGD(lr=0.1, regularizers=None) # Test the None case
    initial_loss = mg.tensor(100.0)

    # This should run without error
    optim.optimize(mock_model, initial_loss)
    assert np.isclose(initial_loss.data, 100.0) # Loss should be unchanged


def test_gradient_clipping(mock_model):
    """Tests that gradient clipping is applied correctly when the norm is exceeded."""
    # Set a clip value that is smaller than the gradient's L2 norm
    # Gradient is [2, 4], its L2 norm is sqrt(2^2 + 4^2) = sqrt(20) ~= 4.47
    grad_clip_value = 2.0
    optim = optimizers.SGD(lr=0.1, clip=grad_clip_value)
    param = mock_model.layers[0].trainable_params[0]

    optim.optimize(mock_model, mg.tensor(0.0)) # Loss value doesn't matter here

    original_grad_norm = np.sqrt(np.sum(np.array([2.0, 4.0])**2))
    scaling_factor = grad_clip_value / original_grad_norm
    expected_clipped_grad = param.grad * scaling_factor

    # The parameter's gradient should have been modified in-place
    assert np.allclose(param.grad, expected_clipped_grad)


def test_gradient_clipping_not_triggered(mock_model):
    """Tests that gradient clipping is NOT applied when the norm is within the limit."""
    param = mock_model.layers[0].trainable_params[0]
    original_grad = param.grad.copy()

    # Set a clip value larger than the gradient's L2 norm (~4.47)
    grad_clip_value = 5.0
    optim = optimizers.SGD(lr=0.1, clip=grad_clip_value)

    optim.optimize(mock_model, mg.tensor(0.0))

    # The gradient should be unchanged
    assert np.allclose(param.grad, original_grad)


def test_optimizer_skips_none_grad(mock_model):
    """Ensures the optimizer loop gracefully handles parameters with no gradient."""
    # Add a second parameter that will not have a gradient
    no_grad_param = mg.tensor([1.0, 2.0], constant=False)
    mock_model.layers[0].trainable_params.append(no_grad_param)

    optim = optimizers.SGD(lr=0.1)

    # This should execute without error
    try:
        optim.optimize(mock_model, mg.tensor(0.0))
    except Exception as e:
        pytest.fail(f"Optimizer failed to handle a parameter with no gradient: {e}")

# --- Tests for Concrete Optimizer Implementations ---


def test_sgd_one_step(param_with_grad):
    """Tests SGD with and without momentum."""
    param = param_with_grad
    initial_data = param.data.copy()
    grad = param.grad.copy()
    lr = 0.1

    # Case 1: No momentum
    optim_no_momentum = optimizers.SGD(lr=lr, momentum=None)
    optim_no_momentum.optimize_param(param)
    expected_data_no_momentum = initial_data - lr * grad
    assert np.allclose(param.data, expected_data_no_momentum)

    # Reset parameter for next test
    param.data = initial_data

    # Case 2: With momentum
    optim_with_momentum = optimizers.SGD(lr=lr, momentum=0.9)
    optim_with_momentum.optimize_param(param)
    # First step momentum update: v = m*0 - lr*g -> p += v
    expected_update = -lr * grad
    expected_data_with_momentum = initial_data + expected_update
    assert np.allclose(param.data, expected_data_with_momentum)
    # Check that the momentum cache was updated
    assert np.allclose(optim_with_momentum.momentums[id(param)], expected_update)


def test_sgd_exp_decay(param_with_grad):
    """Tests SGD learning rate decay."""
    optim = optimizers.SGD(lr=optimizers.ExpDecay(lr=0.1, decay=0.1))
    assert optim.learn_rate_obj.get_lr() == 0.1 # Step 1

    optim.epoch_done() # Increment step to 2
    expected_lr = 0.1 * (1. / (1. + 0.1 * 2)) # lr / (1 + decay*step)
    assert np.isclose(optim.learn_rate_obj.get_lr(), expected_lr)


def test_nesterov_sgd_one_step(param_with_grad):
    """Tests NesterovSGD for a single, deterministic step."""
    param = param_with_grad
    initial_data = param.data.copy()
    grad = param.grad.copy()
    lr = 0.1
    momentum = 0.9
    optim = optimizers.NesterovSGD(lr=lr, momentum=momentum)

    prev_momentum = np.zeros_like(param.data)
    current_momentum = momentum * prev_momentum - lr * grad
    update = -momentum * prev_momentum + (1 + momentum) * current_momentum
    expected_data = initial_data + update

    optim.optimize_param(param)
    assert np.allclose(param.data, expected_data)


def test_adam_one_step(param_with_grad):
    """Tests one step of the Adam optimizer."""
    param = param_with_grad
    initial_data = param.data.copy()
    grad = param.grad.copy()

    lr = 0.001
    beta_1, beta_2, epsilon = 0.9, 0.999, 1e-7
    optim = optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

    # Manual calculation for step 1
    m = (1 - beta_1) * grad
    v = (1 - beta_2) * grad**2
    m_hat = m / (1 - beta_1**optim.step)
    v_hat = v / (1 - beta_2**optim.step)
    expected_data = initial_data - lr * m_hat / (np.sqrt(v_hat) + epsilon)

    optim.optimize_param(param)
    assert np.allclose(param.data, expected_data)


def test_rmsprop_one_step(param_with_grad):
    """Tests one step of the RMSProp optimizer."""
    param = param_with_grad
    initial_data = param.data.copy()
    grad = param.grad.copy()

    lr, rho, epsilon = 0.001, 0.9, 1e-7
    optim = optimizers.RMSProp(lr=lr, rho=rho, epsilon=epsilon)

    # Manual calculation for step 1
    cache = (1 - rho) * grad**2
    update = -lr * grad / (np.sqrt(cache) + epsilon)
    expected_data = initial_data + update

    optim.optimize_param(param)
    assert np.allclose(param.data, expected_data)

# --- Tests for Regularizer Classes ---


@pytest.mark.parametrize("value", [10, -10, 0])
def test_l1_regularizer(value):
    """Tests the L1 regularizer calculation."""
    lambda_ = 0.5
    reg = optimizers.L1(lambda_=lambda_)
    param = mg.tensor([float(value)])

    expected_loss = lambda_ * np.abs(value)
    assert np.isclose(reg.regularize(param).item(), expected_loss)


@pytest.mark.parametrize("value", [10, -10, 0])
def test_l2_regularizer(value):
    """Tests the L2 regularizer calculation."""
    lambda_ = 0.5
    reg = optimizers.L2(lambda_=lambda_)
    param = mg.tensor([float(value)])

    expected_loss = lambda_ * (value**2)
    assert np.isclose(reg.regularize(param).item(), expected_loss)


def test_const_lr():
    lr = optimizers.LearnRate(0.75)
    assert lr.get_lr() == 0.75
    for i in range(100):
        lr.step_inc()
        assert lr.step == i + 1
        assert lr.get_lr() == 0.75


def test_exp_lr():
    lr = optimizers.ExpDecay(0.9, 0.002)
    assert lr.get_lr() == 0.9
    for i in range(100):
        lr.step_inc()
        assert lr.step == i + 1
    assert lr.get_lr() == 0.75


def test_linear_lr():
    lr = optimizers.LinearDecay(0.9, -0.002)
    assert lr.get_lr() == 0.9
    for i in range(100):
        lr.step_inc()
        assert lr.step == i + 1

    assert lr.get_lr() == 0.7
