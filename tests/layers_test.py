import pytest
import numpy as np
import mygrad as mg
from helixnet import layers, models, activations

# --- Test for the base Layer class and its core features ---


def test_layer_name_generation():
    """Tests that layers get unique, incrementing names."""
    # Reset the internal name counter for a clean test
    layers.names = {}

    layer1 = layers.Dense(10, 5, activation=activations.ReLU)
    assert layer1.name == "Dense 1"

    layer2 = layers.Flatten() # Flatten has no trainable params
    assert layer2.name == "Flatten" # Should not have a number

    layer3 = layers.Dense(5, 2, activation=activations.ReLU)
    assert layer3.name == "Dense 2"


def test_total_params():
    """Tests that the total parameter count is correct."""
    # FIX: Add a dummy activation function
    layer = layers.Dense(inputs=10, params=5, use_bias=True, activation=lambda x: x)
    assert layer.total_params() == 55 # Weights (10*5) + Bias (5)

    layer_no_bias = layers.Dense(inputs=10, params=5, use_bias=False, activation=lambda x: x)
    assert layer_no_bias.total_params() == 50


def test_null_grads():
    """Tests that null_grad() correctly clears gradients."""
    layer = layers.Dense(4, 2, activation=lambda x: x)
    dummy_input = mg.tensor([[1, 2, 3, 4]], dtype=float)

    # Create a gradient using the "dummy loss" trick
    dummy_loss = mg.sum(layer.forward(dummy_input))
    dummy_loss.backward()

    # Assert that gradients exist after backprop
    assert layer.weights.grad is not None
    assert layer.bias.grad is not None

    # Action: Null the gradients
    layer.null_grad()

    # Assert that gradients are now None
    assert layer.weights.grad is None
    assert layer.bias.grad is None


def test_layer_convert_array():
    case = layers.Layer([np.array([1.0, 2.0, 3.0]),
                         np.array([1.0, 2.0, 3.0])])
    for param in case.trainable_params:
        assert isinstance(param, mg.Tensor)


def test_layer_raise():
    with pytest.raises(TypeError) as exp:
        case = layers.Layer("My String", "Another String")


# --- Tests for the new v0.5.0 Saving/Loading Architecture ---


def test_get_config_and_get_set_weights():
    """
    Tests the core saving/loading methods on a Dense layer.
    This is the most important test for your new architecture.
    """
    # 1. Create the original layer
    original_layer = layers.Dense(inputs=10, params=5, activation=activations.ReLU, use_bias=True)

    # 2. Get its configuration and weights
    config = original_layer.get_config()
    config.pop("class_name")
    original_weights = original_layer.get_weights()

    # --- Assert the config (the "blueprint") ---
    assert config['inputs'] == 10
    assert config['params'] == 5
    assert config['activation'] == 'ReLU'
    assert config['use_bias'] is True

    # --- Assert the weights (the "state") ---
    assert len(original_weights) == 2 # weights and bias
    assert original_weights[0].shape == (10, 5)
    assert original_weights[1].shape == (1, 5)

    # 3. Create a NEW layer from the config
    # We manually look up the activation for the test
    config['activation'] = activations.ReLU
    new_layer = layers.Dense(**config)

    # 4. Set the weights on the new layer
    new_layer.set_weights(original_weights)

    # 5. Assert that the new layer's weights are identical to the original's
    for orig_w, new_w in zip(original_weights, new_layer.get_weights()):
        assert np.allclose(orig_w, new_w)


@pytest.fixture(autouse=True)
def reset_layer_names():
    """This fixture automatically runs before each test function."""
    layers.names = {}

# --- Tests for the Base Layer ---


def test_base_layer_predict_no_autodiff():
    """Ensures the predict method does not build a computational graph."""
    layer = layers.Dense(10, 5, activation=lambda x: x)
    dummy_input = np.ones((1, 10))

    # predict() is decorated with @mg.no_autodiff
    # We create a "loss" from its output. If the graph were built,
    # the layer's parameters would have gradients.
    output = layer.predict(dummy_input)
    loss = mg.sum(output)
    loss.backward()

    # Assert that no gradients were formed on the layer's parameters
    for param in layer.trainable_params:
        assert param.grad is None, "predict() should not create gradients!"

# --- Tests for Dense Layer ---


class TestDense:
    def test_initialization(self):
        """Tests that Dense layer weights and biases have correct shapes."""
        layer = layers.Dense(inputs=784, params=128, activation=activations.ReLU, use_bias=True)
        assert layer.weights.shape == (784, 128)
        assert layer.bias.shape == (1, 128)
        assert len(layer.trainable_params) == 2

    def test_initialization_no_bias(self):
        layer = layers.Dense(inputs=784, params=128, activation=activations.ReLU, use_bias=False)
        assert layer.weights.shape == (784, 128)
        assert not hasattr(layer, 'bias')
        assert len(layer.trainable_params) == 1

    def test_forward_shape(self):
        """Tests the output shape of the forward pass."""
        layer = layers.Dense(inputs=784, params=128, activation=activations.ReLU)
        dummy_input = mg.tensor(np.zeros((32, 784))) # Batch size of 32
        output = layer.forward(dummy_input)
        assert output.shape == (32, 128)

    def test_output_shape_method(self):
        layer = layers.Dense(inputs=784, params=128, activation=activations.ReLU)
        assert layer.output_shape() == (128,)

# --- Tests for Convolutional Layers ---


class TestConv2D:
    def test_forward_shape(self):
        """Tests Conv2D output shape with standard parameters."""
        # Input: 32 samples, 3 channels, 28x28 pixels
        layer = layers.Conv2D(input_channels=3, output_channels=16, kernel_size=3, stride=1, padding=0, activation=activations.ReLU)
        dummy_input = mg.tensor(np.zeros((32, 3, 28, 28)))
        output = layer.forward(dummy_input)

        # Expected size: (W - K + 2P)/S + 1 => (28 - 3 + 0)/1 + 1 = 26
        assert output.shape == (32, 16, 26, 26)


class TestMaxPooling2D:
    def test_forward_shape(self):
        """Tests MaxPooling2D output shape with standard parameters."""
        layer = layers.MaxPooling2D(pool_size=2, stride=2)
        dummy_input = mg.tensor(np.zeros((32, 16, 28, 28)))
        output = layer.forward(dummy_input)

        # Expected size: (W - F)//S + 1 => (28 - 2)//2 + 1 = 14
        assert output.shape == (32, 16, 14, 14)

    def test_forward_shape_uneven(self):
        """Tests floor division behavior for pooling."""
        layer = layers.MaxPooling2D(pool_size=2, stride=2)
        dummy_input = mg.tensor(np.zeros((32, 16, 27, 27))) # Uneven size
        output = layer.forward(dummy_input)

        # Expected size: (27 - 2)//2 + 1 = 12 + 1 = 13
        assert output.shape == (32, 16, 13, 13)


class TestConvTranspose2D:
    def test_forward_shape(self):
        """Tests that ConvTranspose2D correctly upsamples the input."""
        layer = layers.ConvTranspose2D(input_channels=16, output_channels=3, kernel_size=3, stride=2, activation=activations.ReLU)
        dummy_input = mg.tensor(np.zeros((32, 16, 14, 14)))
        output = layer.forward(dummy_input)

        # Expected size: H_out = (H_in - 1) * stride + K => (14-1)*2 + 3 = 26 + 3 = 29
        # With our manual padding fix:
        # H_up = (14-1)*2+1 = 27. H_out = 27 + 2*pad - K + 1 = 27 + 2*2 - 3 + 1 = 29
        assert output.shape == (32, 3, 29, 29) # This might need adjustment based on final implementation

# --- Tests for Helper and Recurrent Layers ---


class TestFlatten:
    def test_forward_shape(self):
        layer = layers.Flatten()
        dummy_input = mg.tensor(np.zeros((32, 16, 4, 4)))
        output = layer.forward(dummy_input)
        assert output.shape == (32, 16 * 4 * 4)


class TestDropout:
    def test_training_mode_forward(self):
        """Tests that forward() drops neurons and scales the output."""
        layer = layers.Dropout(proba=0.5)
        # Use integers > 1 to make scaling obvious
        dummy_input = mg.ones((10, 100)) * 2
        output = layer.forward(dummy_input)

        assert 0 in output.data, "Dropout should set some values to zero."
        # The mean of non-zero elements should be scaled up by `1 / (1-proba)`
        assert np.mean(output.data[output.data != 0]) == pytest.approx(4.0)

    def test_inference_mode_predict(self):
        """Tests that predict() does nothing and returns the identical input."""
        layer = layers.Dropout(proba=0.5)
        dummy_input = mg.tensor(np.random.randn(10, 100))
        output = layer.predict(dummy_input)

        assert np.array_equal(output.data, dummy_input.data), "predict() should be an identity function."


class TestBatchNorm:
    @pytest.fixture
    def bn_layer(self):
        # Fixture to provide a standard BatchNorm layer
        return layers.BatchNorm(input_shape=(10, 20))

    def test_training_mode_updates_stats(self, bn_layer):
        """Tests that forward() updates the running_mean and running_var."""
        initial_mean = bn_layer.running_mean.copy()
        dummy_input = mg.tensor(np.random.randn(32, 10, 20)) # Batch of 32

        bn_layer.forward(dummy_input)

        # Assert that the running stats are no longer their initial zero values
        assert not np.array_equal(bn_layer.running_mean, initial_mean)

    def test_inference_mode_uses_running_stats(self, bn_layer):
        """Tests that predict() uses running stats and works on a single sample."""
        # Pre-populate running stats to be non-trivial
        bn_layer.running_mean = np.full((10, 20), 5.0)
        bn_layer.running_var = np.full((10, 20), 2.0)

        # A single sample would cause division by zero if batch stats were used
        dummy_input = mg.tensor(np.random.randn(1, 10, 20))

        # This should execute without a division-by-zero error
        try:
            output = bn_layer.predict(dummy_input)
            assert output.shape == (1, 10, 20)
        except ValueError:
            pytest.fail("BatchNorm.predict() likely failed by using batch stats instead of running stats.")


class TestLSTMLayer:
    def test_return_sequences_behavior(self):
        """Tests that the `return_sequences` flag produces the correct output shape."""
        batch_size, seq_len, input_size, hidden_size = 8, 10, 4, 16
        dummy_input = mg.tensor(np.zeros((batch_size, seq_len, input_size)))

        # Test case: return_sequences = True
        lstm_full = layers.LSTMLayer(input_size, hidden_size, return_sequences=True)
        output_full = lstm_full.forward(dummy_input)
        assert output_full.shape == (batch_size, seq_len, hidden_size)

        # Test case: return_sequences = False
        lstm_last = layers.LSTMLayer(input_size, hidden_size, return_sequences=False)
        output_last = lstm_last.forward(dummy_input)
        assert output_last.shape == (batch_size, hidden_size)
