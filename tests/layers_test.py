# tests/layers_test.py

import pytest
import numpy as np
import mygrad as mg

# Import all necessary components from your framework
from helixnet import layers, models, activations

# --- Fixtures and Setup ---


@pytest.fixture(autouse=True)
def reset_layer_names():
    """This fixture automatically runs before each test to ensure name counters are reset."""
    layers.names = {}
    yield # This allows the test to run


@pytest.fixture
def dummy_model_and_input():
    """Provides a simple model and a compatible input tensor."""
    model = models.Sequential([
        layers.Dense(10, 5, activation=lambda x: x)
    ])
    dummy_input = mg.tensor(np.random.rand(1, 10))
    return model, dummy_input

# --- Tests for the Base `Layer` Class ---


class TestBaseLayer:
    def test_name_generation(self):
        """Tests that layers get unique, incrementing names for trainable layers."""
        d1 = layers.Dense(1, 1, lambda x: x)
        f1 = layers.Flatten() # No trainable params
        d2 = layers.Dense(1, 1, lambda x: x)
        r1 = layers.Reshape((1,)) # No trainable params

        assert d1.name == "Dense 1"
        assert f1.name == "Flatten" # Should not get a number
        assert d2.name == "Dense 2"
        assert r1.name == "Reshape"

    def test_param_type_conversion(self):
        """Ensures numpy arrays passed as params are converted to mygrad tensors."""
        param_as_numpy = np.array([1., 2.])
        layer = layers.Layer(trainable_params=[param_as_numpy])
        assert isinstance(layer.trainable_params[0], mg.Tensor)

    def test_param_type_error(self):
        """Ensures a TypeError is raised for invalid parameter types."""
        with pytest.raises(TypeError, match="must be np.ndarray or mygrad.Tensor"):
            layers.Layer(trainable_params=["this is not a tensor"])

    def test_null_grads(self, dummy_model_and_input):
        """Tests that null_grad() correctly clears gradients."""
        model, dummy_input = dummy_model_and_input
        layer = model.layers[0]

        loss = mg.sum(layer.forward(dummy_input))
        loss.backward()

        assert layer.weights.grad is not None, "Gradient should exist after backward pass."
        layer.null_grad()
        assert layer.weights.grad is None, "Gradient should be None after null_grad()."

    def test_total_params(self):
        """Tests the calculation of the total number of parameters."""
        layer_with_params = layers.Dense(10, 5, activation=lambda x: x, use_bias=True)
        layer_no_params = layers.Flatten()
        assert layer_with_params.total_params() == 55 # 10*5 weights + 5 bias
        assert layer_no_params.total_params() == 0

    def test_get_weights_and_set_weights(self):
        """Tests the full get/set weights cycle."""
        original_layer = layers.Dense(10, 5, activation=lambda x: x)
        original_weights = original_layer.get_weights()

        new_layer = layers.Dense(10, 5, activation=lambda x: x)
        # Ensure new layer has different initial weights
        assert not np.allclose(original_weights[0], new_layer.get_weights()[0])

        new_layer.set_weights(original_weights)
        for original, new in zip(original_weights, new_layer.get_weights()):
            assert np.allclose(original, new)

    def test_predict_no_autodiff(self):
        """Ensures predict() does not create a computational graph."""
        layer = layers.Dense(10, 2, activation=lambda x: x)
        dummy_input = np.random.rand(1, 10)

        output = layer.predict(dummy_input)
        # If a graph were built, a backward pass from the output would create grads
        loss = mg.sum(output)
        loss.backward()

        assert layer.weights.grad is None, "predict() should not create gradients"

# --- Tests for Specific Layer Implementations ---


class TestDense:
    def test_init_with_and_without_bias(self):
        layer_with_bias = layers.Dense(10, 5, lambda x: x, use_bias=True)
        assert len(layer_with_bias.trainable_params) == 2

        layer_no_bias = layers.Dense(10, 5, lambda x: x, use_bias=False)
        assert len(layer_no_bias.trainable_params) == 1
        assert layer_no_bias.bias is None

    def test_forward_and_output_shape(self):
        layer = layers.Dense(128, 64, activation=lambda x: x)
        dummy_input = mg.tensor(np.zeros((32, 128)))
        output = layer.forward(dummy_input)
        assert output.shape == (32, 64)
        assert layer.output_shape() == (64,)


class TestConv2D:
    def test_forward_and_output_shape(self):
        layer = layers.Conv2D(3, 16, kernel_size=3, stride=1, padding=1, activation=lambda x: x)
        dummy_input = mg.tensor(np.zeros((32, 3, 28, 28)))
        output = layer.forward(dummy_input)
        # With padding=1, (28 - 3 + 2*1)/1 + 1 = 28
        assert output.shape == (32, 16, 28, 28)
        assert layer.output_shape((3, 28, 28)) == (16, 28, 28)


class TestMaxPooling2D:
    @pytest.mark.parametrize("size, expected", [(28, 14), (27, 13)])
    def test_forward_and_output_shape(self, size, expected):
        layer = layers.MaxPooling2D(pool_size=2, stride=2)
        dummy_input = mg.tensor(np.zeros((32, 16, size, size)))
        output = layer.forward(dummy_input)
        assert output.shape == (32, 16, expected, expected)
        assert layer.output_shape((16, size, size)) == (16, expected, expected)


class TestFlatten:
    def test_forward_and_output_shape(self):
        layer = layers.Flatten()
        dummy_input = mg.tensor(np.zeros((32, 16, 4, 4)))
        output = layer.forward(dummy_input)
        assert output.shape == (32, 16 * 4 * 4)
        assert layer.output_shape((16, 4, 4)) == (256,)


class TestReshape:
    def test_forward_and_output_shape(self):
        layer = layers.Reshape((2, 8, 8))
        dummy_input = mg.tensor(np.zeros((32, 128)))
        output = layer.forward(dummy_input)
        assert output.shape == (32, 2, 8, 8)
        assert layer.output_shape() == (2, 8, 8)


class TestInputShape:
    def test_shape_validation(self):
        layer = layers.InputShape((3, 28, 28), ensure_shape=True)
        correct_input = mg.tensor(np.zeros((32, 3, 28, 28)))
        incorrect_input = mg.tensor(np.zeros((32, 1, 28, 28)))

        assert layer.forward(correct_input) is correct_input # Should pass
        with pytest.raises(ValueError):
            layer.forward(incorrect_input) # Should fail

    def test_no_validation(self):
        layer = layers.InputShape((3, 28, 28), ensure_shape=False)
        incorrect_input = mg.tensor(np.zeros((32, 1, 28, 28)))
        # Should not raise an error
        try:
            layer.forward(incorrect_input)
        except ValueError:
            pytest.fail("InputShape with ensure_shape=False should not validate shape.")

    def test_output_shape(self):
        shape = (3, 28, 28)
        layer = layers.InputShape(shape)
        assert layer.output_shape() == shape


class TestDropout:
    def test_training_mode_forward(self):
        layer = layers.Dropout(proba=0.5)
        dummy_input = mg.ones((10, 100))
        output = layer.forward(dummy_input)
        # Ensure some neurons are dropped
        assert np.count_nonzero(output.data) < output.data.size
        # Ensure scaling is correct
        assert np.isclose(np.mean(output.data[output.data != 0]), 2.0)

    def test_inference_mode_predict(self):
        layer = layers.Dropout(proba=0.5)
        dummy_input = np.random.rand(10, 100)
        output = layer.predict(dummy_input)
        assert np.allclose(output, dummy_input)


class TestBatchNorm:
    @pytest.fixture
    def bn_layer(self):
        return layers.BatchNorm(input_shape=(10, 20))

    def test_training_mode_updates_stats(self, bn_layer):
        initial_mean = bn_layer.running_mean.copy()
        dummy_input = mg.tensor(np.random.randn(32, 10, 20) + 5) # Mean of 5

        bn_layer.forward(dummy_input)
        assert not np.allclose(bn_layer.running_mean, initial_mean)

    def test_inference_mode_uses_running_stats(self, bn_layer):
        bn_layer.running_mean = np.full((10, 20), 5.0)
        bn_layer.running_var = np.full((10, 20), 2.0)

        # Using a single sample would cause std=0 if using batch stats
        dummy_input = mg.tensor(np.random.randn(1, 10, 20))

        try:
            bn_layer.predict(dummy_input)
        except ValueError:
            pytest.fail("BatchNorm.predict() likely failed by using batch stats.")


class TestLSTMLayer:
    @pytest.mark.parametrize("return_sequences, expected_shape_tpl", [
        (True, (8, 10, 16)),
        (False, (8, 16))
    ])
    def test_return_sequences_behavior(self, return_sequences, expected_shape_tpl):
        batch, seq, in_feat, hidden = 8, 10, 4, 16
        lstm = layers.LSTMLayer(in_feat, hidden, return_sequences=return_sequences)
        dummy_input = mg.tensor(np.zeros((batch, seq, in_feat)))
        output = lstm.forward(dummy_input)
        assert output.shape == expected_shape_tpl


class TestEmbedding:
    def test_forward_and_output_shape(self):
        layer = layers.Embedding(vocab_size=100, dim=32)
        dummy_input = mg.tensor(np.random.randint(0, 100, size=(16, 10)))
        output = layer.forward(dummy_input)
        assert output.shape == (16, 10, 32)
        assert layer.output_shape((10,)) == (10, 32)


class TestTransposeLayers:
    def test_dense_transpose(self):
        source_layer = layers.Dense(10, 32, activation=activations.ReLU, use_bias=True)
        # Case 1: With bias
        transpose_layer = layers.DenseTranspose(source_layer, activation=activations.sigmoid, use_bias=True)
        assert transpose_layer.weight.shape == (32, 10)
        assert np.allclose(transpose_layer.weight.data, source_layer.weights.data.T)
        assert len(transpose_layer.trainable_params) == 1 # Only its own bias is trainable
        assert transpose_layer.bias is not None
        assert transpose_layer.output_shape() == (10,)

        # Case 2: Without bias
        transpose_layer_no_bias = layers.DenseTranspose(source_layer, use_bias=False)
        assert transpose_layer_no_bias.bias is None
        assert len(transpose_layer_no_bias.trainable_params) == 0

    def test_conv_transpose_2d(self):
        layer = layers.ConvTranspose2D(16, 3, kernel_size=3, stride=2, activation=lambda x: x)
        dummy_input = mg.tensor(np.zeros((32, 16, 14, 14)))
        output = layer.forward(dummy_input)

        # H_out = (H_in - 1) * stride + K - (2*pad) is a common formula
        # Here: H_up = (14-1)*2+1 = 27. Conv with pad=K-1 -> 27 + 2*2 - 3 + 1 = 29
        expected_size = 29
        assert output.shape == (32, 3, expected_size, expected_size)
