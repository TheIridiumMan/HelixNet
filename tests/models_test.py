# tests/models_test.py

import pytest
import numpy as np
import mygrad as mg
from unittest.mock import Mock, patch, call

from helixnet import models, layers, activations, optimizers

# --- Fixtures for Reusable Test Components ---


@pytest.fixture
def simple_model_and_data():
    """Provides a basic Sequential model and compatible dummy data."""
    model = models.Sequential([
        layers.Dense(10, 32, activation=activations.ReLU),
        layers.Dense(32, 5, activation=lambda x: x) # Logits
    ])
    X = np.random.rand(16, 10) # 16 samples, 10 features
    Y = np.random.randint(0, 5, size=(16,))
    return model, X, Y


@pytest.fixture
def complex_model():
    """Provides a more complex model for shape and summary testing."""
    # This also implicitly tests that layers.names is reset by the autouse fixture
    return models.Sequential([
        layers.InputShape((1, 28, 28)),
        layers.Conv2D(1, 16, kernel_size=3), # -> (16, 26, 26)
        layers.MaxPooling2D(pool_size=2),    # -> (16, 13, 13)
        layers.Flatten(),                    # -> (16 * 13 * 13 = 2704)
        layers.Dense(2704, 128, activation=activations.ReLU),
        layers.Dense(128, 10, activation=lambda x:x)
    ])

# --- Tests for Core Sequential Model Functionality ---


class TestSequentialCore:
    def test_init_and_forward(self, simple_model_and_data):
        """Tests model initialization and the forward pass."""
        model, X, _ = simple_model_and_data
        output = model.forward(mg.tensor(X))
        assert isinstance(output, mg.Tensor)
        assert output.shape == (16, 5)

    def test_add_layer(self, simple_model_and_data):
        """Tests the .add() method."""
        model, X, _ = simple_model_and_data
        assert len(model.layers) == 2
        model.add(layers.Dense(5, 2, activation=lambda x: x))
        assert len(model.layers) == 3
        output = model.forward(mg.tensor(X))
        assert output.shape == (16, 2)

    def test_get_names(self, complex_model):
        """Tests that get_names() returns the correct list of layer names."""
        expected_names = [
            "InputShape", "Conv2D 1", "MaxPooling2D", "Flatten", "Dense 1", "Dense 2"
        ]
        assert complex_model.get_names() == expected_names

    def test_null_grads(self, simple_model_and_data):
        """Ensures the model correctly calls null_grad() on all its layers."""
        model, _, _ = simple_model_and_data
        # Mock the null_grad method on each layer
        for layer in model.layers:
            layer.null_grad = Mock()

        model.null_grads()

        for layer in model.layers:
            layer.null_grad.assert_called_once()

    def test_predict(self, simple_model_and_data):
        """Ensures the model calls predict() on all layers for inference."""
        model, X, _ = simple_model_and_data
        for layer in model.layers:
            layer.predict = Mock(side_effect=lambda x: x) # Pass through

        model.predict(mg.tensor(X))

        for layer in model.layers:
            layer.predict.assert_called_once()

    def test_get_and_set_weights(self, simple_model_and_data):
        """Tests the full get/set weights cycle."""
        model, _, _ = simple_model_and_data
        original_weights = model.get_weights()

        # Create a new, identical model
        new_model = models.Sequential([
            layers.Dense(10, 32, activation=activations.ReLU),
            layers.Dense(32, 5, activation=lambda x: x)
        ])

        # Their initial weights should be different
        assert not np.allclose(original_weights[0], new_model.get_weights()[0])

        new_model.set_weights(original_weights)

        for orig_w, new_w in zip(original_weights, new_model.get_weights()):
            assert np.allclose(orig_w, new_w)

    def test_output_shape_success(self, complex_model):
        """Tests successful shape calculation on a complex model."""
        assert complex_model.output_shape() == (10,)

    def test_output_shape_failure(self):
        """Tests that an exception is raised for incompatible layer shapes."""
        model = models.Sequential([
            layers.Dense(10, 5, activation=lambda x:x),
            layers.Dense(4, 2, activation=lambda x:x) # Incompatible input (needs 5)
        ])
        with pytest.raises(Exception, match="An Error occurred at"):
            model.output_shape()

    def test_summary(self, complex_model, capsys):
        """Tests the model summary output."""
        complex_model.summary()
        captured = capsys.readouterr()

        assert "The Model Summary" in captured.out
        assert "Conv2D 1 (Conv2D)" in captured.out
        assert "Total Parameters" in captured.out
        # Check if the total param count is correct
        # Dense1: 2704*128 + 128 = 346240
        # Dense2: 128*10 + 10 = 1290
        # Conv1: 1*16*3*3 + 16 = 160
        # Total: 346240 + 1290 + 160 = 347690
        assert "347690" in captured.out


# --- Tests for the High-Level `fit` Method ---

class TestSequentialFit:
    @pytest.fixture
    def mock_optimizer(self):
        """A mocked optimizer to track calls."""
        optimizer = Mock(spec=optimizers.Optimizer)
        # We need to define get_current_lr as it is called within the fit method.
        optimizer.get_current_lr.return_value = 0.01
        return optimizer

    def test_fit_full_batch(self, simple_model_and_data, mock_optimizer):
        """Tests fitting with batch_size=None."""
        model, X, Y = simple_model_and_data
        epochs = 5

        model.fit(X, Y, loss_func=lambda pred, true: mg.sum(pred), optimizer=mock_optimizer, epochs=epochs)

        assert mock_optimizer.optimize.call_count == epochs
        assert mock_optimizer.epoch_done.call_count == epochs

    def test_fit_mini_batch(self, simple_model_and_data, mock_optimizer):
        """Tests fitting with a specified batch_size."""
        model, X, Y = simple_model_and_data
        epochs = 3
        batch_size = 4
        num_batches = len(X) // batch_size # 16 // 4 = 4

        model.fit(X, Y, loss_func=lambda pred, true: mg.sum(pred), optimizer=mock_optimizer, epochs=epochs, batch_size=batch_size)

        assert mock_optimizer.optimize.call_count == epochs * num_batches
        assert mock_optimizer.epoch_done.call_count == epochs

    def test_fit_with_preprocessing(self, simple_model_and_data, mock_optimizer):
        """Tests that the preprocessing function is correctly applied."""
        model, X, Y = simple_model_and_data

        # This function will be applied to each batch
        def preprocessing_func(x): return x * 2

        # Patch the model's forward pass so we can inspect what it receives
        with patch.object(model, 'forward', wraps=model.forward) as mock_forward:
            model.fit(X, Y,
                      loss_func=lambda pred, true: mg.sum(pred),
                      optimizer=mock_optimizer,
                      epochs=1,
                      batch_size=len(X), # Full batch for simplicity
                      preprocessing=preprocessing_func)

            # Check that the first argument to the first call to forward() was the preprocessed X
            received_x = mock_forward.call_args[0][0]
            assert np.allclose(received_x, X * 2)

    def test_fit_with_metrics(self, simple_model_and_data, mock_optimizer, capsys):
        """Tests that metrics are calculated and displayed."""
        model, X, Y = simple_model_and_data

        # Create a mock metric function that we can track
        mock_metric = Mock(return_value=0.99)
        metrics_dict = {"Accuracy": mock_metric}

        model.fit(X, Y,
                  loss_func=lambda pred, true: mg.sum(pred),
                  optimizer=mock_optimizer,
                  epochs=1,
                  batch_size=4,
                  metrics=metrics_dict)

        # Assert that our metric function was called
        assert mock_metric.called

        # The number of calls should equal the number of batches
        num_batches = len(X) // 4
        assert mock_metric.call_count == num_batches

        # Check that the rich output contains the metric name
        captured = capsys.readouterr()
        assert "Avg Train Accuracy" in captured.out
