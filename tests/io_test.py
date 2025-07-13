import pytest
import numpy as np
import mygrad as mg
from pathlib import Path
from helixnet import models, layers, activations, io


def test_save_and_load_model_end_to_end(tmp_path: Path):
    """
    Tests the full save/load cycle to ensure a model is perfectly reconstructed.
    `tmp_path` is a built-in pytest fixture that creates a temporary directory.
    """
    # 1. Define and build the original model
    original_model = models.Sequential([
        layers.Dense(10, 32, activation=activations.ReLU),
        layers.Dropout(0.5),
        layers.Dense(32, 5, activation=activations.sigmoid)
    ])

    # Store its original weights for later comparison
    original_weights = [w.copy() for w in original_model.get_weights()]

    # 2. Save the model to a temporary file
    model_path = tmp_path / "test_model.json"
    io.save_model(original_model, str(model_path))

    # Assert that the file was actually created
    assert model_path.is_file()

    # 3. Load the model back from the file
    loaded_model = io.load_model(str(model_path))

    # --- 4. Assertions ---

    # Assert the architecture is the same
    assert isinstance(loaded_model, models.Sequential)
    assert len(loaded_model.layers) == len(original_model.layers)
    assert isinstance(loaded_model.layers[0], layers.Dense)
    assert isinstance(loaded_model.layers[1], layers.Dropout)
    assert isinstance(loaded_model.layers[2], layers.Dense)
    assert loaded_model.layers[1].proba == 0.5 # Check hyperparameters

    # Assert the weights are identical
    loaded_weights = loaded_model.get_weights()
    assert len(original_weights) == len(loaded_weights)
    for orig_w, loaded_w in zip(original_weights, loaded_weights):
        assert np.allclose(orig_w, loaded_w)
