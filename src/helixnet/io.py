import json
import mygrad as mg
from . import models, layers, activations

LAYER_REGISTRY = {cls.__name__: cls for cls in layers.Layer.__subclasses__()}
ACTIVATION_REGISTRY = {}

for key, func in activations.__dict__.items():
    if callable(func):
        ACTIVATION_REGISTRY[func.__name__] = func


def save_model(model: models.Sequential, filepath: str):
    """
    Saves a model by cleanly separating architecture and weights.

    :param models.Sequential model: The model that will be saved
    :param str filepath: The filename of the model that should be saved in
    """
    model_data = {
        # The architecture is just a list of configs
        "architecture": [layer.get_config() for layer in model.layers],
        # The weights are just a list of numpy arrays
        "weights": [w.tolist() for layer in model.layers for w in layer.get_weights()]
    }
    with open(filepath, 'w') as f:
        json.dump(model_data, f, indent=4)


def load_model(filepath: str) -> models.Sequential:
    """
    Loads a model from its architecture and weights.

    :param str filepath: The filename of the model that will be loaded from.
    """
    with open(filepath, 'r') as f:
        model_data = json.load(f)

    # 1. Rebuild the empty model from the architecture blueprint
    loaded_layers = []
    for layer_config in model_data['architecture']:
        class_name = layer_config.pop('class_name') # Assumes get_config adds this
        LayerClass = LAYER_REGISTRY[class_name]

        # Restore the activation function from the registry
        if 'activation' in layer_config:
            layer_config['activation'] = ACTIVATION_REGISTRY.get(layer_config['activation'])

        # Create the layer!
        layer = LayerClass(**layer_config)
        loaded_layers.append(layer)

    model = models.Sequential(loaded_layers)

    # 2. Load the saved weights into the empty model
    all_weights = [mg.tensor(w) for w in model_data['weights']]
    model.set_weights(all_weights) # We can add a helper to the model for this

    return model
