import numpy as np
import mygrad as mg
from rich import print
import models


class SGD:
    def __init__(self, lr, decay=None, momentum=None) -> None:
        self.lr = self.init_lr = lr
        self.step = 1
        self.decay = decay
        self.momentum = momentum

    def get_current_lr(self):
        # ... (this part is fine)
        return self.init_lr * \
            (1. / (1 + self.decay * self.step)) if self.decay else self.lr

    def epoch_done(self):
        """A simple method that should be called after each epoch"""
        self.step += 1

    def optimise(self, model: models.Sequental) -> None:
        """A simple optimization for Sequental Models

        Args:
            model (models.Sequental): the model
        """
        self.lr = self.get_current_lr()

        for layer in model.layers:
            # Skip update if there's no gradient (e.g., from a frozen layer)
            if layer.weights.grad is None:
                continue

            if self.momentum:
                # Initialize momentum terms if they don't exist.
                # Use np.zeros to store raw data, NOT a tensor.
                if not hasattr(layer, "weights_momentum"):
                    layer.weights_momentum = np.zeros_like(layer.weights.data)

                # CORRECT: Perform all calculations using raw .data to avoid graph-building
                weight_update_val = (
                    self.momentum * layer.weights_momentum) - (self.lr * layer.weights.grad)

                # CORRECT: Store the updated momentum as raw data
                layer.weights_momentum = weight_update_val

                # Apply the update
                layer.weights.data += weight_update_val

                # CORRECT: Handle bias logic *inside* the use_bias check
                if layer.use_bias and layer.bias.grad is not None:
                    if not hasattr(layer, "bias_momentum"):
                        layer.bias_momentum = np.zeros_like(layer.bias.data)

                    bias_update_val = (
                        self.momentum * layer.bias_momentum) - (self.lr * layer.bias.grad)
                    layer.bias_momentum = bias_update_val
                    layer.bias.data += bias_update_val

            else:
                # This is the standard (no momentum) update
                layer.weights.data -= self.lr * layer.weights.grad
                if layer.use_bias and layer.bias.grad is not None:
                    layer.bias.data -= self.lr * layer.bias.grad


class Adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999) -> None:
        self.lr = self.init_lr = learning_rate
        self.decay = decay
        self.iters = 0
        self.epilson = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def get_current_lr(self):
        return self.init_lr * \
            (1. / (1 + self.decay * self.iters)) if self.decay else self.lr

    def optimise(self, model: models.Sequental):
        self.lr = self.get_current_lr()

        for layer in model.layers:
            if not hasattr(layer, "weight_cache"):
                layer.weight_momentum = mg.zeros_like(layer.weights.data)
                layer.weight_cache = mg.zeros_like(layer.weights.data)
                layer.bias_momentum = mg.zeros_like(layer.bias.data)
                layer.bias_cache = mg.zeros_like(layer.bias.data)

            layer.weight_momentum = (self.beta_1
                                     * layer.weight_momentum
                                     + (1 - self.beta_1) * layer.weights.grad)

            layer.bias_momentum = (self.beta_1
                                   * layer.bias_momentum
                                   + (1 - self.beta_1) * layer.bias.grad)

            weight_momentum_corrected = (layer.weight_momentum /
                                         ((1 - self.beta_1)**(self.iters + 1)))

            bias_momentum_corrected = (layer.bias_momentum /
                                       ((1 - self.beta_1)**(self.iters + 1)))

            layer.weight_cache = (self.beta_2 * layer.weight_cache +
                                  (1 - self.beta_2) * layer.weights.grad**2)

            layer.bias_cache = (self.beta_2 * layer.bias_cache +
                                (1 - self.beta_2) * layer.bias.grad**2)

            weight_cache_corrected = (layer.weight_cache /
                                      (1 - self.beta_2 ** (self.iters + 1)))

            bias_cache_corrected = (layer.bias_cache /
                                    (1 - self.beta_2 ** (self.iters + 1)))

            layer.weights += mg.tensor(layer.weights.data + (-self.lr
                                                             * weight_momentum_corrected
                                                             / (mg.sqrt(weight_cache_corrected).data + self.epilson)), constant=False)

            layer.bias = mg.tensor(layer.bias.data + (-self.lr
                                                      * bias_momentum_corrected
                                                      / (np.sqrt(bias_cache_corrected) + self.epilson)), constant=False)
