from abc import ABC

import numpy as np
import mygrad as mg
from rich import print
from . import models
from . import layers


class Optimiser(ABC):
    def __init__(self) -> None:
        self.step = 1

    def epoch_done(self):
        """A simple method should be called after every epoch"""
        self.step += 1

    def optimise_param(self, parameter: mg.Tensor, layer: layers.Layer, index: int) -> None:
        """This function takes parameters one by one

        Args:
            parameter (mg.Tensor): The parameter itself
            layer (layers.Layer): The parent layer for accesing any attribute
        """

    def optimise(self, model: models.Sequental) -> None:
        """This method trains models and calls optimise_param
        for every parameter in the layer

        Args:
            model (models.Sequental): The model that needs to be trained
        """
        for layer in model.layers:
            for i, parameter in enumerate(layer.trainable_params):
                self.optimise_param(parameter, layer, i)


class SGD(Optimiser):
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

    def optimise_param(self, parameter: mg.Tensor, layer: layers.Layer,
                       i: int) -> None:
        self.lr = self.get_current_lr()
        if self.momentum:
            if not hasattr(layer, f"param_momentum_{i}"):
                setattr(layer, f"param_momentum_{i}",
                        np.zeros_like(parameter.data))
            param_update_value = ((self.momentum * getattr(layer,
                                                           f"param_momentum_{i}"))
                                  - (self.lr * parameter.grad))
            setattr(layer, f"param_momentum_{i}", param_update_value)
            parameter.data += param_update_value
        else:
            parameter.data -= self.lr * parameter.grad


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
