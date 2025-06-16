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
            layer (layers.Layer): The parent layer for accessing any attribute
        """

    def optimise(self, model: models.Sequental) -> None:
        """This method trains models and calls optimise_param
        for every parameter in the layer

        Args:
            model (models.Sequental): The model that needs to be trained
        """
        for layer in model.layers:
            for parameter in layer.trainable_params:
                self.optimise_param(parameter, layer)


class SGD(Optimiser):
    def __init__(self, lr, decay=None, momentum=None) -> None:
        self.lr = self.init_lr = lr
        self.step = 1
        self.decay = decay
        self.momentum = momentum
        if self.momentum:
            self.momentums = {}

    def get_current_lr(self):
        # ... (this part is fine)
        return self.init_lr * \
            (1. / (1 + self.decay * self.step)) if self.decay else self.lr

    def epoch_done(self):
        """A simple method that should be called after each epoch"""
        self.step += 1

    def optimise_param(self, parameter: mg.Tensor, layer: layers.Layer) -> None:
        if parameter.grad is None:
            return  # Skip this parameter if it has no gradient
        self.lr = self.get_current_lr()
        if self.momentum:
            if id(parameter) not in self.momentums:
                self.momentums[id(parameter)] = np.zeros_like(parameter.data)
            param_update_value = ((self.momentum * self.momentums[id(parameter)])
                                  - (self.lr * parameter.grad))
            self.momentums[id(parameter)] = param_update_value
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
        self.momentum = {}
        self.cache = {}

    def get_current_lr(self) -> float:
        return self.init_lr * \
            (1. / (1 + self.decay * self.iters)) if self.decay else self.lr
    
    def epoch_done(self) -> None:
        """Should be called after every parameter for updating it's internal values"""
        self.iters += 1

    def optimise_param(self, parameter: mg.Tensor, layer: layers.Layer) -> None:
        self.lr = self.get_current_lr()
        if parameter.grad is None:
            return  # Skip this parameter if it has no gradient
        if id(parameter) not in self.momentum:
            self.momentum[id(parameter)] = np.zeros_like(parameter.data)
            self.cache[id(parameter)] = np.zeros_like(parameter.data)

        self.momentum[id(parameter)] = (self.beta_1
                                        * self.momentum[id(parameter)]
                                        + (1 - self.beta_1) * parameter.grad)
        param_momentum_corrected = (self.momentum[id(parameter)] /
                                    ((1 - self.beta_1)**(self.iters + 1)))
        self.cache[id(parameter)] = (self.beta_2 * self.cache[id(parameter)] +
                                     (1 - self.beta_2) * parameter.grad**2)
        param_cache_corrected = (self.cache[id(parameter)] /
                                 (1 - self.beta_2 ** (self.iters + 1)))
        parameter.data += -self.lr * param_momentum_corrected / \
            (np.sqrt(param_cache_corrected) + self.epilson)
