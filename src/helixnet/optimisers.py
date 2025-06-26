from abc import ABC

import numpy as np
import mygrad as mg
from rich import print
from . import models
from . import layers


class Optimiser(ABC):
    """
    An Abstract class that is used by other optimisers and it also performs
    the primary training loop
    """
    def __init__(self) -> None:
        self.step = 1

    def epoch_done(self):
        """A simple method should be called after every epoch"""
        self.step += 1

    def optimise_param(self, parameter: mg.Tensor, layer: layers.Layer) -> None:
        """This function takes parameters one by one and must be
        inhereited by the children

        Args:
            parameter (mg.Tensor): The parameter itself
            layer (layers.Layer): The parent layer for accessing any attribute
        """

    def optimise(self, model: models.Sequential) -> None:
        """This method trains models and calls optimise_param
        for every parameter in the layer and it's called when the training happens

        Args:
            model (models.Sequential): The model that needs to be trained
        """
        for layer in model.layers:
            for parameter in layer.trainable_params:
                self.optimise_param(parameter, layer)


class SGD(Optimiser):
    """
    Stochastic Gradient Descend is a powerful optimiser and 
    is more stable than Adam numerically
    """
    def __init__(self, lr, decay=None, momentum=None) -> None:
        """
        :param float lr: The learn rate of the optimiser
        :param float decay: The rate of learn rate decay can be ``None`` or ``False`` in order to avoid decay
        :param float momentum: The momentum but can be ``None`` or ``False`` in order to avoid decay
        """
        self.lr = self.init_lr = lr
        self.step = 1
        self.decay = decay
        self.momentum = momentum
        if self.momentum:
            self.momentums = {}

    def get_current_lr(self):
        """
        :return: The learn rate with decay if existed
        :rtype: float
        
        This method returns the learn rate with respect to the current step
        """
        return self.init_lr * \
            (1. / (1 + self.decay * self.step)) if self.decay else self.lr

    def epoch_done(self):
        """A simple method that should be called after each epoch.
        
        This method should be called after every epoch is done in order to inform the optimiser to
        update it's parameters like weight decay
        """
        self.step += 1

    def optimise_param(self, parameter: mg.Tensor, layer: layers.Layer) -> None:
        """
        :param mg.Tensor model: The model that needs to be trained
        
        This method performs training sequential models
        """
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


class Adam(Optimiser):
    """
    Adam a very good optimiser can converge quickly but less stable numerically

    :param float lr: The learn rate of the optimiser
    :param float decay: The rate of learn rate decay can be
        ``None`` in order to avoid decay
    
    """
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
        """
        This method returns the learn rate with respect to the current step
        
        :return: The learn rate with decay if existed
        :rtype: float
        """
        return self.init_lr * \
            (1. / (1 + self.decay * self.iters)) if self.decay else self.lr
    
    def epoch_done(self) -> None:
        """This method should be called after every epoch_done is done in order to inform the optimiser to
    update it's parameters like weight decay"""
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


class RMSProp(Optimiser):
        def __init__(self, lr=0.001, decay=0.0, epsilon=1e-7, rho=0.9):
            super().__init__()
            self.learning_rate = lr
            self.decay = decay
            # ... (rest of the init is the same)
            self.cache = {}

        def optimise(self, model: models.Sequential) -> None:
            """
            This method is called once per batch. We override it to update
            the learning rate before calling the parent's optimisation loop.
            """
            # Update the learning rate based on the current step/iteration
            if self.decay:
                self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.step))

            # Call the parent's optimise method, which runs the loop over optimise_param
            super().optimise(model)

            # Increment the step counter after the batch is processed
            self.step += 1

        def optimise_param(self, parameter: mg.Tensor, layer: layers.Layer) -> None:
            """This method contains the update logic for a single parameter."""

            # First, check if the parameter has a gradient. If not, skip it.
            if parameter.grad is None:
                return

            # Initialize the cache for this parameter if it's the first time we've seen it.
            # Using id(parameter) is a robust way to get a unique key.
            if id(parameter) not in self.cache:
                self.cache[id(parameter)] = np.zeros_like(parameter.data)

            # --- The Core RMSProp Logic ---
            # 1. Update the cache with the exponentially weighted average of squared gradients
            self.cache[id(parameter)] = (self.rho * self.cache[id(parameter)] +
                                        (1 - self.rho) * parameter.grad**2)

            # 2. Update the parameter's data using the cache
            parameter.data += -self.current_learning_rate * \
                              parameter.grad / (np.sqrt(self.cache[id(parameter)]) + self.epsilon)
