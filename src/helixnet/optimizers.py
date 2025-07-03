from abc import ABC
from typing import List

import numpy as np
import mygrad as mg

from . import models
from . import layers


class Regularizer(ABC):
    """A Simple regularizer class for parameter regularization"""

    def __init__(self):
        pass

    def regularize(self, parameter: mg.Tensor) -> mg.Tensor:
        """
        :param mg.Tensor parameter: the parameter that will be regularized

        This method should be overloaded by the regularizers because this method 
            will perform the calculations
        """


class Optimizer(ABC):
    """
    An Abstract class that is used by other optimisers and it also performs
    the primary training loop

    :param float lr: The learn rate of the optimizer
    :param List[Regularizer] regularizers: The regularizers that
        will be applied to the parameters by the optimizer
    """

    def __init__(self, lr: float, regularizers: List[Regularizer] = None) -> None:
        self.step = 1
        self.regularizers = regularizers
        self.lr = lr

    def epoch_done(self):
        """
        A simple method that should be called after each epoch.

        This method should be called after every epoch is done in order to inform the optimiser to
        update it's parameters like weight decay
        """
        self.step += 1

    def optimize_param(self, parameter: mg.Tensor) -> None:
        """
        This function takes parameters one by one and must be
        inherited by the children and overload it with the update logic

        :param parameter (mg.Tensor): The parameter itself
        """

    def get_current_lr(self) -> float:
        """
        :return: The learn rate
        :rtype: float

        This method returns the learn rate with respect to the current step
        """

    def optimize(self, model: models.Sequential, loss: mg.Tensor) -> None:
        """
        This method trains models and calls optimise_param
            for every parameter in the layer and it's called 
            when the training happens
        Also if there any parameter in the model that doesn't 
            have any gradients it will skip it.

        :param (models.Sequential) model: The model that will be trained
        :param mg.Tensor loss: The loss of the model
        """
        self.lr = self.get_current_lr() if self.get_current_lr() is not None else self.lr

        # This loop made for regularization loss calculation
        if self.regularizers:
            for layer in model.layers:
                for parameter in layer.trainable_params:
                    loss += mg.sum([reg.regularize(parameter)
                                   for reg in self.regularizers])

        loss.backward()
        for layer in model.layers:
            for parameter in layer.trainable_params:
                if parameter.grad is None:
                    continue
                self.optimize_param(parameter)


class SGD(Optimizer):
    """
    Stochastic Gradient Descend is a powerful optimiser and
        is more stable than Adam numerically
    """

    def __init__(self, lr, decay=None, momentum=None,
                 regularizers: List[Regularizer] = None) -> None:
        """
        :param float lr: The learn rate of the optimiser
        :param float decay: The rate of learn rate decay can be ``None`` or
            ``False`` in order to avoid decay
        :param float momentum: The momentum but can be ``None`` or
            ``False`` in order to avoid decay
        :param list regularizers: The list which contains the regularizers
        """
        self.lr = self.init_lr = lr
        self.decay = decay
        self.momentum = momentum
        if self.momentum:
            self.momentums = {}
        super().__init__(lr, regularizers)

    def get_current_lr(self):
        """
        :return: The learn rate with decay if existed
        :rtype: float

        This method returns the learn rate with respect to the current step
        """
        return self.init_lr * \
            (1. / (1 + self.decay * self.step)) if self.decay else self.lr

    def optimize_param(self, parameter: mg.Tensor) -> None:
        """
        :param mg.Tensor model: The model that needs to be trained

        This method performs training sequential models
        """
        self.lr = self.get_current_lr()
        if self.momentum:
            if id(parameter) not in self.momentums:
                self.momentums[id(parameter)] = np.zeros_like(parameter.data)
            param_update_value = ((self.momentum * self.momentums[id(parameter)]) -
                                  (self.lr * parameter.grad))
            self.momentums[id(parameter)] = param_update_value
            parameter.data += param_update_value
        else:
            parameter.data -= self.lr * parameter.grad


class Adam(Optimizer):
    """
    Adam a very good optimiser can converge quickly but less stable numerically

    :param float lr: The learn rate of the optimiser
    :param float decay: The rate of learn rate decay can be
        ``None`` in order to avoid decay
    """

    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999,
                 regularizers: List[Regularizer] = None) -> None:
        self.lr = self.init_lr = learning_rate
        self.decay = decay

        self.epilson = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.momentum = {}
        self.cache = {}
        super().__init__(learning_rate, regularizers)

    def get_current_lr(self) -> float:
        """
        This method returns the learn rate with respect to the current step

        :return: The learn rate with decay if existed
        :rtype: float
        """
        return self.init_lr * \
            (1. / (1 + self.decay * self.step)) if self.decay else self.lr

    def optimize_param(self, parameter: mg.Tensor) -> None:
        if id(parameter) not in self.momentum:
            self.momentum[id(parameter)] = np.zeros_like(parameter.data)
            self.cache[id(parameter)] = np.zeros_like(parameter.data)

        self.momentum[id(parameter)] = (self.beta_1 *
                                        self.momentum[id(parameter)] +
                                        (1 - self.beta_1) * parameter.grad)
        param_momentum_corrected = (self.momentum[id(parameter)]
                                    / ((1 - self.beta_1)**(self.step + 1)))
        self.cache[id(parameter)] = (self.beta_2 * self.cache[id(parameter)]
                                     + (1 - self.beta_2) * parameter.grad**2)
        param_cache_corrected = (self.cache[id(parameter)]
                                 / (1 - self.beta_2 ** (self.step + 1)))
        parameter.data += -self.lr * param_momentum_corrected / \
            (np.sqrt(param_cache_corrected) + self.epilson)


class L1(Regularizer):
    """A L1 regularizer"""

    def __init__(self, lambda_: float):
        """
        :param float lambda_: The rate of penalty
        """
        self.lambda_ = lambda_
        super().__init__()

    def regularize(self, parameter: mg.Tensor) -> mg.Tensor:
        return self.lambda_ * mg.sum(mg.abs(parameter))


class L2(Regularizer):
    """A L2 regularizer"""

    def __init__(self, lambda_: float):
        """
        :param float lambda_: The rate of penalty
        """
        self.lambda_ = lambda_
        super().__init__()

    def regularize(self, parameter: mg.Tensor) -> mg.Tensor:
        return self.lambda_ * mg.sum(mg.power(parameter, 2))


class RMSProp(Optimizer):
    """Root Mean Square Propagation optimiser or for short named RMSProp"""

    def __init__(self, lr=0.001, decay=None, epsilon=1e-7, rho=0.9,
                 regularizers: List[Regularizer] = None):
        """
        :param float lr: The learning rate of optimizer
        :param float decay: The decay rate of the learning rate
            can be ``None`` to stop the decay
        :param float epsilon: The epsilon of the optimizer
        :param float rho: The rho of the optimizer
        :param List[:class:`helixnet.optimizers.Regularizer`] regularizers:
            The list which contains the regularizers that will regularize
            the parameters of the model
        """
        super().__init__(lr, regularizers)
        self.decay = decay
        self.epsilon = epsilon
        self.rho = rho
        self.cache = {}

    def get_current_lr(self):
        # Update the learning rate based on the current step/iteration
        if self.decay:
            return self.lr * (1.0 / (1.0 + self.decay * self.step))
        return self.lr

    def optimize_param(self, parameter: mg.Tensor) -> None:
        """This method contains the update logic for a single parameter."""
        # Initialize the cache for this parameter if it's the first time we've seen it.
        # Using id(parameter) is a robust way to get a unique key.
        if id(parameter) not in self.cache:
            self.cache[id(parameter)] = np.zeros_like(parameter.data)

        # --- The Core RMSProp Logic ---
        # 1. Update the cache with the exponentially weighted average of squared gradients
        self.cache[id(parameter)] = (self.rho * self.cache[id(parameter)] +
                                     (1 - self.rho) * parameter.grad**2)

        # 2. Update the parameter's data using the cache
        parameter.data += -self.lr * \
            parameter.grad / \
            (np.sqrt(self.cache[id(parameter)]) + self.epsilon)
