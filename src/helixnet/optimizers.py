from abc import ABC
from typing import List

import numpy as np
import mygrad as mg

from . import models
from . import layers


class Regularizer(ABC):
    """A Simple regularizer class for parameter regularization"""

    def regularize(self, parameter: mg.Tensor) -> mg.Tensor:
        """
        :param mg.Tensor parameter: the parameter that will be regularized

        This method should be overloaded by the regularizers because this method
            will perform the calculations
        """


class LearnRate:
    """
    A constant learn rate

    :param float lr:
    """
    def __init__(self, lr: float):
        self.lr = lr
        self.step = 0

    def step_inc(self) -> None:
        self.step += 1

    def get_lr(self) -> float:
        return self.lr


class Optimizer(ABC):
    """
    An Abstract class that is used by other optimisers and it also performs
    the primary training loop

    :param float | LearnRate lr_O: The object or float learn rate of the optimizer
    :param List[Regularizer] regularizers: The regularizers that
        will be applied to the parameters by the optimizer
    :param float grad_clip: The values which the gradients will be clipped to it. Or
        pass ``None`` in order to avoid performing gradient clipping
    """

    def __init__(self, lr_o: LearnRate, regularizers: List[Regularizer] = None,
                 grad_clip=None) -> None:
        self.step = 1
        self.regularizers = regularizers
        if isinstance(lr_o, (int, float)):
            lr_o = LearnRate(lr_o)
        self.learn_rate_obj = lr_o
        self.lr = lr_o.get_lr()
        self.grad_clip_value = grad_clip

    def epoch_done(self):
        """
        A simple method that should be called after each epoch.

        This method should be called after every epoch is done in order to inform the optimiser to
        update it's parameters like weight decay
        """
        self.step += 1
        self.learn_rate_obj.step_inc()

    def optimize_param(self, parameter: mg.Tensor) -> None:
        """
        This function takes parameters one by one and must be
        inherited by the children and overload it with the update logic

        :param parameter (mg.Tensor): The parameter itself that will optimized
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
        self.lr = self.learn_rate_obj.get_lr()

        # This loop made for regularization loss calculation
        if self.regularizers:
            for layer in model.layers:
                for parameter in layer.trainable_params:
                    loss += mg.sum([reg.regularize(parameter)
                                   for reg in self.regularizers])

        loss.backward()
        if self.grad_clip_value is not None:
            all_grads = [p.grad for layer in model.layers for p in
                         layer.trainable_params if p.grad is not None]
            if all_grads: # Ensure there are gradients to process
                # Calculate the L2 norm of all gradients combined
                norm = np.sqrt(sum(np.sum(g**2) for g in all_grads))
                if norm > self.grad_clip_value:
                    # Scale all gradients down if the norm is too high
                    scaling_factor = self.grad_clip_value / norm
                    for grad_tensor in all_grads:
                        grad_tensor *= scaling_factor

        for layer in model.layers:
            for parameter in layer.trainable_params:
                if parameter.grad is not None:
                    self.optimize_param(parameter)


class SGD(Optimizer):
    """
    Stochastic Gradient Descend is a powerful optimiser and
        is more stable than Adam numerically
    """

    def __init__(self, lr, momentum=None,
                 regularizers: List[Regularizer] = None, clip=None) -> None:
        """
        :param float|LearnRate lr: The learn rate of the optimiser
        :param float momentum: The momentum but can be ``None`` or
            ``False`` in order to avoid decay
        :param list regularizers: The list which contains the regularizers
        """
        self.momentum = momentum
        if self.momentum:
            self.momentums = {}
        super().__init__(lr, regularizers, clip)

    def optimize_param(self, parameter: mg.Tensor) -> None:
        """
        :param mg.Tensor model: The model that needs to be trained

        This method performs training sequential models
        """
        if self.momentum:
            if id(parameter) not in self.momentums:
                self.momentums[id(parameter)] = np.zeros_like(parameter.data)
            param_update_value = ((self.momentum * self.momentums[id(parameter)])
                                  - (self.lr * parameter.grad))
            self.momentums[id(parameter)] = param_update_value
            parameter.data += param_update_value
        else:
            parameter.data -= self.lr * parameter.grad


class Adam(Optimizer):
    """
    Adam a very good optimiser can converge quickly but less stable numerically

    :param float|LearnRate lr: The learn rate of the optimiser
    """

    def __init__(self, learning_rate=LearnRate(0.001), epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999,
                 regularizers: List[Regularizer] = None, clip=None) -> None:
        self.epilson = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.momentum = {}
        self.cache = {}
        super().__init__(learning_rate, regularizers, clip)

    def optimize_param(self, parameter: mg.Tensor) -> None:
        if id(parameter) not in self.momentum:
            self.momentum[id(parameter)] = np.zeros_like(parameter.data)
            self.cache[id(parameter)] = np.zeros_like(parameter.data)

        self.momentum[id(parameter)] = (self.beta_1
                                        * self.momentum[id(parameter)]
                                        + (1 - self.beta_1) * parameter.grad)

        param_momentum_corrected = (self.momentum[id(parameter)] /
                                    ((1 - self.beta_1)**(self.step)))

        self.cache[id(parameter)] = (self.beta_2 * self.cache[id(parameter)] +
                                     (1 - self.beta_2) * parameter.grad**2)

        param_cache_corrected = (self.cache[id(parameter)] /
                                 (1 - self.beta_2 ** (self.step)))

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

    def __init__(self, lr=LearnRate(0.001), epsilon=1e-7, rho=0.9,
                 regularizers: List[Regularizer] = None):
        """
        :param float|LearnRate lr: The learning rate of optimizer
        :param float epsilon: The epsilon of the optimizer
        :param float rho: The rho of the optimizer
        :param List[:class:`helixnet.optimizers.Regularizer`] regularizers:
            The list which contains the regularizers that will regularize
            the parameters of the model
        """
        super().__init__(lr, regularizers)

        self.epsilon = epsilon
        self.rho = rho
        self.cache = {}

    def optimize_param(self, parameter: mg.Tensor) -> None:
        """This method contains the update logic for a single parameter."""
        # Initialize the cache for this parameter if it's the first time we've seen it.
        # Using id(parameter) is a robust way to get a unique key.
        if id(parameter) not in self.cache:
            self.cache[id(parameter)] = np.zeros_like(parameter.data)

        # --- The Core RMSProp Logic ---
        # 1. Update the cache with the exponentially weighted average of squared gradients
        self.cache[id(parameter)] = (self.rho * self.cache[id(parameter)]
                                     + (1 - self.rho) * parameter.grad**2)

        # 2. Update the parameter's data using the cache
        parameter.data += -self.lr * \
            parameter.grad / \
            (np.sqrt(self.cache[id(parameter)]) + self.epsilon)


class NesterovSGD(Optimizer):
    """
    A Nesterov Stochastic Gradient Descend optimizer.

    :param float|LearnRate lr: The learn rate of the optimiser
    :param float momentum: The momentum value
    :param list regularizers: The list which contains the regularizers
    """

    def __init__(self, lr, momentum=0.9,
                 regularizers: List[Regularizer] = None, clip=None) -> None:
        self.lr = self.init_lr = lr

        self.momentum = momentum
        if self.momentum:
            self.momentums = {}
        super().__init__(lr, regularizers, clip)

    def optimize_param(self, parameter: mg.Tensor) -> None:
        """
        :param mg.Tensor model: The model that needs to be trained

        This method performs training sequential models
        """
        if id(parameter) not in self.momentums:
            self.momentums[id(parameter)] = np.zeros_like(parameter.data)

        # Store the previous momentum update
        prev_momentum = self.momentums[id(parameter)]

        current_momentum = self.momentum * prev_momentum - self.lr \
            * parameter.grad
        self.momentums[id(parameter)] = current_momentum

        parameter.data += -self.momentum * prev_momentum + (1 + self.momentum) * \
            current_momentum


class ExpDecay(LearnRate):
    """    
    :param float lr: The inital learn rate
    :param float decay: The decay rate of learn rate
    
    Exponential decay of the learn rate
    """

    def __init__(self, lr: float, decay: float):
        self.learn_rate = lr
        self.decay = decay
        self.step = 0

    def get_lr(self) -> float:
        return self.learn_rate * (1. / (1 + self.decay * self.step))


class LinearDecay(LearnRate):
    """
    :param float lr: The inital learn rate
    :param float decay: The decay rate of learn rate

    Linear Decay of the learn rate where :math:`{lr} = {decay} * {steps} + lr_{init}`
    """

    def __init__(self, lr: float, decay: float):
        self.b = lr
        self.m = decay
        self.step = 0

    def get_lr(self):
        return self.m * self.step + self.b

