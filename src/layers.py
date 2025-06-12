"""
This module contains layers.
"""
from abc import ABC

import numpy as np
import mygrad as mg
from rich import print


class Layer(ABC):
    """
    The Base class for creating layers.
    """
    def __init__(self, type_) -> None:
        self.type = type_

    def __call__(self, *args, **kwargs) -> mg.Tensor:
        """
        This operator should perform a forward propagation
        """
        self.forward(*args, **kwargs)
    def forward(self, X: mg.Tensor) -> mg.Tensor:
        """
        The forward propagation
        """

class Dense(Layer):
    """A basic dense layer

    Args:
        Layer (inputs): the size of inputs
        Layer (params): the number of parameters
        Layer (activation): The activation function with `()` operator
        Layer (use_bias): whether to include bias or not

    """
    def __init__(self, inputs: int, params: int, activation,
    use_bias: bool = True, dtype = mg.float32) -> None:
        super().__init__("Dense")
        self.weights = mg.tensor(np.random.rand(inputs, params), constant=False,
                                 dtype=dtype)
        self.use_bias = use_bias
        self.activation = activation
        if self.use_bias:
            self.bias = mg.tensor(np.random.randn(1, params), constant=False,
                                  dtype=dtype)


    def forward(self, X: np.array):
        """Perform a forward propagation

        Args:
            X (np.array): the inputs

        Returns:
            mg.tensor: the predictions with gradients
        """
        return (self.activation(mg.matmul(X, self.weights)) \
            + self.bias) if self.use_bias else \
            mg.matmul(X, self.weights)

    def to_dict(self, inference_only=False) -> dict:
        res = dict()
        res["weights"] = self.weights
        res["func"] = self.activation.__name__
        if self.use_bias:
            res["bias"] = self.bias
        if inference_only:
            return res
        else:
            raise NotImplementedError("Creating model with training data is not"
                                      "Supported yet. Use pickle")
