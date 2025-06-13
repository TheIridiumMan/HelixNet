"""
This module contains layers.
"""
from abc import ABC
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import mygrad as mg
from mygrad import nnet
from rich import print

def _conv_worker(x_chunk: np.ndarray, filters: np.ndarray, stride: tuple, padding: int) -> np.ndarray:
    """
    A worker function that performs a convolution on a chunk of the batch.
    It takes NumPy arrays as input to be pickle-friendly.
    """
    # We create tensors inside the worker process
    x_tensor = mg.tensor(x_chunk)
    filters_tensor = mg.tensor(filters)

    # Perform the convolution
    conv_result = nnet.conv_nd(x_tensor, filters_tensor, stride=stride, padding=padding)
    return conv_result.data

class Layer(ABC):
    """
    The Base class for creating layers.
    """
    def __init__(self, type_) -> None:
        self.type = type_
        self.name = type_

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
        super().__init__("dense")
        he_stddev = np.sqrt(2. / inputs)
        self.weights = mg.tensor(np.random.randn(inputs, params) * he_stddev, constant=False)
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

class Conv2D(Layer):
    """
    A 2D convolution layer that uses multiple processes to compute the forward pass.

    Assumes input data is of shape (N, C_in, H, W):
    N: batch size
    C_in: number of input channels
    H: height of the input feature map
    W: width of the input feature map
    """
    def __init__(self, input_channels: int, output_channels: int, kernel_size,
                 stride=1, padding=0, activation=None, use_bias: bool = True):
        super().__init__("Conv2D")

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        # TODO: Add name generator

        # Xavier/Glorot initialization for weights (filters)
        # Shape: (C_out, C_in, K_H, K_W)
        weight_shape = (output_channels, input_channels, *kernel_size)
        self.weights = mg.tensor(
            np.random.randn(*weight_shape) * np.sqrt(2. / (input_channels * kernel_size[0] * kernel_size[1]))
        )

        self.use_bias = use_bias
        if self.use_bias:
            # Bias has one value per output channel
            self.bias = mg.tensor(np.zeros(output_channels))
        else:
            self.bias = None

        self.stride = stride
        self.padding = padding
        self.activation = activation

        # Use all available CPU cores by default
        self.num_workers = os.cpu_count()

    def forward(self, X: mg.Tensor) -> mg.Tensor:
        """
        Performs a forward pass, splitting the batch across multiple processes.
        """
        batch_size = X.shape[0]
        conv_result = nnet.conv_nd(X, self.weights.data, stride=self.stride,
                                   padding=self.padding)
        
        if self.use_bias:
            # Reshape bias for broadcasting: (C_out,) -> (1, C_out, 1, 1)
            conv_result = conv_result + self.bias.reshape(1, -1, 1, 1)
        return self.activation(conv_result)
        

class Flatten(Layer):
    """
    A layer to flatten the input, typically used to transition
    from convolutional to dense layers.
    """
    def __init__(self):
        super().__init__("Flatten")
        self.input_shape = None

    def forward(self, X: mg.Tensor) -> mg.Tensor:
        """
        Takes an input of shape (N, C, H, W) and flattens it
        to a shape of (N, C*H*W).
        """
        self.input_shape = X.data.shape
        # The first dimension (N, batch size) is preserved.
        # The rest of the dimensions are flattened.
        return X.reshape(self.input_shape[0], -1)

class MaxPooling2D(Layer):
    """
    A layer to perform max pooling over a 4D input (N, C, H, W).
    """
    def __init__(self, pool_size, stride=None):
        super().__init__("MaxPooling2D")
        if isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size)
        else:
            self.pool_size = pool_size

        # If stride is not specified, it defaults to the pool_size
        if stride is None:
            self.stride = self.pool_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

    def forward(self, X: mg.Tensor) -> mg.Tensor:
        """
        Applies the max pooling operation.
        """
        # `mygrad.nnet.max_pool_2d` handles the operation.
        return nnet.max_pool(X, self.pool_size, self.stride)
