"""
This module contains layers.
"""
from abc import ABC
import os
from concurrent.futures import ProcessPoolExecutor
from numbers import Integral  # Max Pooling

from mygrad.operation_base import Operation
from mygrad.tensor_base import Tensor
from mygrad.nnet.layers.utils import sliding_window_view

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
                                      "Supported yet. Use pickle instead")

# Well class _MaxPoolND taken from MyGrad.nnet implemenation but it doesn't use
# Floor division so I had to create mine with floor division
class _MaxPoolND(Operation):
    def __call__(self, x, pool, stride):
        self.variables = (x,)  # data: ((N0, ...), C0, ...)
        x = x.data

        assert isinstance(pool, (tuple, list, np.ndarray)) and all(
            i >= 0 and isinstance(i, Integral) for i in pool
        )
        pool = np.asarray(pool, dtype=int)
        assert all(i > 0 for i in pool)
        assert x.ndim >= len(
            pool
        ), "The number of pooled dimensions cannot exceed the dimensionality of the data."

        stride = (
            np.array([stride] * len(pool))
            if isinstance(stride, Integral)
            else np.asarray(stride, dtype=int)
        )
        assert len(stride) == len(pool) and all(
            s >= 1 and isinstance(s, Integral) for s in stride
        )

        self.pool = pool  # (P0, ...)
        self.stride = stride  # (S0, ...)

        num_pool = len(pool)
        num_no_pool = x.ndim - num_pool

        x_shape = np.array(x.shape[num_no_pool:])
        w_shape = pool
        # MODIFIED: BY ME `//` instead of `/`
        out_shape = (x_shape - w_shape) // stride + 1

        if not all(i.is_integer() and i > 0 for i in out_shape):
            msg = "Stride and kernel dimensions are incompatible: \n"
            msg += f"Input dimensions: {(tuple(x_shape))}\n"
            msg += f"Stride dimensions: {(tuple(stride))}\n"
            msg += f"Pooling dimensions: {(tuple(w_shape))}\n"
            raise ValueError(msg)

        pool_axes = tuple(-(i + 1) for i in range(num_pool))

        # (G0, ...) is the tuple of grid-positions for placing each window (not including stride)
        # sliding_window_view(x): ((N0, ...), C0, ...)          -> (G0, ..., (N0, ...), P0, ...)
        # max-pool:               (G0, ..., (N0, ...), P0, ...) -> (G0, ..., (N0, ...))
        maxed = sliding_window_view(x, self.pool, self.stride).max(axis=pool_axes)
        axes = tuple(range(maxed.ndim))

        # (G0, ..., (N0, ...)) -> ((N0, ...), G0, ...)
        out = maxed.transpose(axes[-num_no_pool:] + axes[:-num_no_pool])
        return out if out.flags["C_CONTIGUOUS"] else np.ascontiguousarray(out)

    def backward_var(self, grad, index, **kwargs):
        """Parameters
        ----------
        grad : numpy.ndarray, shape=((N0, ...), G0, ...),
        index : int"""
        var = self.variables[index]
        x = var.data
        num_pool = len(self.pool)

        sl = sliding_window_view(x, self.pool, self.stride)
        grid_shape = sl.shape
        maxed = sl.reshape(*sl.shape[:-num_pool], -1).argmax(-1)
        axes = tuple(range(maxed.ndim))

        # argmax within a given flat-window
        maxed = maxed.transpose(
            axes[num_pool:] + axes[:num_pool]
        )  # ((N0, ...), G0, ...)

        # flat-index offset associated with reshaped window within `x`
        row_major_offset = tuple(np.cumprod(x.shape[-num_pool:][:0:-1])[::-1]) + (1,)

        # flat index of argmax, updated based on position within window, according to shape of `x`
        in_window_offset = sum(
            ind * off
            for ind, off in zip(np.unravel_index(maxed, self.pool), row_major_offset)
        )

        # flat-index of strided window placement, relative to `x`
        window_offset = sum(
            ind * s * off
            for ind, s, off in zip(
                np.indices(grid_shape[:num_pool]), self.stride, row_major_offset
            )
        )

        # indices required to traverse pool-axis-flattened array
        # ((N0, ...) G0*...)
        flat_grid_shape = (*maxed.shape[:-num_pool], np.prod(maxed.shape[-num_pool:]))
        index = np.indices(flat_grid_shape)

        # update trailing indices to traverse location of max entries within pooled axes
        index[-1] = (in_window_offset + window_offset).reshape(
            *flat_grid_shape[:-1], -1
        )

        # accumulate gradient within pool-axis-flattened dx, then reshape to match shape of `x`
        dx = np.zeros(x.shape[:-num_pool] + (np.prod(x.shape[-num_pool:]),))
        np.add.at(dx, tuple(index), grad.reshape(*x.shape[:-num_pool], -1))
        return dx.reshape(x.shape)

def _max_pool(
    x,
    pool,
    stride,
    *,
    constant = None,
) -> Tensor:
    return Tensor._op(_MaxPoolND, x, op_args=(pool, stride), constant=constant)

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
        return _max_pool(X, self.pool_size, self.stride)
