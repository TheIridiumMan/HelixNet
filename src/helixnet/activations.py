"""
This module contains the activation functions 
"""
import numpy as np
import mygrad as mg
import mygrad.nnet as nnet

def ReLU(x: np.array) -> np.array:
    return mg.maximum(0, x)

def softmax(x: mg.Tensor, axis=-1) -> mg.Tensor:
    x_shifted = x - mg.max(x, axis=axis, keepdims=True)
    e_x = mg.exp(x_shifted)
    return e_x / mg.sum(e_x, axis=axis, keepdims=True)

def sigmoid(x: mg.Tensor) -> mg.Tensor:
    return nnet.activations.sigmoid(x)

def tanh(x: mg.Tensor) -> mg.Tensor:
    return nnet.activations.tanh(x)
