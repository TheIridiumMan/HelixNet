"""
This module contains the activation functions 
"""
import numpy as np
import mygrad as mg

def ReLU(x: np.array) -> np.array:
    return mg.maximum(0, x)

def softmax(x: mg.tensor, axis=-1) -> mg.tensor:
    x_shifted = x - mg.max(x, axis=axis, keepdims=True)
    e_x = mg.exp(x_shifted)
    return e_x / mg.sum(e_x, axis=axis, keepdims=True)
