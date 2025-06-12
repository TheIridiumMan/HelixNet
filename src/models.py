from typing import List

import numpy as np
import mygrad as mg

import layers


class Sequental:
    def __init__(self, layers: List[layers.Dense]) -> None:
        self.layers = layers

    def forward(self, x: mg.tensor) -> mg.tensor:
        """Perfrom a prediction across multiple layers

        Args:
            x (mg.tensor): the input

        Returns:
            mg.tensor: the predictions
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def get_names(self):
        """Returns a list of layers names

        Returns:
            list: A list of strings
        """
        return [layer.name for layer in self.layers]
    
    def null_grads(self):
        for layer in self.layers:
            layer.weights.null_grad()
            if layer.use_bias:
                layer.bias.null_grad()
