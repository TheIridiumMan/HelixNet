from typing import List

import numpy as np
import mygrad as mg

import helixnet.layers as layers


class Sequental:
    def __init__(self, layers: List[layers.Layer]) -> None:
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
            layer.null_grad()

    def output_shape(self):
        """A simple function that shows the models output shape"""
        shape = []
        for layer in self.layers:
            shape = layer.output_shape(shape)
        return shape
    
    def add(self, layer: layers.Layer):
        """This function can append layers to the model"""
        self.layers.append(layer)
