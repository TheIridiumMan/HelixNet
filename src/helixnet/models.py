from typing import List

import numpy as np
import mygrad as mg

import helixnet.layers as layers


class Sequential:
    """A Simple model that propagate through the layers in a linear way
    
    :param list layer: the list which contains the layers"""
    def __init__(self, layers: List[layers.Layer]) -> None:
        self.layers = layers

    def forward(self, x: mg.Tensor) -> mg.Tensor:
        """Perform a prediction across multiple layers

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
        """Reset the gradients of every layer"""
        for layer in self.layers:
            layer.null_grad()

    def output_shape(self):
        """A simple function that shows the model's last layer's output shape"""
        shape = tuple()
        for layer in self.layers:
            shape = layer.output_shape(shape)
        return shape

    def add(self, layer: layers.Layer):
        """This function can append layers to the model

        :param layer.Layer layer: The layer that will be appended to the end of the model"""
        self.layers.append(layer)

    def summary(self):
        """This method prints the model summary which contains the name of every layer and it's shape"""
        print("Layer", 11 * " ", "Output Shape", 10 * " ")
        print("=" * 40)
        shape = []
        for layer in self.layers:
            shape = layer.output_shape(shape)
            print(layer.name.ljust(17), ("(N, " + str(shape)[1:]))

    def predict(self, x: mg.Tensor) -> mg.Tensor:
        """This method let the model predict without building computational graph

        :param mg.Tensor x: The models input
        :return mg.Tensor: The models predictions"""
        for layer in self.layers:
            x = layer.predict(x)
        return x
