"""This module contains model creation capablities and tools"""
from typing import List, Tuple, Dict

import mygrad as mg

from helixnet import layers


class Sequential:
    """A Simple model that propagate through the layers in a linear way
    
    :param list layer: the list which contains the layers"""

    def __init__(self, layers_: List[layers.Layer]) -> None:
        self.layers = layers_

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

    def get_names(self) -> List[str]:
        """Returns a list of layers names

        Returns:
            list: A list of strings
        """
        return [layer.name for layer in self.layers]

    def null_grads(self) -> None:
        """Reset the gradients of every layer"""
        for layer in self.layers:
            layer.null_grad()

    def output_shape(self) -> Tuple[int]:
        """A simple function that shows the model's last layer's output shape"""
        shape = tuple()
        for layer in self.layers:
            shape = layer.output_shape(shape)
        return shape

    def add(self, layer: layers.Layer) -> None:
        """This function can append layers to the model

        :param layer.Layer layer: The layer that will be appended to the end of the model"""
        self.layers.append(layer)

    def summary(self) -> None:
        """
        This method prints the model summary which contains
        the name of every layer and it's shape
        """
        print("Layer", 11 * " ", "Output Shape", 10 * " ", "Total Parameters")
        print("=" * 60)
        shape = []
        tot_params = 0
        for layer in self.layers:
            shape = layer.output_shape(shape)
            params = layer.total_params()
            tot_params += params
            print(layer.name.ljust(17), ("(N, " + str(shape)[1:]).ljust(23),
                  params)
        print("=" * 60)
        print(f"Total parameters {tot_params}")

    def predict(self, x: mg.Tensor) -> mg.Tensor:
        """This method let the model predict without building computational graph

        :param mg.Tensor x: The models input
        :return mg.Tensor: The models predictions"""
        for layer in self.layers:
            x = layer.predict(x)
        return x


def save_layer(layer: layers.Layer) -> Dict[str, List[List[float]]]:
    """
    This converts the layer into dictionary.
    
    .. warning::
        It will discard the gradients of parameters

    :param layers.Layer layer: The layer that will be saved
    :return Dict[str, List[List[float]]]: The dictionary that holds information
        about the model which will be saved to JSON later
    """
    output = {}
    output["name"] = layer.name
    output["type"] = layer.type
    for i, parameter in enumerate(layer.trainable_params):
        parameter: mg.Tensor
        output[f"param_{i}"] = parameter.data.tolist()
    return output


def load_layer(struct: Dict[str, List[List[float]]],
               extra_layers: Dict[str, layers.Layer] = None) -> layers.Layer:
    """
    loads layers from dictionary that is created by :func:`helixnet.models.save_layer`
    
    :return layers.Layer: The layer that will be loaded
    :param Dict[str, List[List[float]]] layer: The dictionary that holds information
        about the model which will be loaded to :class:`helixnet.layers.Layer`
    :param Dict[str, layers.Layer] extra_layers: A dictionary that holds information
        for custom created :class:`helixnet.layers.Layer`
    """
    layer_map = layers.layers_map | dict(extra_layers)
    output = layer_map[struct["type"]]()
