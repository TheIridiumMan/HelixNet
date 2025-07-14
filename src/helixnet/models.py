
"""This module contains model creation capablities and tools"""
from typing import List, Tuple, Dict
from functools import partial


import mygrad as mg
import numpy as np
from progress_table import ProgressTable

from helixnet import layers


class Sequential:
    """A Simple model that propagate through the layers in a linear way
    
    :param list layer: the list which contains the layers"""

    def __init__(self, layers_: list[layers.Layer]) -> None:
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

        try:
            shape = tuple()
            for layer in self.layers:
                shape = layer.output_shape(shape)
            return shape
        except Exception as e:
            raise Exception(f"An Error occurred at {layer.name=} with {shape=}\n"
                            f"Err: {e.args}")

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

    def get_weights(self) -> List[np.ndarray]:
        """Returns a flat list of all trainable weights in the model."""
        return [weight for layer in self.layers for weight in layer.get_weights()]

    def set_weights(self, weights: List[np.ndarray]):
        """
        Sets the model's weights from a flat list.

        :param List[np.ndarray] weights: The weights what will be produced by
            :func:`helixnet.io.load_model`
        """
        weight_iterator = iter(weights)
        for layer in self.layers:
            # Slicing the iterator to get the correct number of weights for this layer
            num_params = len(layer.trainable_params)
            layer_weights = [next(weight_iterator) for _ in range(num_params)]
            layer.set_weights(layer_weights)

    def fit(self, X, Y, loss_func, optimizer, epochs=1, batch_size=None, preprocessing=None,
            metric=None):
        """
        A high-level training loop for the model.
    
        :param X: Training data.
        :param Y: Training labels.
        :param loss_func: A callable loss function.
        :param optimizer: An optimizer instance.
        :param int epochs: The number of epochs to train for.
        :param int, optional batch_size: If specified, training will be done in mini-batches.
                                         If None, training is done on the full dataset at once.
        :param callable, optional preprocessing: A function to apply to input data.
        """
        table = ProgressTable(
            pbar_embedded=False,
            pbar_style_embed="rich",
            pbar_style="circle alt red blue",
            custom_cell_format=lambda val: f"{val:.4f}" if isinstance(val, float) else str(val),
            pbar_show_progress=True,
            pbar_show_percents=True,
        )

        for epoch in table(epochs, description="Epoch",
                           show_throughput=True, show_eta=True):
            table["epoch"] = epoch
            if batch_size is not None:
                # --- Mini-Batch Training Path ---
                batch_indices = np.arange(len(X))
                np.random.shuffle(batch_indices)

                for i in table(range(0, len(X), batch_size),
                               description="Batch",
                               show_throughput=True, show_eta=True):
                    batch_slice = batch_indices[i: i + batch_size]
                    x_batch = X[batch_slice]
                    y_batch = Y[batch_slice]

                    if preprocessing:
                        x_batch = preprocessing(x_batch)

                    prediction = self.forward(x_batch) # FIX: Use x_batch
                    loss = loss_func(prediction, y_batch)

                    optimizer.optimize(self, loss)
                    if metric:
                        met = metric(prediction, y_batch)
                        table.update("Additional Metrics", met.item(), aggregate="mean")
                    table.update("Training Loss", loss.item(), aggregate="mean")

            else:
                # --- Full-Batch Training Path ---
                x_processed = X if not preprocessing else preprocessing(X)
                prediction = self.forward(x_processed)
                loss = loss_func(prediction, Y)

                optimizer.optimize(self, loss)
                if metric:
                    met = metric(prediction, y_batch)
                    table.update("Additional Metrics", met.item(), aggregate="mean")
                table.update("Training Loss", loss.item(), aggregate="mean")

            optimizer.epoch_done()
            table.next_row()
        table.close() # Close the table only if it was used
