
"""This module contains model creation capablities and tools"""
from typing import List, Tuple, Dict
from functools import partial


import mygrad as mg
import numpy as np

from typing import List, Tuple, Dict, Callable, Optional # Make sure these are present
from rich.live import Live
from rich.progress import (Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn,
                           TaskProgressColumn, track)
from rich.table import Table
from rich.console import Group
from rich.panel import Panel


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

            # Add these imports at the top of your models.py file

            # ... inside your Sequential class ...

    def fit(self, X, Y, loss_func: Callable, optimizer, epochs: int = 1,
            batch_size: Optional[int] = None,
            preprocessing: Optional[Callable] = None,
            metrics: Optional[Dict[str, Callable]] = None):
        """
        A high-level training loop with a rich, interactive display.
        """
        # --- Full-Batch Training (Simpler UI) ---
        if batch_size is None:
            # ... (This part was correct and can remain the same) ...
            print("[bold cyan]Starting full-batch training...[/bold cyan]")
            for epoch in track(range(epochs), description="Training epochs"):
                x_processed = X if not preprocessing else preprocessing(X)
                prediction = self.forward(x_processed)
                loss = loss_func(prediction, Y)
                optimizer.optimize(self, loss)
                if epoch % 100 == 0 or epoch == epochs - 1:
                    log_line = f"Epoch: {epoch}, Loss: {loss.item():.4f}"
                    if metrics:
                        for name, func in metrics.items():
                            metric_val = func(prediction.data, Y)
                            log_line += f", {name.title()}: {metric_val:.4f}"
                    print(log_line)
                optimizer.epoch_done()
            print("[bold green]Full-batch training complete.[/bold green]")
            return

        overall_progress = Progress(
            TextColumn("[bold red]Overall Progress", justify="right"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            "•",
            TimeRemainingColumn(), # Stable ETA for the entire training run
        )

        batch_progress = Progress(
            # THE FIX: We use a simple, generic description column.
            # We will update its content dynamically every batch.
            TextColumn("{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            "•",
            TimeRemainingColumn(), # ETA for the current epoch
        )

            # Define the results table
        results_table = Table(title="Training Log", show_header=True, header_style="bold magenta")
        results_table.add_column("Epoch", justify="right", style="cyan")
        results_table.add_column("Avg Train Loss", justify="center", style="green")
        if metrics:
            for name in metrics.keys():
                results_table.add_column(f"Avg Train {name.replace('_', ' ').title()}", justify="center")

        # Group UI elements inside a Panel
        render_group = Panel(
           Group(results_table, batch_progress, overall_progress),
           title="[bold yellow]HelixNet Training[/bold yellow]",
           border_style="blue"
            )

        with Live(render_group, refresh_per_second=10, vertical_overflow="visible") as live:
            num_batches_per_epoch = (len(X) + batch_size - 1) // batch_size
            total_batches = num_batches_per_epoch * epochs
            overall_task = overall_progress.add_task("Batches", total=total_batches)

            for epoch in range(epochs):
                epoch_loss_total = 0.0
                epoch_metrics_totals = {name: 0.0 for name in metrics.keys()} if metrics else {}

                # Add the batch task with a simple initial description
                batch_task = batch_progress.add_task(f"Epoch {epoch+1}", total=num_batches_per_epoch)

                batch_indices = np.arange(len(X))
                np.random.shuffle(batch_indices)

                for i in range(num_batches_per_epoch):
                    start, end = i * batch_size, (i + 1) * batch_size
                    # ... (batch creation logic) ...
                    x_batch, y_batch = X[start:end], Y[start:end]

                    # ... (forward pass, optimizer.optimize, etc.) ...
                    prediction = self.forward(x_batch)
                    loss = loss_func(prediction, y_batch)
                    optimizer.optimize(self, loss)

                    current_loss = loss.item()
                    epoch_loss_total += current_loss
                    if metrics:
                        for name, func in metrics.items():
                            epoch_metrics_totals[name] += func(prediction.data, y_batch)

                    # THE FIX: Construct the full description string and update it directly.
                    # This is the most reliable method.
                    live_description = (f"[bold cyan]Epoch {epoch+1}[/bold cyan] "
                                        f"[green]Loss: {current_loss:.4f}[/green]")

                    batch_progress.update(batch_task, advance=1, description=live_description)
                    overall_progress.update(overall_task, advance=1)

                # --- After an epoch is complete ---
                avg_loss = epoch_loss_total / num_batches_per_epoch
                row_data = [f"{epoch + 1}", f"{avg_loss:.4f}"]
                if metrics:
                    for name in metrics.keys():
                        avg_metric = epoch_metrics_totals[name] / num_batches_per_epoch
                        row_data.append(f"{avg_metric:.4f}")

                results_table.add_row(*row_data)

                # Remove the completed batch task
                batch_progress.remove_task(batch_task)

                optimizer.epoch_done()
