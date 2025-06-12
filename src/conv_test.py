import numpy as np
import mygrad as mg
from rich import print

import layers
import activations
import models

def run_cnn_test():
    """Demonstrates a simple CNN using the new Conv2D and Flatten layers."""
    print("[bold green]--- Running CNN Test ---[/bold green]")

    # 1. Create some dummy input data
    # Let's simulate a batch of 16 grayscale images of size 28x28
    batch_size = 16
    input_channels = 1
    height, width = 28, 28

    # Shape: (N, C_in, H, W)
    dummy_images = mg.tensor(np.random.rand(batch_size, input_channels, height, width))
    print(f"Input shape: {dummy_images.shape}")

    # 2. Define the CNN model using our Sequental class
    model = models.Sequental([
        # First conv layer: 1 input channel, 4 output channels, 3x3 kernel
        layers.Conv2D(input_channels=1, output_channels=4, kernel_size=3, activation=activations.ReLU),

        # Second conv layer: 4 input channels, 8 output channels, 3x3 kernel
        layers.Conv2D(input_channels=4, output_channels=8, kernel_size=3, activation=activations.ReLU),

        # Flatten the output to feed into a Dense layer
        layers.Flatten(),

        # A dense layer for classification (e.g., 10 classes like MNIST)
        # The input size to this layer depends on the output of the conv layers.
        # Output of Conv2: (N, 8, 24, 24). Flattened: 8 * 24 * 24 = 4608
        layers.Dense(inputs=8 * 24 * 24, params=10, activation=activations.softmax)
    ])

    print("\n[bold yellow]Model Architecture:[/bold yellow]")
    for layer in model.layers:
        print(f"- {layer.name}")
        if hasattr(layer, 'weights'):
            print(f"  - Weights shape: {layer.weights.shape}")

    # 3. Perform a forward pass
    print("\n[bold yellow]Performing forward pass...[/bold yellow]")
    predictions = model.forward(dummy_images)

    # 4. Check the final output
    print(f"\n[bold green]Forward pass successful![/bold green]")
    print(f"Final output shape: {predictions.shape}")
    assert predictions.shape == (batch_size, 10)

    # Check that softmax output sums to 1 for each sample in the batch
    assert np.allclose(np.sum(predictions.data, axis=1), 1.0)
    print("Softmax output is valid (rows sum to 1).")

    # You could now calculate loss and backpropagate as usual!
    # For example:
    # dummy_labels = np.random.randint(0, 10, batch_size)
    # loss = mg.nnet.losses.softmax_crossentropy(predictions, dummy_labels)
    # loss.backward()
    # print(f"\nComputed Loss: {loss.data}")
    # print("Backward pass completed.")


# The `if __name__ == "__main__"` guard is crucial for multiprocessing on some
# platforms (like Windows) to prevent infinite recursion of worker processes.
if __name__ == "__main__":
    run_cnn_test()