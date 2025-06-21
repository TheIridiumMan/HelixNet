from helixnet import *

INPUT_CHANNELS = 1   # Grayscale images
CONV1_CHANNELS = 16  # Number of filters in the first conv layer
CONV2_CHANNELS = 32  # Number of filters in the second conv layer
KERNEL_SIZE = 3
POOL_SIZE = 2
DENSE_UNITS = 128
OUTPUT_CLASSES = 10

model = models.Sequential([
    layers.InputShape([1, 28, 28]),
    # Input shape: (N, 1, 28, 28)
    layers.Conv2D(input_channels=INPUT_CHANNELS, output_channels=CONV1_CHANNELS, kernel_size=KERNEL_SIZE, activation=activations.ReLU),
    # Output shape: (N, 16, 26, 26)

    layers.MaxPooling2D(pool_size=POOL_SIZE),
    # Output shape: (N, 16, 13, 13)

    layers.Conv2D(input_channels=CONV1_CHANNELS, output_channels=CONV2_CHANNELS, kernel_size=KERNEL_SIZE, activation=activations.ReLU),
    # Output shape: (N, 32, 11, 11)

    layers.MaxPooling2D(pool_size=POOL_SIZE),
    # Output shape: (N, 32, 5, 5) -> The '11' becomes '5' due to integer division in pooling

    layers.Flatten(),
    # Output shape: (N, 32 * 5 * 5) -> (N, 800)

    layers.Dense(inputs=CONV2_CHANNELS * 5 * 5, params=DENSE_UNITS, activation=activations.ReLU),
    # Output shape: (N, 128)

    layers.Dense(inputs=DENSE_UNITS, params=OUTPUT_CLASSES, activation=(lambda x: x)) # Output logits
    # Output shape: (N, 10)
])

print(f"The Final shape {model.output_shape()}")

model.summary()
