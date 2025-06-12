# src/train_cnn.py
import numpy as np
import mygrad as mg
import pandas as pd
from sklearn.model_selection import train_test_split

from rich.progress import track
from rich import print

import layers
import optimisers
import activations
import models

# --- 1. SETUP & DATA PREPARATION ---
print("[bold yellow]Loading and preparing MNIST data for CNN...[/bold yellow]")
df = pd.read_csv("K:/Redmi 9e/Data Analysis/MNIST Digits/train.csv")
train, test = train_test_split(df, test_size=0.2, random_state=42)

# --- 2. MODEL CONFIGURATION: Build the CNN ---
print("[bold yellow]Initializing CNN model...[/bold yellow]")

# Hyperparameters for the CNN
INPUT_CHANNELS = 1   # Grayscale images
CONV1_CHANNELS = 16  # Number of filters in the first conv layer
CONV2_CHANNELS = 32  # Number of filters in the second conv layer
KERNEL_SIZE = 3
POOL_SIZE = 2
DENSE_UNITS = 128
OUTPUT_CLASSES = 10

model = models.Sequental([
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

# --- 3. OPTIMIZER AND HYPERPARAMETERS ---
INITIAL_LR = 0.05
MOMENTUM = 0.9
EPOCHS = 5  # CNNs often converge faster than Dense nets
BATCH_SIZE = 64

optim = optimisers.SGD(INITIAL_LR, False, momentum=MOMENTUM)
print(f"Optimizer: SGDWithMomentum(lr={INITIAL_LR}, momentum={MOMENTUM})")

# --- 4. TRAINING LOOP ---
def batch_gen(df, batch_size):
    for start in range(0, len(df), batch_size):
        end = start + batch_size
        yield df.iloc[start:end]

print("[bold red on yellow]CNN Training has started[/]")
loss_history = []

for epoch in range(EPOCHS):
    # Optional: Learning Rate Scheduler can be added here as before

    for batch in track(batch_gen(train, BATCH_SIZE), description=f"Training CNN Epoch {epoch+1}/{EPOCHS}...", total=len(train)//BATCH_SIZE):
        # IMPORTANT: Reshape data for CNN input (N, C, H, W)
        x = batch.drop(columns="label").values.astype(np.float32) / 255
        x = mg.tensor(x.reshape(-1, INPUT_CHANNELS, 28, 28))

        y_true = batch["label"].values

        logits = model.forward(x)
        loss_value = mg.nnet.losses.softmax_crossentropy(logits, y_true)
        loss_history.append(loss_value.data.item())

        loss_value.backward()
        optim.optimise(model)
        model.null_grads()

    # --- 5. EVALUATION AT END OF EPOCH ---
    test_x_flat = test.drop(columns="label").values.astype(np.float32) / 255
    test_x_reshaped = mg.tensor(test_x_flat.reshape(-1, INPUT_CHANNELS, 28, 28))

    test_logits = model.forward(test_x_reshaped)
    predictions = np.argmax(test_logits.data, axis=1)
    accuracy = (test["label"].values == predictions).mean()

    print(f"[bold green]Epoch {epoch+1} Complete | Final Batch Loss: {loss_value.data.item():.4f} | "
          f"Test Accuracy: {accuracy:.2%}[/bold green]")

print(f"\n[bold purple on white] Final CNN Test Accuracy: {accuracy:.2%} :tada: [/]")