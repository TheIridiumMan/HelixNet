import numpy as np
import mygrad as mg
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from mygrad.computational_graph import build_graph

from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, track
from rich import print

import helixnet.layers as layers
import helixnet.optimisers as optimisers
import helixnet.activations as activations
import helixnet.models as models

print("[bold yellow]Libraries are imported loading data[/bold yellow]")
df = pd.read_csv("K:/Redmi 9e/Data Analysis/MNIST Digits/train.csv")
print("[bold yellow]Data loaded[/bold yellow]")
train, test = train_test_split(df, test_size=0.2)


# FIX 1: The final layer should have `activation=None` because softmax_crossentropy
# applies the softmax internally for better numerical stability.
lr1 = layers.Dense(784, 256, activation=activations.ReLU)
lr2 = layers.Dense(256, 256, activation=activations.ReLU)
lr3 = layers.Dense(256, 10, activation=(lambda x: x))
model = models.Sequental([lr1, lr2, lr3]) # Simplified model for faster demonstration

optim = optimisers.SGD(0.05, False, 0.9)
print("[bold yellow]layers initiated and connected successfully[/]")

def batch_gen(df, batch_size):
    for start in range(0, len(df), batch_size):
        end = start + batch_size
        yield df.iloc[start:end]

print("[bold red on yellow]The Training has started[/]")

loss_history = []
batch_len = 32
epochs = 12

for i in range(epochs):
    # FIX 2: Correct total for the progress bar
    for batch in track(batch_gen(train, batch_len), description=f"Training Epoch {i+1}/{epochs}...",
                       total=len(train)//batch_len):
        x = mg.tensor(batch.drop(columns="label").values.astype(np.float32) / 255)

        # FIX 3: softmax_crossentropy expects integer class labels, not one-hot encoded vectors.
        y_true = batch["label"].values

        # Forward pass produces logits (raw scores)
        logits = (model.forward(x))

        # The loss function takes logits and integer labels
        loss_value = mg.nnet.losses.softmax_crossentropy(logits, y_true)

        loss_history.append(loss_value.data.item())

        loss_value.backward()
        optim.optimise(model)
        # Clear grads for the next iteration
        model.null_grads()

    # Evaluate loss on the last batch of the epoch
    print(f"[bold green]Epoch {i+1} trained successfully with final batch loss [/bold green]"
          f"[bold white]{loss_value.data.item():.4f}[/bold white]")
    optim.epoch_done()

# After training, get predictions on the test set
test_x = mg.tensor(test.drop(columns="label").values.astype(np.float32) / 255)
# Pass through the model to get logits
test_logits = model.forward(test_x)
# The predicted class is the index of the highest score (argmax)
predictions = np.argmax(test_logits.data, axis=1)

accuracy = (test["label"].values == predictions).mean()

print(f"[bold purple] Final Test Accuracy: {accuracy:.2%} :tada:[/]")

plt.plot(loss_history)
plt.title("Training Loss Over Time")
plt.xlabel("Training Step")
plt.ylabel("Softmax Cross-Entropy Loss")
plt.show()