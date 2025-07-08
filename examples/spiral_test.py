import numpy as np
import mygrad as mg
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

from rich.progress import track
from rich import print

import nnfs
from nnfs.datasets.spiral import create_data

import helixnet.layers as layers
import helixnet.optimizers as optimizers
import helixnet.activations as activations
import helixnet.models as models

nnfs.init()
X, y = create_data(1000, 3)
# For CatCrossEntropy, we DO need one-hot encoding
encoder = OneHotEncoder(sparse_output=False)
y_enc = encoder.fit_transform(y.reshape(-1, 1))

X = mg.tensor(X)
y_enc = mg.tensor(y_enc)

lr1 = layers.Dense(2, 128, activation=activations.ReLU, dtype=mg.float64)
lr2 = layers.Dense(128, 64, activation=activations.ReLU, dtype=mg.float64)
lr3 = layers.Dense(64, 64, activation=activations.ReLU, dtype=mg.float64)
lr4 = layers.Dense(64, 3, activation=(lambda x: x), dtype=mg.float64)
model = models.Sequential([lr1, lr2, lr3, lr4])
optim = optimizers.Adam(0.01)

losses = []
accs = []
try:
    for i in track(range(2000), description="Training..."):
        # The forward pass now produces probabilities because of softmax
        pred_probs = model.forward(X)

        # FIX 2: Now the loss function receives valid probabilities
        loss_val = mg.nnet.losses.softmax_crossentropy(pred_probs,
                                                       y)
        losses.append(loss_val.data.item())

        # Accuracy calculation
        accuracy = (np.argmax(pred_probs.data, axis=1) == y).mean()
        accs.append(accuracy)

        if i % 100 == 0:
            print(f"[bold yellow] Iteration: {i} | Loss: {loss_val.data.item():.4f}"
                  f" | Acc: {accuracy:.2%}[/]")

        optim.optimize(model, loss_val)
except KeyboardInterrupt:
    pass

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss', color=color)
ax1.plot(losses, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Accuracy', color=color)
ax2.plot(accs, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title("Training Loss and Accuracy")
plt.show()
