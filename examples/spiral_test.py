import numpy as np
import mygrad as mg
from nnfs.datasets import spiral_data
import nnfs

from helixnet import models, layers, activations, optimizers, loss
from mygrad.nnet.losses import softmax_crossentropy # <-- Import the real loss function

nnfs.init()
X, y = spiral_data(samples=100_000, classes=3)

# Build model
model = models.Sequential([
    layers.Dense(2, 256, activation=activations.ReLU),
    layers.Dense(256, 128, activation=activations.ReLU),
    layers.Dense(128, 64, activation=activations.ReLU),
    layers.Dense(64, 3, activation=lambda x: x)
])

# Use Adam, which should now work correctly because epoch_done() is called
optim = optimizers.RMSProp(0.01, 5e-5)

# Call the clean, high-level fit method
# We pass the real loss function and the raw integer labels 'y'
model.fit(X, y, loss_func=softmax_crossentropy, optimizer=optim, epochs=10, batch_size=32,
          metric=lambda x, y: np.mean(activations.softmax(x).argmax(1) == y))

# You can add evaluation logic here after the model is trained.
