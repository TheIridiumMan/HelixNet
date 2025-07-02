[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A small, extensible, and lightweight framework for building powerful neural networks in Python, built on `mygrad`.

HelixNet is designed to be a transparent and easy-to-understand tool for learning and experimentation. It is powerful enough to run complex models like CNNs and LSTMs, but simple enough to run efficiently on modest hardware.

---

## Key Features

*   **Lightweight & Simple:** No complex compilation or heavy dependencies. Just plug and play.
*   **Extensible by Design:** A clean, object-oriented structure (`Layer`, `Optimiser`) makes it easy to create your own custom layers and optimizers.
*   **Modern Architecture:** Includes common layers like `Dense`, `Conv2D`, `MaxPooling2D`, `LSTM`, and `Embedding`.
*   **Powerful Optimizers:** Comes with robust implementations of `SGD` (with momentum), `RMSProp` and`Adam`.
*   **Full Documentation:** Comprehensive documentation available [here](https://helixnet.readthedocs.io/en/latest/).

## Installation

Currently, `HelixNet` can be installed directly from the source repository. Ensure you have `setuptools` and `wheel` installed.

```bash
# For the latest stable version
pip install .

# For development (editable) mode
pip install -e .
```

## Quickstart: Training a Model

Here's a quick example of how to build and train a model on the classic "spiral" dataset.

```python
import numpy as np
import mygrad as mg
import helixnet.layers as layers
import helixnet.optimisers as optimisers
import helixnet.activations as activations
import helixnet.models as models
from nnfs.datasets import spiral_data

# Create dataset
X, y = spiral_data(samples=100, classes=3)
X = mg.tensor(X)
y = mg.tensor(y, dtype=int)

# Build model
model = models.Sequental([
    layers.Dense(2, 64, activation=activations.ReLU),
    layers.Dense(64, 3, activation=(lambda x: x)) # Logits
])

optim = optimisers.Adam(learning_rate=0.05, decay=5e-7)

# Train the model
for epoch in range(10001):
    logits = model.forward(X)
    loss = mg.nnet.losses.softmax_crossentropy(logits, y)

    loss.backward()
    optim.optimise(model)
    model.null_grads()

    if epoch % 100 == 0:
        accuracy = np.mean(np.argmax(logits.data, axis=1) == y.data)
        print(f'Epoch: {epoch}, Loss: {loss.data:.4f}, Accuracy: {accuracy:.4f}')

```

## Documentation

For a full guide to all layers, optimizers, and functionalities, please see the **[Full HelixNet Documentation](https://helixnet.readthedocs.io/en/latest/)**.

## Contributing

Contributions are welcome! If you find a bug or have a feature request, please open an issue. If you'd like to contribute code, please see the notes on our own contributions to `mygrad` ([#445](https://github.com/rsokl/MyGrad/pull/445)) for an example of our development philosophy.
