DIY: Extending HelixNet
=======================

HelixNet is designed for easy extensibility. This guide will show you how 
to create your own optimizers, layers, and activation functions.

.. warning:: API Under Development
    HelixNet is in the very early stages of development, and the API may change.
    If you build custom components, always check the latest documentation 
    and changelogs for breaking changes.

Optimizers
----------
Creating a custom optimizer is straightforward. You need to create a class that
inherits from ``helixnet.optimisers.Optimiser`` and, most importantly,
implement the ``optimise_param`` method.

The base ``Optimiser`` class provides the main training loop, which iterates
through all the trainable parameters in a model and passes them to your
``optimise_param`` method one by one.

Let's create an ``RMSProp`` optimizer as an example:

.. code-block:: python

    from helixnet.optimisers import Optimiser
    import numpy as np
    import mygrad as mg

    class RMSProp(Optimiser):
        def __init__(self, lr=0.001, decay=0.0, epsilon=1e-7,
                     rho=0.9):
            # It's good practice to call the parent's init
            super().__init__()

            self.learning_rate = lr
            self.current_learning_rate = lr
            self.decay = decay
            self.epsilon = epsilon
            self.rho = rho
            # This dictionary will store the cache for each parameter
            self.cache = {}

        def optimise_param(self, parameter: mg.Tensor, layer: layers.Layer) -> None:
            """This method contains the update logic for a single parameter."""

            # First, check if the parameter has a gradient. If not, skip it.
            if parameter.grad is None:
                return

            # Initialize the cache for this parameter if it's the first time we've seen it.
            # Using id(parameter) is a robust way to get a unique key.
            if id(parameter) not in self.cache:
                self.cache[id(parameter)] = np.zeros_like(parameter.data)

            # --- The Core RMSProp Logic ---
            # 1. Update the cache with the exponentially weighted average of squared gradients
            self.cache[id(parameter)] = (self.rho * self.cache[id(parameter)] +
                                        (1 - self.rho) * parameter.grad**2)

            # 2. Update the parameter's data using the cache
            parameter.data += -self.current_learning_rate * \
                              parameter.grad / (np.sqrt(self.cache[id(parameter)]) + self.epsilon)

Now, how do we handle learning rate decay? The base ``Optimiser`` class has an ``optimise`` method that you can override to control what happens at each training step (i.e., for each batch).

Let's modify our ``RMSProp`` optimizer to handle decay correctly. The step counter should be incremented after each batch, not each epoch.

.. code-block:: python

    class RMSProp(Optimiser):
        def __init__(self, lr=0.001, decay=0.0, epsilon=1e-7, rho=0.9):
            super().__init__()
            self.learning_rate = lr
            self.decay = decay
            # ... (rest of the init is the same)
            self.cache = {}

        def optimise(self, model: models.Sequental) -> None:
            """
            This method is called once per batch. We override it to update
            the learning rate before calling the parent's optimisation loop.
            """
            # Update the learning rate based on the current step/iteration
            if self.decay:
                self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.step))

            # Call the parent's optimise method, which runs the loop over optimise_param
            super().optimise(model)

            # Increment the step counter after the batch is processed
            self.step += 1

        def optimise_param(self, parameter: mg.Tensor, layer: layers.Layer) -> None:
            # ... (the implementation from above is perfect and needs no changes)

By overriding the main ``optimise`` method, you gain full control over the training step while still leveraging the useful ``optimise_param`` loop from the base class.