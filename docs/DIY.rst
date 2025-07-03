DIY: Extending HelixNet
=======================

HelixNet is designed for easy extensibility. This guide will show you how 
to create your own optimizers, layers, and activation functions.

.. warning:: API is under heavy development.
    HelixNet is in the very early stages of development and the API may change
    at any moment. If you have built custom components, always check the latest
    documentation and changelogs for breaking changes.

Optimizers
----------
Creating a custom optimizer is a very easy and straightforward process.
All you need is just inheritance of :class:`helixnet.optimizers.Optimizer` 
and, most importantly,
overload the :class:`helixnet.optimizers.Optimizer.optimize_param` method.


The base :class:`helixnet.optimizers.Optimizer` class provides the main training loop, which iterates
through all the trainable parameters in a model and passes them to your
:class:`helixnet.optimizers.Optimizer.optimize_param` method one by one.

Let's create an ``RMSProp`` optimizer as an example:

.. code-block:: python

    from helixnet.optimisers import Optimiser
    import numpy as np
    import mygrad as mg

    class RMSProp(Optimizer):
    """Root Mean Square Propagation optimiser or for short named RMSProp"""

        def __init__(self, lr=0.001, decay=None, epsilon=1e-7, rho=0.9,
                    regularizers: List[Regularizer] = None):
            # Here we will pass the learn rate to parent optimizer class
            # And the last of regularizers that will applied on the parameters
            super().__init__(lr, regularizers)
            # The learn rate will be stored in self.lr
            self.decay = decay
            self.epsilon = epsilon
            self.rho = rho
            self.cache = {}

        def get_current_lr(self):
            # Update the learning rate based on the current step/iteration
            if self.decay:
                return self.lr * (1.0 / (1.0 + self.decay * self.step))
            return self.lr

        def optimize_param(self, parameter: mg.Tensor) -> None:
            """This method contains the update logic for a single parameter."""
            # Initialize the cache for this parameter if it's the first time we've seen it.
            # Using id(parameter) is a robust way to get a unique key.
            if id(parameter) not in self.cache:
                self.cache[id(parameter)] = np.zeros_like(parameter.data)

            self.cache[id(parameter)] = (self.rho * self.cache[id(parameter)] +
                                        (1 - self.rho) * parameter.grad**2)

            # 2. Update the parameter's data using the cache
            parameter.data += -self.lr * \
                parameter.grad / (np.sqrt(self.cache[id(parameter)]) + self.epsilon)

