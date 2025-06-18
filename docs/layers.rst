Layers
======

Here we have our building blocks **The Layers**

.. autoclass:: helixnet.layers.Layer
    :members:

.. warning:: 
    Don't attempt to modify :class:`helixnet.layers.Layer.trainable_params`
    because this might cause the optimizer to lose its state data (e.g., momentum)

    Also it's not recommended to overload :class:`helixnet.layers.Layer.predict`
    and :class:`helixnet.layers.Layer.null_grad` to ensure consistent behavior across the framework

.. autoclass:: helixnet.layers.Dense
    :members:

.. autoclass:: helixnet.layers.Conv2D
    :members:

.. autoclass:: helixnet.layers.MaxPooling2D

.. autoclass:: helixnet.layers.Flatten