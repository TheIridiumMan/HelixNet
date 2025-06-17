Layers
======

Here we have our building blocks **The Layers**

.. autoclass:: helixnet.layers.Layer
    :members:

.. warning:: 
    Don't attempt to modify :class:`helixnet.layers.Layer.trainable_params`
    because this might cause that the optimiser loses it's data

    Also it's not recommended to overload :class:`helixnet.layers.Layer.predict`
    and :class:`helixnet.layers.Layer.null_grad` in order to have consistent
    behavior

.. autoclass:: helixnet.layers.Dense
    :members:

.. autoclass:: helixnet.layers.Conv2D
    :members:

.. autoclass:: helixnet.layers.MaxPooling2D

.. autoclass:: helixnet.layers.Flatten