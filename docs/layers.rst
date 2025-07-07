Layers
======

Here we have our building blocks **The Layers**

.. autoclass:: helixnet.layers.Layer
    :members:

.. warning::
    Don't attempt to modify :class:`helixnet.layers.Layer.trainable_params`
    because this might cause the optimizer to lose its state data (e.g., momentum)

    Also it's not recommended to overload :class:`helixnet.layers.Layer.predict`
    ,:class:`helixnet.layers.Layer.null_grad` and
    :class:`helixnet.layers.Layer.total_params`
    to ensure consistent behaviour across the framework

.. note::
    If you can calculate the output of a custom created layer more
    efficiently you should overload :class:`helixnet.layers.Layer.output_shape`

.. autoclass:: helixnet.layers.Dense
    :members:

.. autoclass:: helixnet.layers.Conv2D
    :members:

.. autoclass:: helixnet.layers.MaxPooling2D
    :members:

.. autoclass:: helixnet.layers.Flatten
    :members:

.. autoclass:: helixnet.layers.Embedding
    :members:

.. autoclass:: helixnet.layers.InputShape
    :members:

.. autoclass:: helixnet.layers.LSTMLayer
    :members:

.. autoclass:: helixnet.layers.LSTMCell
    :members:

.. autoclass:: helixnet.layers.BatchNorm
    :members:

.. autoclass:: helixnet.layers.Dropout
    :members: