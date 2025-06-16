Layers
======

Here we have our building blocks **The Layers**

.. py:class:: Layer(type_: str, trainable_params: List[mg.Tensor])

    :param str type_: The type of layer (e.g. Dense, Convolution). This the type will be the name of layer (e.g. Dense 1, Dense2)
    :param List[mg.Tensor] trainable_params: A list that consists of the parameters that will be trained by the optimiser. Pass an empty list to indicate that it doesn't have any trainable parameters

    This the base class of all layers that gives them a unique name if the have a trainable parameters
    and if they don't they will just use their type name without counting.

    .. py:method:: forward(X: mg.Tensor)

        :param mg.Tensor X: The data should be forward propagated 
        :return: The prediction of the layer
        :rtype: mg.Tensor

        This the function that should overloaded by the inherited layers
        And perform the forward propagation logic
    
    .. py:method:: predict(X: mg.Tensor)

        :param mg.Tensor X: The data of forward pass
        :return: The prediction of the layer
        :rtype: mg.Tensor

        This function should in inference not training because
        it doesn't track the gradients

    .. py:method:: null_grad()

        :return: This method doesn't return anything

        This function should be used we want to be sure the gradients don't stack up

.. warning::
    Don't overload these methods :class:`Layer.null_grad` and :class:`Layer.predict`
    
    And don't modify ``trainable_params`` because the optimisers might 
    lose the track of temporary gradients
    (e.g. **momentum** in :class:`SGD` or **cache** :class:`Adam`)

.. py:class:: Dense(inputs: int, params: int, activation, use_bias: bool = True, dtype = mg.float32)

    :param int inputs: The size of inputs.
    :param int params: The size of parameters. It is the size of the output.
    :param activation: The activation function that will be used with the layer. Any function or object with ``__open()__`` method.
    :param bool use_bias: Whether to have a bias or not.
    :param dtype: The data type of the parameters of the layers also using data types from **MyGrad** is preferred over NumPy

    A simple dense layer that can be used.
    Also All inherited methods are the same

    .. py:method:: forward(X: mg.Tensor)

        :param mg.Tensor X: the inputs of the layer
        :return mg.Tensor: The layer predictions

.. py:class:: Conv2D(input_channels: int, output_channels: int, kernel_size, stride=1, padding=0, activation=None, use_bias: bool = True):

    :param int input_channels: the input channels of the images e.g. 1 for grayscale pictures 3 for coloured pictures

    :param int output_channels: the number of filters in the layer
    :param int kernel_size: the size of the kernel
    :param activation: The activation function that will be used with the layer. Any function or object with ``__open()__`` method.
    :param bool use_bias: Whether to have a bias or not.

    A Simple 2D convolution layer

.. py:class:: MaxPooling2D(pool_size, stride=None)

    :param pool_size: The size of pooling can be an `int` or it can be a tuple of the size
    :param stride: The stride of tha layer

    A pooling layer that takes the maximum value

.. py:class:: Flatten

    A simple flatten layer that turns it's inputs into a flat layer
