Changelog
=========
New in X.X.x
------------
#. Ability to configure any learn rate using :class:`helixnet.optimizers.LearnRate`
#. Added Exponential Learn Rate decay :class:`helixnet.optimizers.ExpDecay`
#. Added linear Learn Rate decay :class:`helixnet.optimizers.LinearDecay`
#. Optimizers now can accept a `float` as a constant learn rate or :class:`helixnet.optimizers.LearnRate`

Breaking Changes in X.X.x
~~~~~~~~~~~~~~~~~~~~~~~~~
#. Now removed ``get_current_lr`` method and ``decay`` argument in all optimizers

New in 0.6.1
------------
#. Switch to hatch as a build system
#. The ``__version__`` is in ``__init__`` and used to be shared across build system and documentation

New in 0.6.0
------------
#. Automatic training loop with live views that shows the loss
    in realtime and also supports validation metrics
#. New :mod:`helixnet.metrics`
#. New losses :func:`helixnet.metrics.BinaryCrossEntropy` 
    :func:`helixnet.metrics.CosineSimilarityLoss`
    :func:`helixnet.metrics.HingeLoss`
    :func:`helixnet.metrics.FocalLoss`
    :func:`helixnet.metrics.LogCoshLoss`
    :func:`helixnet.metrics.HuberLoss`
#. :class:`helixnet.models.Sequential.summary` uses rich tables

New in 0.5.1
------------
#. Fixed a bug in loading the models
#. Added :class:`helixnet.layer.Layer.populate_self`

New in 0.5.0
------------
#. Now :class:`helixnet.layers.Layer` can determine the name of layer automatically
#. Added the ability to save layers and get their configuration from ``__init__`` method
#. Added the ability to save & load :class:`helixnet.models.Sequential`
#. Added :class:`helixnet.layers.DenseTranspose`
#. Added :class:`helixnet.layers.ConvTranspose2D`
#. Added :class:`helixnet.layers.Reshape`
#. Migrated the tests to pytest
#. Added new module :mod:`helixnet.io` for saving and loading :class:`helixnet.models.Sequential`


Breaking Changes
~~~~~~~~~~~~~~~~
Now :class:`helixnet.layers.Layer` doesn't accept the **type** of the layer

New in 0.4.0
------------
#. Added :class:`helixnet.layers.BatchNorm` for batch normalization
#. Added :class:`helixnet.layers.Dropout` for dropping out some features
#. Added new optimizer :class:`helixnet.optimizers.NesterovSGD` for applying nesterov trick on momentum :class:`helixnet.optimizers.SGD`
#. Added gradient clipping in :class:`helixnet.optimizers.Optimizer`

New in 0.3.0
------------
#. Added support for regularizers which introduced

    #. Added new class :class:`helixnet.optimizer.Regularizer`
    #. Created :class:`helixnet.optimizer.L1` & :class:`helixnet.optimizer.L2`

#. refactored the logic of :class:`helixnet.optimizer.Optimizer`
    which itself handles :class:`helixnet.optimizer.Regularizer`

#. Now :class:`helixnet.layers.Layer.predict` works correctly

   With :class:`helixnet.layers.Layer.predict` you can use the model
   for inference with out building a computational graph

Breaking changes
~~~~~~~~~~~~~~~~
0.3.0 has introduced many breaking changes like

#. renamed the module ``optimiser`` to :mod:`helixnet.optimizer`
#. renamed the ABC class ``optimiser`` to :class:`helixnet.optimizer.Optimizer`
#. renamed ``Optimizer.optimise`` to :class:`helixnet.optimizer.Optimizer.optimize`
#. :class:`helixnet.optimizer.Optimizer.optimize` now needs the loss to be passed

    #. The loss to be passed without Performing the backpropagation on the loss a.k.a (``loss_val.backward()``)
    #. If your have written a custom optimizer with a custom training loop through \
        :class:`helixnet.optimizer.Optimizer.optimize` you'll need to write to handle the regularization.
        But if you didn't write a custom loop your optimizer will be fully compatible

the training should be as follows

.. code-block:: python

        optim = helixnet.optimizers.SGD(0.1, None, 0.9)
        # Forward pass produces logits (raw scores)
        logits = model.forward(x)

        # The loss function takes logits and integer labels
        loss_value = mg.nnet.losses.softmax_crossentropy(logits, y_true)

        # You should call `item` instead of saving the loss itself
        # Because it's value will be changed by regularizer
        loss_history.append(loss_value.item())

        optim.optimize(model, loss_value)
        # Clear grads for the next iteration

5. Inheriting :class:`helixnet.optimizers.Optimizer` now needs **learn rate** and
a list of **regularizers** to be passed.