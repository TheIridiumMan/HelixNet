Optimizers
==========
``helixnet.optimizer`` is the module that contains the optimisers

.. autoclass:: helixnet.optimizers.Optimizer
  :members:


.. autoclass:: helixnet.optimizers.SGD
  :members:


.. important::
    Don't set high parameter values especially for the **momentum** because it might be
    numerically unstable


.. autoclass:: helixnet.optimizers.Adam
  :members:

.. autoclass:: helixnet.optimizers.RMSProp
  :members:

.. autoclass:: helixnet.optimizers.NesterovSGD
  :members:

Regularizers
============
HelixNet offers multiple regularizers which are :class:`helixnet.optimizers.L1` and
:class:`helixnet.optimizers.L2` also with a very easy way to
create others using :class:`helixnet.optimizers.Regularizer`

.. autoclass:: helixnet.optimizers.Regularizer
  :members:

.. autoclass:: helixnet.optimizers.L1
  :members:

.. autoclass:: helixnet.optimizers.L2
  :members: