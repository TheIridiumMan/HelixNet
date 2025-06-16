Optimisers
==========
``helixnet.optimiser`` is the file that contains two optimisers which are
**SGD** and **Adam** and more coming soon

SGD
---

.. py:class:: SGD(lr: float, decay: float = None, momentum: float = None)

  :param float lr: The learn rate of the optimiser
  :param float decay: The rate of learn rate decay can be ``None`` or ``False`` in order to avoid decay
  :param float momentum: The momentum but can be ``None`` or ``False`` in order to avoid decay

  Stochastic Gradient Descend is a powerful optimiser and is more stable than Adam numerically

  .. py:method:: get_current_lr()

      :return: The learn rate with decay if existed
      :rtype: float

    This method returns the learn rate with respect to the current step

  .. py:method:: epoch_done()

    This method should be called after every epoch_done is done in order to inform the optimiser to
    update it's parameters like weight decay
  
  .. py:method:: optimise(model: helixnet.models.Sequental)

      :param models.Sequental model: The model that needs to be trained

    This method performs training sequental models


.. important::
    Don't set high parameter especially the **momentum** because it might be
    numerically unstable

Adam
----

.. py:class:: Adam(learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999)
  

  :param float lr: The learn rate of the optimiser
  :param float decay: The rate of learn rate decay can be ``None`` or ``False`` in order to avoid decay

  Adam a very good optimiser can converge quickly but less stable numerically
  .. py:method:: get_current_lr()

      :return: The learn rate with decay if existed
      :rtype: float

    This method returns the learn rate with respect to the current step

  .. py:method:: epoch_done()
    This method should be called after every epoch_done is done in order to inform the optimiser to
    update it's parameters like weight decay
  
  .. py:method:: optimise(model: helixnet.models.Sequental)

      :param models.Sequental model: The model that needs to be trained
  
      This method performs training sequental models
