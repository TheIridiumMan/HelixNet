Installation
============

Using Pip
---------
Go to the directory of sources and start your terminal there or ``cd`` to it.
Because currently we don't have `PyPI`_ package yet.
Also you should a have ``setuptools`` and ``wheel`` for building the wheel


for ``setuptools`` and ``wheel`` and if you have them already you can skip them

.. code-block:: bash

    pip install wheel setuptools

.. code-block:: bash

    pip install .

Installation for Development
----------------------------
For editable in-place installation. Which when you modify the code in sources
directory it take effect and this very helpful for development

.. code-block:: bash

    pip install -e .

.. _PyPi: https://pypi.org
