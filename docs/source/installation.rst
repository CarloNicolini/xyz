Installation
============

Requirements
------------

- Python 3.12+
- ``numpy``
- ``scipy``
- ``scikit-learn``

Install package
---------------

From the repository root:

.. code-block:: bash

   pip install -e .

Or with ``uv``:

.. code-block:: bash

   uv pip install -e .

Install docs dependencies
-------------------------

If you use ``uv`` dependency groups:

.. code-block:: bash

   uv sync --group docs

Or with ``pip``:

.. code-block:: bash

   pip install sphinx pydata-sphinx-theme sphinx-autobuild sphinx-copybutton
