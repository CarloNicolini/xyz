Development
===========

Build docs (HTML)
-----------------

From the repository root:

.. code-block:: bash

   sphinx-build -b html docs/source docs/build/html

The build executes embedded Plotly code blocks (``.. plotly-exec::``), so
interactive charts are regenerated at each build.

Serve docs with live reload
---------------------------

.. code-block:: bash

   sphinx-autobuild docs/source docs/build/html

Then open ``http://127.0.0.1:8000``.

Makefile shortcuts
------------------

If using GNU Make:

.. code-block:: bash

   make -C docs html
   make -C docs clean
