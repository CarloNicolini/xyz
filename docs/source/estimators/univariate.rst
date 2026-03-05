Univariate helper functions
===========================

The module ``xyz.univariate`` provides lightweight function-based estimators.

Available functions
-------------------

- ``entropy_linear(A)``: Gaussian entropy from covariance.
- ``entropy_kernel(Y, r, metric="chebyshev")``: fixed-radius kernel entropy.
- ``entropy_binning(Y, c, quantize, log_base="nat")``: binned entropy utility.

Example
-------

.. code-block:: python

   import numpy as np
   from xyz.univariate import entropy_linear, entropy_kernel

   y = np.random.randn(2000, 1)
   print("H linear:", entropy_linear(y))
   print("H kernel:", entropy_kernel(y, r=0.3))
