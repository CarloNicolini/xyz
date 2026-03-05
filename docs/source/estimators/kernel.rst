Kernel estimators
=================

Kernel estimators in ``xyz`` are fixed-radius neighborhood methods.
They approximate probabilities from pair counts within a radius ``r``.

Implemented classes
-------------------

- ``xyz._continuos.KernelTransferEntropy``
- ``xyz._continuos.KernelPartialTransferEntropy``
- ``xyz._continuos.KernelSelfEntropy``

Interpretation
--------------

- ``r`` controls locality: too small yields sparse/noisy counts, too large
  oversmooths dynamics.
- Useful when you want a geometric radius interpretation instead of a fixed
  neighbor count.

Example
-------

.. code-block:: python

   import numpy as np
   from xyz._continuos import KernelTransferEntropy

   data = np.random.randn(1500, 3)
   est = KernelTransferEntropy(
       driver_indices=[0],
       target_indices=[1],
       lags=1,
       r=0.5,
       metric="chebyshev",
   )
   est.fit(data)
   print("TE:", est.transfer_entropy_)
   print("H(Y|Y-,X-):", est.conditional_entropy_)

Choosing ``r``
--------------

- Start from a fraction of marginal standard deviation (ITS-style heuristic).
- Run sensitivity checks over a grid of ``r``.
- Prefer stable inference across a range of ``r`` values over a single-point optimum.
