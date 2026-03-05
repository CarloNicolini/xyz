Discrete (binning) estimators
=============================

Discrete estimators quantize continuous signals (or consume already-discrete
states) and compute information quantities from empirical frequencies.

Implemented classes
-------------------

- ``xyz._discrete.DiscreteTransferEntropy``
- ``xyz._discrete.DiscretePartialTransferEntropy``
- ``xyz._discrete.DiscreteSelfEntropy``

Quantization
------------

When ``quantize=True``, ``xyz`` applies MATLAB-compatible uniform quantization
with ``c`` bins, then operates on integer state labels.

Formulas
--------

- ``TE(X->Y) = H(Y_n|Y_n^-) - H(Y_n|Y_n^-,X_n^-)``
- ``PTE(X->Y|Z) = H(Y_n|Y_n^-,Z_n^-) - H(Y_n|Y_n^-,X_n^-,Z_n^-)``
- ``SE(Y) = H(Y_n) - H(Y_n|Y_n^-)``

Example
-------

.. code-block:: python

   import numpy as np
   from xyz._discrete import DiscreteTransferEntropy

   data = np.random.randn(2000, 3)
   est = DiscreteTransferEntropy(
       driver_indices=[0],
       target_indices=[1],
       lags=1,
       c=8,
       quantize=True,
   )
   est.fit(data)
   print("TE:", est.transfer_entropy_)

Practical guidance
------------------

- ``c`` too small underfits, too large increases sparsity.
- For financial returns, test multiple ``c`` values and report robustness.
- With high-dimensional embeddings, prefer KSG or Gaussian baselines if counts
  become too sparse.
