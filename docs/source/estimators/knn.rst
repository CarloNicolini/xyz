kNN / KSG estimators
====================

This family implements nearest-neighbor estimators for continuous variables.
The default metric is Chebyshev (``metric="chebyshev"``), consistent with
common KSG practice and ITS/TSTOOL workflows.

Implemented classes
-------------------

- ``xyz._continuos.KSGMutualInformation``
- ``xyz._continuos.KSGEntropy``
- ``xyz._continuos.MVKSGCondEntropy``
- ``xyz._continuos.MVKSGCondMutualInformation``
- ``xyz._continuos.MVKSGTransferEntropy``
- ``xyz._continuos.KSGTransferEntropy``
- ``xyz._continuos.KSGPartialTransferEntropy``
- ``xyz._continuos.KSGSelfEntropy``
- ``xyz._continuos.MVKSGPartialInformationDecomposition``

Key idea
--------

For each sample, compute the distance to the ``k``-th neighbor in a higher
dimensional space, then project/count points in lower-dimensional spaces.
This reduces bias in differences of entropy terms.

Example: bivariate TE
---------------------

.. code-block:: python

   import numpy as np
   from xyz._continuos import KSGTransferEntropy

   data = np.random.randn(1500, 3)
   est = KSGTransferEntropy(
       driver_indices=[0],
       target_indices=[1],
       lags=1,
       k=3,
       metric="chebyshev",
   )
   est.fit(data)
   print("TE:", est.transfer_entropy_)
   print("H(Y|Y-,X-):", est.conditional_entropy_)

Example: partial TE
-------------------

.. code-block:: python

   from xyz._continuos import KSGPartialTransferEntropy

   pte = KSGPartialTransferEntropy(
       driver_indices=[0],
       target_indices=[1],
       conditioning_indices=[2],
       lags=1,
       k=3,
   )
   pte.fit(data)
   print("PTE:", pte.transfer_entropy_)

Parameter guidance
------------------

- ``k``: common values are 3-10. Smaller ``k`` lowers bias but increases variance.
- ``metric``: keep Chebyshev for ITS-style comparability.
- ``lags``: increases dimensionality quickly; prefer compact embeddings.

Numerical parity note
---------------------

For ITS-aligned TE/PTE/SE estimates, self-neighbors are excluded in projected
range counts, mirroring ``range_search(..., past=0)`` behavior.
