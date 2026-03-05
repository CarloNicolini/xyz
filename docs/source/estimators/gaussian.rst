Gaussian and linear estimators
==============================

These estimators assume linear-Gaussian structure and use covariance or linear
regression residuals.

Implemented classes
-------------------

- ``xyz._continuos.MVNEntropy``
- ``xyz._continuos.MVLNEntropy``
- ``xyz._continuos.MVParetoEntropy`` *(placeholder, not implemented)*
- ``xyz._continuos.MVExponentialEntropy`` *(placeholder, not implemented)*
- ``xyz._continuos.MVCondEntropy``
- ``xyz._continuos.MVNMutualInformation``
- ``xyz._continuos.GaussianTransferEntropy``
- ``xyz._continuos.GaussianPartialTransferEntropy``
- ``xyz._continuos.GaussianSelfEntropy``

Interpretation
--------------

- Fast and stable for approximately Gaussian returns.
- Often a strong baseline in finance when linear dynamics dominate.
- TE/PTE/SE classes provide ``p_value_`` via an F-test on restricted vs
  unrestricted regressions.

Example: Gaussian transfer entropy
----------------------------------

.. code-block:: python

   import numpy as np
   from xyz._continuos import GaussianTransferEntropy

   data = np.random.randn(2000, 3)
   est = GaussianTransferEntropy(driver_indices=[0], target_indices=[1], lags=2)
   est.fit(data)
   print("TE:", est.transfer_entropy_)
   print("p-value:", est.p_value_)

Practical guidance
------------------

- Normalize/standardize time series before fitting.
- Choose ``lags`` based on domain knowledge or IC-based model order selection.
- Use Gaussian estimates as baseline against nonparametric KSG/kernel estimates.
