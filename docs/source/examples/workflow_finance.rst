Finance workflow example
========================

This example illustrates a practical workflow for return-based information
dynamics.

Problem setup
-------------

- ``Y``: target return (asset or portfolio).
- ``X``: candidate driver (factor, market index, signal).
- ``Z``: controls/confounders (other factors, volatility proxies, sector index).

Suggested sequence
------------------

1. **Baseline linear inference**

   - Fit ``GaussianTransferEntropy`` and ``GaussianPartialTransferEntropy``.
   - Use ``p_value_`` for quick significance screening.

2. **Nonparametric confirmation**

   - Fit ``KSGTransferEntropy`` / ``KSGPartialTransferEntropy``.
   - Check robustness over ``k`` (for example 3, 5, 8).

3. **Storage diagnostics**

   - Fit ``KSGSelfEntropy`` (or Gaussian/Kernel variants) for persistence in ``Y``.

4. **Sensitivity analysis**

   - Vary lags/embedding size.
   - Compare metrics (Chebyshev vs Euclidean if needed).
   - Validate on rolling windows for nonstationarity.

Code sketch
-----------

.. code-block:: python

   import numpy as np
   from xyz import (
       GaussianPartialTransferEntropy,
       KSGPartialTransferEntropy,
       KSGSelfEntropy,
   )

   # data columns: [target, driver, control]
   data = np.random.randn(3000, 3)

   gpte = GaussianPartialTransferEntropy(
       driver_indices=[1],
       target_indices=[0],
       conditioning_indices=[2],
       lags=1,
   ).fit(data)

   kpte = KSGPartialTransferEntropy(
       driver_indices=[1],
       target_indices=[0],
       conditioning_indices=[2],
       lags=1,
       k=3,
       metric="chebyshev",
   ).fit(data)

   kse = KSGSelfEntropy(target_indices=[0], lags=1, k=3).fit(data[:, [0]])

   print("Gaussian PTE:", gpte.transfer_entropy_, "p=", gpte.p_value_)
   print("KSG PTE:", kpte.transfer_entropy_)
   print("KSG SE:", kse.self_entropy_)
