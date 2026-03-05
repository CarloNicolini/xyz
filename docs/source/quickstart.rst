Quickstart
==========

KSG mutual information
----------------------

.. code-block:: python

   import numpy as np
   from xyz import KSGMutualInformation

   X = np.random.randn(500)
   Y = X + 0.3 * np.random.randn(500)

   est = KSGMutualInformation(k=3)
   est.fit(X, Y)
   mi = est.score(X, Y)
   print(mi)

KSG transfer entropy
--------------------

.. code-block:: python

   import numpy as np
   from xyz import KSGTransferEntropy

   data = np.random.randn(1000, 3)
   te = KSGTransferEntropy(
       driver_indices=[0],
       target_indices=[1],
       lags=1,
       tau=1,
       delay=1,
       k=3,
       metric="chebyshev",
   )
   te.fit(data)
   print(te.transfer_entropy_)

Gaussian transfer entropy
-------------------------

.. code-block:: python

   from xyz import GaussianTransferEntropy

   model = GaussianTransferEntropy(driver_indices=[0], target_indices=[1], lags=1)
   model.fit(data)
   print(model.transfer_entropy_, model.p_value_)

Delay search
------------

.. code-block:: python

   from xyz import GaussianTransferEntropy, InteractionDelaySearchCV

   search = InteractionDelaySearchCV(
       GaussianTransferEntropy(driver_indices=[0], target_indices=[1], lags=1),
       delays=[1, 2, 3, 4],
   )
   search.fit(data)
   print(search.best_delay_, search.best_score_)
