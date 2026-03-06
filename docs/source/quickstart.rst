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

Gaussian-copula MI (robust to nonlinear marginals)
--------------------------------------------------

.. code-block:: python

   from xyz import GaussianCopulaMutualInformation

   mi_copula = GaussianCopulaMutualInformation().fit(X, Y).score()
   # Matches Gaussian MI on Gaussian data; stable under monotone transforms

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

Gaussian transfer entropy (with multivariate drivers)
-----------------------------------------------------

.. code-block:: python

   from xyz import GaussianTransferEntropy

   model = GaussianTransferEntropy(driver_indices=[0], target_indices=[1], lags=1)
   model.fit(data)
   print(model.transfer_entropy_, model.p_value_)

   # Multiple drivers: driver_indices=[1, 2] uses both columns 1 and 2
   te_multi = GaussianTransferEntropy(driver_indices=[1, 2], target_indices=[0], lags=1).fit(data)

Delay search (with parallelization)
------------------------------------

.. code-block:: python

   from xyz import GaussianTransferEntropy, InteractionDelaySearchCV

   search = InteractionDelaySearchCV(
       GaussianTransferEntropy(driver_indices=[0], target_indices=[1], lags=1),
       delays=[1, 2, 3, 4],
       n_jobs=2,
   )
   search.fit(data)
   print(search.best_delay_, search.best_score_)

Bootstrap confidence intervals
-------------------------------

.. code-block:: python

   from xyz import BootstrapEstimate, GaussianTransferEntropy

   bootstrap = BootstrapEstimate(
       GaussianTransferEntropy(driver_indices=[1], target_indices=[0], lags=1),
       n_bootstrap=100,
       method="trial",
       ci=0.95,
       n_jobs=2,
       random_state=0,
   ).fit(data)
   print(bootstrap.estimate_, bootstrap.ci_low_, bootstrap.ci_high_)

Greedy source selection
------------------------

.. code-block:: python

   from xyz import GaussianPartialTransferEntropy, GreedySourceSelectionTransferEntropy

   selector = GreedySourceSelectionTransferEntropy(
       GaussianPartialTransferEntropy(
           driver_indices=[1],
           target_indices=[0],
           conditioning_indices=[],
           lags=1,
       ),
       candidate_sources=[1, 2, 3],
       max_sources=3,
       min_improvement=0.01,
   ).fit(data)
   print(selector.selected_sources_, selector.best_score_)
