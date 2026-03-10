Workflows and meta-estimators
=============================

Beyond single-call estimators, ``xyz`` provides meta-estimators and workflows for
model selection, uncertainty quantification, and multivariate source selection.

Bootstrap confidence intervals
------------------------------

``BootstrapEstimate`` wraps any estimator and returns a point estimate plus a
bootstrap distribution and confidence interval.

- **method** ``"iid"``: resample rows with replacement (suitable for non–time-series
  or when trials are exchangeable).
- **method** ``"trial"``: resample whole trials with replacement (multi-trial data);
  requires at least two trials.
- **method** ``"block"``: block bootstrap along time within each trial; use
  ``block_length`` to control block size.

Fitted attributes: ``estimate_``, ``bootstrap_distribution_``, ``ci_low_``,
``ci_high_``, ``standard_error_``. Use ``n_jobs`` to parallelize bootstrap replicates.

Example:

.. code-block:: python

   from xyz import BootstrapEstimate, GaussianTransferEntropy

   bootstrap = BootstrapEstimate(
       GaussianTransferEntropy(driver_indices=[1], target_indices=[0], lags=1),
       n_bootstrap=200,
       method="trial",
       ci=0.95,
       n_jobs=2,
       random_state=0,
   ).fit(data)
   print(bootstrap.estimate_, bootstrap.ci_low_, bootstrap.ci_high_)

Greedy source selection (multivariate TE)
-----------------------------------------

``GreedySourceSelectionTransferEntropy`` performs forward selection of driver
variables using a partial transfer entropy estimator. You provide a base estimator
with ``driver_indices`` and ``conditioning_indices``, a list of ``candidate_sources``
(column indices), and optional ``max_sources`` and ``min_improvement``. The meta-estimator
repeatedly adds the source that most improves the TE score, and stops when no candidate
adds at least ``min_improvement``.

Fitted attributes: ``selected_sources_``, ``best_estimator_``, ``best_score_``,
``selection_history_``. Supports ``n_jobs`` for evaluating candidate sets in parallel.

Example:

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
   # selector.selected_sources_ = [1, 2]  # chosen drivers
   # selector.best_estimator_.driver_indices == [1, 2]

Multivariate drivers in TE estimators
--------------------------------------

All time-series TE estimators accept ``driver_indices`` as a list of column indices.
When multiple indices are given, the embedded driver past is the concatenation of
the embedded pasts of each driver variable (same ``lags``, ``tau``, ``delay`` for all).
This allows a single TE model to include several source variables without running
greedy selection.

Example:

.. code-block:: python

   from xyz import GaussianTransferEntropy

   # data columns: 0=target, 1=driver1, 2=driver2, 3=noise
   te = GaussianTransferEntropy(
       driver_indices=[1, 2],
       target_indices=[0],
       lags=1,
   ).fit(data)

Parallelization (n_jobs)
------------------------

The following support an ``n_jobs`` parameter for parallel execution (default 1;
use ``-1`` for all cores with joblib):

- ``RagwitzEmbeddingSearchCV``: parallel over (dimension, tau) candidates.
- ``InteractionDelaySearchCV``: parallel over delay candidates.
- ``SurrogatePermutationTest``: parallel over surrogate fits.
- ``BootstrapEstimate``: parallel over bootstrap replicates.
- ``GreedySourceSelectionTransferEntropy``: parallel over candidate source sets.

Results are deterministic for a fixed ``random_state`` when applicable.
