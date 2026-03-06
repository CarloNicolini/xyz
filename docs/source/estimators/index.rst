Estimators
==========

This section documents the estimator families available in ``xyz``.

How to navigate this section
----------------------------

- Start with :doc:`../theory` if you want the notation and the identities that
  connect entropy, mutual information, transfer entropy, and information
  storage.
- Read :doc:`gaussian` first if you want the simplest and fastest family.
- Read :doc:`knn` if you want the most flexible continuous estimators and the
  closest conceptual match to ITS/TRENTOOL.
- Read :doc:`kernel` if you want a fixed-radius view of local neighborhoods.
- Read :doc:`discrete` if your data are symbolic or deliberately quantized.
- Read :doc:`univariate` for quick helper functions and sanity checks.
- Read :doc:`workflows` for bootstrap confidence intervals, greedy source selection,
  and parallelization (``n_jobs``).

.. toctree::
   :maxdepth: 2

   gaussian
   knn
   kernel
   discrete
   univariate
   workflows
