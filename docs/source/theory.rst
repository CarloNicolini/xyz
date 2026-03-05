Theory and notation
===================

This page defines the notation used by ``xyz`` and summarizes the quantities
computed by the estimators.

Time-series notation
--------------------

Let ``Y_n`` be the current sample of a target process, and let ``Y_n^-`` denote
its embedded past (lagged vector). Likewise, ``X_n^-`` and ``Z_n^-`` denote
embedded past vectors for driver and conditioning processes.

``xyz`` uses delay-embedding matrices built by ``xyz.utils.buildvectors``.

Core quantities
---------------

Entropy
^^^^^^^

For a continuous random vector ``Y``:

.. math::

   H(Y) = -\int p(y)\log p(y)\,dy

For Gaussian ``Y \sim \mathcal{N}(0,\Sigma)``:

.. math::

   H(Y) = \tfrac{1}{2}\log\!\left((2\pi e)^d\det\Sigma\right)

Conditional entropy
^^^^^^^^^^^^^^^^^^^

.. math::

   H(Y|X) = H(X,Y) - H(X)

Mutual information
^^^^^^^^^^^^^^^^^^

.. math::

   I(X;Y) = H(X) + H(Y) - H(X,Y)

Transfer entropy
^^^^^^^^^^^^^^^^

Bivariate transfer entropy from ``X`` to ``Y``:

.. math::

   TE_{X\to Y} = I(X_n^-;Y_n \mid Y_n^-)
               = H(Y_n \mid Y_n^-) - H(Y_n \mid Y_n^-,X_n^-)

Partial transfer entropy
^^^^^^^^^^^^^^^^^^^^^^^^

Conditioning on additional processes ``Z``:

.. math::

   PTE_{X\to Y\mid Z}
   = I(X_n^-;Y_n \mid Y_n^-,Z_n^-)
   = H(Y_n \mid Y_n^-,Z_n^-)-H(Y_n \mid Y_n^-,X_n^-,Z_n^-)

Self-entropy (information storage)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   SE_Y = I(Y_n;Y_n^-)=H(Y_n)-H(Y_n\mid Y_n^-)

Estimator families in ``xyz``
-----------------------------

- **Gaussian (linear):** covariance/residual based, fast, interpretable, F-test
  significance available for TE/PTE/SE.
- **KSG / kNN:** nonparametric nearest-neighbor estimators for continuous data.
- **Kernel (fixed radius):** count points in neighborhoods of radius ``r``.
- **Discrete binning:** quantize values and estimate entropies from frequencies.

ITS/TSTOOL alignment
--------------------

The nearest-neighbor estimators are aligned to ITS toolbox conventions, where
neighbor search is done in high-dimensional spaces and projected counts are
performed in lower-dimensional spaces. In particular, self-matches are excluded
for range counting in KSG TE/PTE/SE parity mode.
