Theory and notation
===================

This page defines the notation used throughout ``xyz`` and summarizes the
mathematical quantities estimated by the library.

Why this matters
----------------

The same high-level quantity, such as transfer entropy, can be estimated under
very different assumptions:

- a linear-Gaussian model,
- a nonparametric nearest-neighbor model,
- a fixed-radius kernel approximation,
- or a discrete/binned empirical distribution.

The estimators in ``xyz`` are therefore best understood as different numerical
approximations to the same information-theoretic functionals.

Notation
--------

Let :math:`Y_t` be the target process at time :math:`t`. Its embedded past is
written as

.. math::

   Y_t^- = \left[Y_{t-\tau}, Y_{t-2\tau}, \ldots, Y_{t-d\tau}\right],

where :math:`d` is the embedding dimension and :math:`\tau` is the embedding
spacing. Likewise:

- :math:`X_t^-` denotes the embedded past of a driver process,
- :math:`Z_t^-` denotes the embedded past of one or more conditioning processes,
- :math:`u` denotes an interaction delay between source and target when a
  delay-specific TE estimator is used.

In ``xyz``, these state vectors are assembled by the delay-embedding helpers in
``xyz.preprocessing``.

Units
-----

Unless otherwise stated, values are reported in **nats**:

.. math::

   1\ \text{nat} = \log_2(e)\ \text{bits} \approx 1.4427\ \text{bits}.

Core information quantities
---------------------------

Entropy
^^^^^^^

For a continuous random vector :math:`Y \in \mathbb{R}^d`,

.. math::

   H(Y) = - \int p(y)\log p(y)\,dy.

If :math:`Y` is Gaussian with covariance matrix :math:`\Sigma`,

.. math::

   H(Y)
   = \frac{1}{2}\log\!\left((2\pi e)^d \det(\Sigma)\right).

This is the quantity estimated by :class:`xyz.MVNEntropy`.

Conditional entropy
^^^^^^^^^^^^^^^^^^^

For two random variables :math:`X` and :math:`Y`,

.. math::

   H(Y \mid X) = H(X, Y) - H(X).

In a regression-based Gaussian setting, this can be expressed via the
covariance of the residual process:

.. math::

   H(Y \mid X)
   = \frac{1}{2}\log\!\left((2\pi e)^d \det(\Sigma_{\varepsilon})\right),

where :math:`\varepsilon` are the residuals from regressing :math:`Y` on
:math:`X`.

Mutual information
^^^^^^^^^^^^^^^^^^

Mutual information measures statistical dependence:

.. math::

   I(X;Y) = H(X) + H(Y) - H(X,Y)
          = H(Y) - H(Y \mid X).

It is symmetric in :math:`X` and :math:`Y` and nonnegative in the population.
In finite samples, nonparametric estimators can produce small negative values
because of estimation variance.

Conditional mutual information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conditional mutual information measures dependence that remains after adjusting
for a third variable:

.. math::

   I(X;Y \mid Z)
   = H(X \mid Z) + H(Y \mid Z) - H(X,Y \mid Z).

This is the core building block of transfer entropy.

Transfer entropy
^^^^^^^^^^^^^^^^

Bivariate transfer entropy from :math:`X` to :math:`Y` quantifies predictive
information flow from the past of :math:`X` to the present of :math:`Y` beyond
the information already contained in the past of :math:`Y`:

.. math::

   TE_{X \to Y}
   = I(X_t^-; Y_t \mid Y_t^-)
   = H(Y_t \mid Y_t^-) - H(Y_t \mid Y_t^-, X_t^-).

If a separate interaction delay :math:`u` is used, the source state can be
written more explicitly as

.. math::

   X_{t,u}^- = \left[X_{t-u}, X_{t-u-\tau}, \ldots, X_{t-u-(d-1)\tau}\right].

This distinction is important in TRENTOOL-style delay reconstruction.

Partial transfer entropy
^^^^^^^^^^^^^^^^^^^^^^^^

Partial transfer entropy adjusts for additional confounding processes
:math:`Z_t^-`:

.. math::

   PTE_{X \to Y \mid Z}
   = I(X_t^-; Y_t \mid Y_t^-, Z_t^-)
   = H(Y_t \mid Y_t^-, Z_t^-)
     - H(Y_t \mid Y_t^-, X_t^-, Z_t^-).

This is the natural choice when the apparent effect of :math:`X` on :math:`Y`
could be mediated or confounded by known controls.

Self-entropy / information storage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``xyz`` uses the term *self-entropy* for information storage:

.. math::

   SE_Y = I(Y_t; Y_t^-)
        = H(Y_t) - H(Y_t \mid Y_t^-).

This quantifies how much of the present of a process is predictable from its
own past.

Estimator families in ``xyz``
-----------------------------

Gaussian / linear
^^^^^^^^^^^^^^^^^

These estimators assume the relevant distributions are well approximated by
linear regressions with Gaussian residuals. They are fast, interpretable, and
provide analytical F-tests for TE, PTE, and self-entropy.

KSG / nearest-neighbor
^^^^^^^^^^^^^^^^^^^^^^

These estimators are nonparametric and approximate entropies from
nearest-neighbor distances. They are more flexible than Gaussian estimators,
especially for nonlinear dependence, but require more data and more careful
parameter tuning.

Kernel / fixed-radius
^^^^^^^^^^^^^^^^^^^^^

These estimators replace the fixed-:math:`k` neighborhood of KSG with a fixed
radius :math:`r`. They are intuitive and useful for sensitivity analysis, but
their performance can change substantially with the chosen radius.

Discrete / binning
^^^^^^^^^^^^^^^^^^

These estimators quantize the data and estimate probabilities from empirical
frequencies. They are especially appropriate for symbolic or truly discrete
state spaces, but can become sparse in high-dimensional embeddings.

How to choose an estimator family
---------------------------------

.. list-table::
   :header-rows: 1

   * - Family
     - Best when
     - Main strengths
     - Main risks
   * - Gaussian
     - Dynamics are approximately linear and homoscedastic
     - Fast, stable, interpretable, analytical significance
     - Misses nonlinear structure
   * - KSG
     - Nonlinear dependence is plausible and sample size is adequate
     - Flexible, widely used, closest to TRENTOOL-style continuous TE
     - Higher variance, more tuning, more expensive
   * - Kernel
     - A local geometric neighborhood view is desirable
     - Simple radius interpretation, useful for robustness sweeps
     - Highly sensitive to ``r``
   * - Discrete
     - Data are symbolic, categorical, or deliberately quantized
     - Conceptually simple, easy to interpret
     - Binning bias and state-space sparsity

ITS / TSTOOL / TRENTOOL alignment
---------------------------------

The continuous nearest-neighbor estimators in ``xyz`` follow the same broad
strategy as ITS/TSTOOL/TRENTOOL:

1. find a neighborhood in the highest-dimensional joint space,
2. project that neighborhood into lower-dimensional marginal spaces,
3. use projected counts to estimate entropy differences with reduced bias.

For the TE/PTE/SE parity tests, ``xyz`` excludes self-matches in the projected
count stage, mirroring the ITS ``range_search(..., past=0)`` behavior.

The TRENTOOL workflow then layers additional methodology on top of those core
estimators: ACT-aware trial selection, Ragwitz embedding search, interaction
delay reconstruction, surrogate testing, and group-level harmonization. Those
workflow components are the bridge between low-level estimator parity and a
full causal-analysis pipeline.
