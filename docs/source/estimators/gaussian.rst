Gaussian and linear estimators
==============================

The Gaussian family is the right starting point when you believe the dependence
structure is mostly linear and the innovation process is reasonably close to
Gaussian.

Implemented classes
-------------------

- ``xyz.MVNEntropy``
- ``xyz.MVLNEntropy``
- ``xyz.MVParetoEntropy`` *(placeholder, not implemented)*
- ``xyz.MVExponentialEntropy`` *(placeholder, not implemented)*
- ``xyz.MVCondEntropy``
- ``xyz.MVNMutualInformation``
- ``xyz.GaussianTransferEntropy``
- ``xyz.GaussianPartialTransferEntropy``
- ``xyz.GaussianSelfEntropy``

What these estimators compute
-----------------------------

Multivariate Gaussian entropy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For :math:`Y \sim \mathcal{N}(\mu, \Sigma)` with dimension :math:`d`,

.. math::

   H(Y) = \frac{1}{2}\log\!\left((2\pi e)^d \det(\Sigma)\right).

This is estimated directly from the sample covariance matrix in
``MVNEntropy``.

Conditional entropy from linear regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If :math:`Y` is linearly regressed on :math:`X`,

.. math::

   Y = XA + \varepsilon,

then the Gaussian conditional entropy is determined by the residual covariance:

.. math::

   H(Y \mid X)
   = \frac{1}{2}\log\!\left((2\pi e)^d \det(\Sigma_{\varepsilon})\right).

This is the quantity returned by ``MVCondEntropy``.

Gaussian transfer entropy
^^^^^^^^^^^^^^^^^^^^^^^^^

For a target :math:`Y_t`, driver :math:`X_t`, and target past :math:`Y_t^-`,
Gaussian transfer entropy is estimated as

.. math::

   TE_{X \to Y}
   = H(Y_t \mid Y_t^-)
   - H(Y_t \mid Y_t^-, X_t^-).

Operationally, this is the difference between the entropy of:

- residuals from the restricted regression
  :math:`Y_t \sim Y_t^-`, and
- residuals from the unrestricted regression
  :math:`Y_t \sim Y_t^- + X_t^-`.

Gaussian partial transfer entropy adds controls :math:`Z_t^-`:

.. math::

   PTE_{X \to Y \mid Z}
   = H(Y_t \mid Y_t^-, Z_t^-)
   - H(Y_t \mid Y_t^-, X_t^-, Z_t^-).

Gaussian self-entropy / information storage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Information storage is

.. math::

   SE_Y = H(Y_t) - H(Y_t \mid Y_t^-).

It quantifies how predictable the present is from the process's own past.

Why use the Gaussian family
---------------------------

- It is usually the fastest estimator family in the library.
- It has a clear regression interpretation that is familiar to users of VAR and
  Granger-causality style models.
- It works well as a baseline even when you later move to KSG or kernel-based
  estimators.
- It provides a natural analytical significance test through the F-test on
  restricted vs unrestricted regressions.

When to prefer it
-----------------

Use Gaussian estimators when:

- your sample size is modest,
- you need fast lag or delay scans,
- interpretability matters more than flexibility,
- or you want a first-pass diagnostic before running more expensive
  nonparametric estimators.

Typical use cases
-----------------

- Finance: factor-to-asset lead-lag analysis, market microstructure influence,
  baseline directional dependence screening.
- Neuroscience: approximate linear interactions, especially in preprocessing or
  quick exploratory pipelines.
- Physiology: autoregulatory dynamics where storage and directed influence are
  both of interest.

How to use them
---------------

.. code-block:: python

   import numpy as np
   from xyz import (
       GaussianPartialTransferEntropy,
       GaussianSelfEntropy,
       GaussianTransferEntropy,
   )

   data = np.random.randn(2000, 3)

   te = GaussianTransferEntropy(
       driver_indices=[0],
       target_indices=[1],
       lags=2,
       tau=1,
       delay=1,
   ).fit(data)

   pte = GaussianPartialTransferEntropy(
       driver_indices=[0],
       target_indices=[1],
       conditioning_indices=[2],
       lags=2,
   ).fit(data)

   se = GaussianSelfEntropy(target_indices=[1], lags=2).fit(data)

   print(te.transfer_entropy_, te.p_value_)
   print(pte.transfer_entropy_, pte.p_value_)
   print(se.self_entropy_, se.p_value_)

Parameter guidance
------------------

- ``lags`` controls how much of the past is included.
- ``tau`` controls the spacing between lagged coordinates.
- ``delay`` controls the source-to-target interaction lag when you want a
  TRENTOOL-like delay scan.
- ``conditioning_indices`` should include plausible confounders when using
  partial TE.

Practical advice
----------------

- Centering and scaling are sensible defaults for most applications.
- If TE changes dramatically with small lag changes, the data may need a more
  explicit embedding search rather than a hand-picked lag.
- Use the Gaussian family as the computational backbone for model-selection
  sweeps, then confirm important findings with KSG where nonlinear structure is
  plausible.

Interactive example
-------------------

The figure below shows Gaussian TE as a function of the assumed interaction
delay in a synthetic linear system. The peak should occur near the true delay.

.. plotly-exec::

   import numpy as np
   import plotly.graph_objects as go
   from xyz import GaussianTransferEntropy

   rng = np.random.default_rng(10)
   n = 900
   true_delay = 2
   driver = rng.normal(size=n)
   target = np.zeros(n)
   for t in range(true_delay, n):
       target[t] = 0.55 * target[t - 1] + 0.45 * driver[t - true_delay] + 0.10 * rng.normal()

   data = np.column_stack([target, driver])
   delays = [1, 2, 3, 4, 5, 6]
   te_vals = []
   p_vals = []
   for delay in delays:
       est = GaussianTransferEntropy(
           driver_indices=[1],
           target_indices=[0],
           lags=1,
           delay=delay,
       ).fit(data)
       te_vals.append(est.transfer_entropy_)
       p_vals.append(est.p_value_)

   fig = go.Figure()
   fig.add_trace(
       go.Scatter(
           x=delays,
           y=te_vals,
           mode="lines+markers",
           name="Gaussian TE",
       )
   )
   fig.add_trace(
       go.Bar(
           x=delays,
           y=[-np.log10(max(p, 1e-12)) for p in p_vals],
           name="-log10(p-value)",
           opacity=0.35,
           yaxis="y2",
       )
   )
   fig.add_vline(x=true_delay, line_dash="dash", annotation_text="True delay")
   fig.update_layout(
       title="Gaussian transfer entropy across candidate delays",
       xaxis_title="Assumed interaction delay",
       yaxis_title="Transfer entropy (nats)",
       yaxis2=dict(
           title="-log10(p-value)",
           overlaying="y",
           side="right",
           showgrid=False,
       ),
       template="plotly_white",
       legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
       height=430,
       margin=dict(l=40, r=40, t=60, b=40),
   )
