Discrete (binning) estimators
=============================

The discrete family is intended for symbolic processes or for continuous data
that you deliberately quantize into a small number of states.

Implemented classes
-------------------

- ``xyz.DiscreteTransferEntropy``
- ``xyz.DiscretePartialTransferEntropy``
- ``xyz.DiscreteSelfEntropy``

Mathematics
-----------

These estimators compute empirical probabilities from counts of repeated states
in embedded observation matrices.

If :math:`Y_t` is the current target state and :math:`Y_t^-` is its embedded
past, then:

.. math::

   TE_{X \to Y}
   = H(Y_t \mid Y_t^-)
   - H(Y_t \mid Y_t^-, X_t^-),

.. math::

   PTE_{X \to Y \mid Z}
   = H(Y_t \mid Y_t^-, Z_t^-)
   - H(Y_t \mid Y_t^-, X_t^-, Z_t^-),

.. math::

   SE_Y = H(Y_t) - H(Y_t \mid Y_t^-).

Each entropy term is evaluated from empirical frequencies:

.. math::

   \hat{H}(Y) = -\sum_y \hat{p}(y)\log \hat{p}(y).

Quantization
------------

When ``quantize=True``, ``xyz`` applies MATLAB-compatible uniform quantization
with ``c`` bins before counting discrete states. This is useful for ITS-style
parity and for exploratory symbolic analysis, but it also introduces a modeling
choice: the result now depends on the quantization scheme.

Why use discrete estimators
---------------------------

- They are natural for genuinely discrete state spaces.
- They are easy to interpret because they reduce everything to frequency tables.
- They are often a useful pedagogical baseline for understanding TE and PTE.

When to use them
----------------

Use the discrete family when:

- your data are already categorical or symbolic,
- you want to compare multiple coarse quantizations of a continuous process,
- or you want a transparent state-counting baseline before moving to KSG or
  Gaussian estimators.

Typical use cases
-----------------

- Symbolic dynamics and regime switching.
- Discretized market states, such as up/flat/down returns.
- Binned neural or physiological activity states.

How to use them
---------------

.. code-block:: python

   import numpy as np
   from xyz import (
       DiscretePartialTransferEntropy,
       DiscreteSelfEntropy,
       DiscreteTransferEntropy,
   )

   data = np.random.randn(2000, 3)

   te = DiscreteTransferEntropy(
       driver_indices=[0],
       target_indices=[1],
       lags=1,
       c=8,
       quantize=True,
   ).fit(data)

   pte = DiscretePartialTransferEntropy(
       driver_indices=[0],
       target_indices=[1],
       conditioning_indices=[2],
       lags=1,
       c=8,
   ).fit(data)

   se = DiscreteSelfEntropy(target_indices=[1], lags=2, c=8).fit(data)

   print(te.transfer_entropy_)
   print(pte.transfer_entropy_)
   print(se.self_entropy_)

Practical advice
----------------

- ``c`` too small merges distinct states and may underfit.
- ``c`` too large creates sparse tables and unstable estimates.
- If estimates change dramatically with the number of bins, report that
  sensitivity rather than hiding it.
- In higher-dimensional embeddings, the discrete state space grows quickly, so
  KSG or Gaussian estimators may become more reliable.

Interactive example
-------------------

The plot below shows discrete TE as a function of the number of quantization
bins in a synthetic lagged system. This is a useful diagnostic because an
estimate that is only present for one narrow bin count is often not robust.

.. plotly-exec::

   import numpy as np
   import plotly.graph_objects as go
   from xyz import DiscreteTransferEntropy

   rng = np.random.default_rng(16)
   n = 900
   driver = rng.normal(size=n)
   target = np.zeros(n)
   for t in range(1, n):
       target[t] = 0.55 * target[t - 1] + 0.40 * driver[t - 1] + 0.10 * rng.normal()

   data = np.column_stack([target, driver])
   bins = [3, 4, 5, 6, 8, 10, 12]
   te_vals = []
   for c in bins:
       est = DiscreteTransferEntropy(
           driver_indices=[1],
           target_indices=[0],
           lags=1,
           c=c,
           quantize=True,
       ).fit(data)
       te_vals.append(est.transfer_entropy_)

   fig = go.Figure()
   fig.add_trace(
       go.Bar(
           x=bins,
           y=te_vals,
           name="Discrete TE",
       )
   )
   fig.update_layout(
       title="Discrete transfer entropy across quantization granularities",
       xaxis_title="Number of bins c",
       yaxis_title="Transfer entropy (nats)",
       template="plotly_white",
       height=420,
       margin=dict(l=40, r=20, t=60, b=40),
   )
