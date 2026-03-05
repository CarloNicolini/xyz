Kernel estimators
=================

Kernel estimators in ``xyz`` are fixed-radius neighborhood methods. They
replace the ``k``-nearest-neighbor logic of KSG with a radius parameter
:math:`r`, then estimate probabilities from counts inside that radius.

Implemented classes
-------------------

- ``xyz.KernelTransferEntropy``
- ``xyz.KernelPartialTransferEntropy``
- ``xyz.KernelSelfEntropy``

Mathematical idea
-----------------

For a radius :math:`r`, the estimator counts how many pairs of points fall
within the metric ball

.. math::

   \|x_i - x_j\| \le r.

The probability of a neighborhood is then approximated from these counts.
Conditional entropies are estimated from log ratios of counts in:

- a full space such as :math:`(Y_t, Y_t^-, X_t^-)`,
- and a reduced conditioning space such as :math:`(Y_t^-, X_t^-)`.

In the TE setting, this yields

.. math::

   TE_{X \to Y}
   = H(Y_t \mid Y_t^-)
   - H(Y_t \mid Y_t^-, X_t^-),

with each conditional entropy approximated by fixed-radius pair counts.

Why use kernel estimators
-------------------------

- They provide a direct geometric interpretation through the radius ``r``.
- They are often useful for sensitivity studies when you want to inspect
  locality explicitly.
- They can be easier to explain to users who think in terms of neighborhoods
  rather than in terms of nearest-neighbor order statistics.

When to use them
----------------

Kernel estimators are helpful when:

- you want a local-scale interpretation of dependence,
- you plan to sweep over neighborhood scales,
- or you want a complementary nonparametric estimate to compare against KSG.

Typical use cases
-----------------

- Exploratory analysis where robustness across locality scales matters.
- Comparative studies where both fixed-``k`` and fixed-``r`` views are useful.
- Educational settings, because the geometry is intuitive.

How to use them
---------------

.. code-block:: python

   import numpy as np
   from xyz import KernelPartialTransferEntropy, KernelSelfEntropy, KernelTransferEntropy

   data = np.random.randn(1500, 3)

   te = KernelTransferEntropy(
       driver_indices=[0],
       target_indices=[1],
       lags=1,
       r=0.5,
       metric="chebyshev",
   ).fit(data)

   pte = KernelPartialTransferEntropy(
       driver_indices=[0],
       target_indices=[1],
       conditioning_indices=[2],
       lags=1,
       r=0.5,
   ).fit(data)

   se = KernelSelfEntropy(target_indices=[1], lags=2, r=0.5).fit(data)

   print(te.transfer_entropy_)
   print(pte.transfer_entropy_)
   print(se.self_entropy_)

Choosing the radius ``r``
-------------------------

- Too small:
  counts become sparse and unstable.
- Too large:
  neighborhoods blur distinct dynamical regimes and bias estimates downward.
- In practice:
  report a sensitivity range instead of trusting one hand-picked value.

Interactive example
-------------------

The plot below shows how a kernel TE estimate changes as the radius ``r`` is
varied in a synthetic system. A stable plateau is usually more convincing than
an isolated spike.

.. plotly-exec::

   import numpy as np
   import plotly.graph_objects as go
   from xyz import KernelTransferEntropy

   rng = np.random.default_rng(14)
   n = 650
   driver = rng.normal(size=n)
   target = np.zeros(n)
   for t in range(1, n):
       target[t] = 0.45 * target[t - 1] + 0.30 * driver[t - 1] + 0.10 * rng.normal()

   data = np.column_stack([target, driver])
   radii = np.linspace(0.15, 1.00, 9)
   te_vals = []
   for radius in radii:
       est = KernelTransferEntropy(
           driver_indices=[1],
           target_indices=[0],
           lags=1,
           r=float(radius),
           metric="chebyshev",
       ).fit(data)
       te_vals.append(est.transfer_entropy_)

   fig = go.Figure()
   fig.add_trace(
       go.Scatter(
           x=radii,
           y=te_vals,
           mode="lines+markers",
           name="Kernel TE",
           line=dict(width=3),
       )
   )
   fig.update_layout(
       title="Kernel transfer entropy across radius choices",
       xaxis_title="Kernel radius r",
       yaxis_title="Transfer entropy (nats)",
       template="plotly_white",
       height=420,
       margin=dict(l=40, r=20, t=60, b=40),
   )
