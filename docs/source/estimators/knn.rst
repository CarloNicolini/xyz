kNN / KSG estimators
====================

The KSG family provides nonparametric estimators for entropy, mutual
information, conditional mutual information, and time-series information flow.
This is the estimator family in ``xyz`` that is conceptually closest to the
continuous nearest-neighbor machinery used by ITS and TRENTOOL.

Implemented classes
-------------------

- ``xyz.KSGMutualInformation``
- ``xyz.KSGEntropy``
- ``xyz.MVKSGCondEntropy``
- ``xyz.MVKSGCondMutualInformation``
- ``xyz.MVKSGTransferEntropy``
- ``xyz.KSGTransferEntropy``
- ``xyz.KSGPartialTransferEntropy``
- ``xyz.KSGSelfEntropy``
- ``xyz.MVKSGPartialInformationDecomposition``

Core mathematical idea
----------------------

KSG estimators build on the Kozachenko-Leonenko nearest-neighbor entropy
estimator. For each sample:

1. find the distance :math:`\varepsilon_i` to the :math:`k`-th nearest
   neighbor in a joint space,
2. project the same radius into lower-dimensional marginal spaces,
3. count how many samples fall inside the projected neighborhoods,
4. combine those counts with digamma functions to estimate entropy differences.

For mutual information, the classic KSG form is

.. math::

   \hat{I}(X;Y)
   = \psi(k) + \psi(N)
     - \frac{1}{N}\sum_{i=1}^{N}\left[\psi(n_x(i)) + \psi(n_y(i))\right],

where :math:`n_x(i)` and :math:`n_y(i)` are projected neighbor counts.

For transfer entropy, the same principle is applied to embedded state spaces:

.. math::

   TE_{X \to Y}
   = H(Y_t \mid Y_t^-)
   - H(Y_t \mid Y_t^-, X_t^-).

In practice, ``xyz`` estimates this from neighborhood counts in

- :math:`(Y_t, Y_t^-)`,
- :math:`(Y_t, Y_t^-, X_t^-)`,
- and the corresponding reduced conditioning spaces.

Why use KSG
-----------

- It is nonparametric and can capture nonlinear dependence missed by Gaussian
  estimators.
- It preserves the information-theoretic interpretation directly in continuous
  spaces.
- It is the most natural choice when you want scientific comparability with
  ITS/TRENTOOL-style continuous TE.

When to prefer KSG
------------------

Use KSG when:

- you suspect nonlinear coupling,
- you have enough samples to support neighborhood estimation,
- and you care more about flexible dependence detection than about speed.

Typical use cases
-----------------

- Neuroscience: directed functional connectivity in nonlinear neural signals.
- Finance: nonlinear information flow between assets, factors, or volatility
  states.
- Dynamical systems: causal coupling between chaotic or weakly nonlinear
  processes.

How to use the KSG estimators
-----------------------------

.. code-block:: python

   import numpy as np
   from xyz import (
       KSGEntropy,
       KSGMutualInformation,
       KSGPartialTransferEntropy,
       KSGTransferEntropy,
   )

   X = np.random.randn(1000, 1)
   Y = X + 0.2 * np.random.randn(1000, 1)
   mi = KSGMutualInformation(k=3).fit(X, Y).score()

   data = np.random.randn(1500, 3)
   te = KSGTransferEntropy(
       driver_indices=[0],
       target_indices=[1],
       lags=1,
       tau=1,
       delay=1,
       k=3,
       metric="chebyshev",
   ).fit(data)

   pte = KSGPartialTransferEntropy(
       driver_indices=[0],
       target_indices=[1],
       conditioning_indices=[2],
       lags=1,
       k=3,
   ).fit(data)

   print(mi)
   print(te.transfer_entropy_, te.conditional_entropy_)
   print(pte.transfer_entropy_)

Parameter guidance
------------------

- ``k``:
  smaller values reduce smoothing bias but increase variance; typical values
  are 3 to 10.
- ``metric``:
  use ``"chebyshev"`` for ITS-style comparability.
- ``lags`` and ``tau``:
  larger embeddings can represent richer dynamics, but quickly increase the
  dimensionality burden.
- ``delay``:
  use this explicitly when scanning interaction delays.

Important caveats
-----------------

- Finite-sample estimates can be slightly negative.
- Performance deteriorates as the effective embedding dimension grows.
- Results can be sensitive to repeated samples, ties, and insufficient
  neighborhood support.

ITS parity note
---------------

For the TE/PTE/SE parity tests in this project, self-neighbors are excluded
from projected counts, matching the ITS behavior of
``range_search(..., past=0)``.

Interactive example
-------------------

The plot below shows KSG transfer entropy as a function of the neighborhood
size :math:`k` for a nonlinear driver-target system. The point is not that a
single :math:`k` is universally best, but that useful estimates should remain
qualitatively stable across a sensible range.

.. plotly-exec::

   import numpy as np
   import plotly.graph_objects as go
   from xyz import KSGTransferEntropy

   rng = np.random.default_rng(12)
   n = 700
   driver = rng.normal(size=n)
   target = np.zeros(n)
   for t in range(1, n):
       target[t] = 0.35 * target[t - 1] + 0.25 * np.tanh(1.5 * driver[t - 1]) + 0.10 * rng.normal()

   data = np.column_stack([target, driver])
   ks = [2, 3, 4, 5, 6, 8, 10]
   te_vals = []
   for k in ks:
       est = KSGTransferEntropy(
           driver_indices=[1],
           target_indices=[0],
           lags=1,
           k=k,
           metric="chebyshev",
       ).fit(data)
       te_vals.append(est.transfer_entropy_)

   fig = go.Figure()
   fig.add_trace(
       go.Scatter(
           x=ks,
           y=te_vals,
           mode="lines+markers",
           name="KSG TE",
           line=dict(width=3),
       )
   )
   fig.update_layout(
       title="KSG transfer entropy sensitivity to neighborhood size",
       xaxis_title="k nearest neighbors",
       yaxis_title="Transfer entropy (nats)",
       template="plotly_white",
       height=420,
       margin=dict(l=40, r=20, t=60, b=40),
   )
