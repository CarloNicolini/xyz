Univariate helper functions
===========================

The module ``xyz.univariate`` contains lightweight function-based helpers.
These are useful for quick exploratory work, teaching, and sanity checks before
moving to the estimator classes.

Available functions
-------------------

- ``entropy_linear(A)``
- ``entropy_kernel(Y, r, metric="chebyshev")``
- ``entropy_binning(Y, c, quantize, log_base="nat")``

What they do
------------

``entropy_linear``
^^^^^^^^^^^^^^^^^^

Computes Gaussian entropy from the sample covariance matrix:

.. math::

   H(Y) = \frac{1}{2}\log\!\left((2\pi e)^d \det(\Sigma)\right).

Use it when a fast Gaussian baseline is enough.

``entropy_kernel``
^^^^^^^^^^^^^^^^^^

Approximates entropy from fixed-radius counts. It is useful when you want a
local geometric notion of uncertainty and are prepared to inspect sensitivity
with respect to ``r``.

``entropy_binning``
^^^^^^^^^^^^^^^^^^^

Approximates entropy by counting discrete states after optional quantization:

.. math::

   \hat{H}(Y) = - \sum_y \hat{p}(y)\log \hat{p}(y).

This is especially useful for symbolic or deliberately discretized processes.

How to use them
---------------

.. code-block:: python

   import numpy as np
   from xyz.univariate import entropy_binning, entropy_kernel, entropy_linear

   y = np.random.randn(2000, 1)

   print("Gaussian entropy:", entropy_linear(y))
   print("Kernel entropy:", entropy_kernel(y, r=0.3))
   print("Binned entropy:", entropy_binning(y.copy(), c=8, quantize=False))

When to use helper functions instead of estimators
--------------------------------------------------

- when you want a one-line numerical check,
- when model selection or fitted attributes are unnecessary,
- or when teaching the differences between Gaussian, kernel, and binned views
  of entropy.

Interactive example
-------------------

The figure below compares three univariate entropy estimates as the variance of
the underlying Gaussian signal changes.

.. plotly-exec::

   import numpy as np
   import plotly.graph_objects as go
   from xyz.univariate import entropy_kernel, entropy_linear

   rng = np.random.default_rng(18)
   sigmas = np.linspace(0.4, 2.0, 9)
   linear_vals = []
   kernel_vals = []
   theoretical_vals = []

   for sigma in sigmas:
       y = rng.normal(0.0, sigma, size=(3000, 1))
       linear_vals.append(entropy_linear(y))
       kernel_vals.append(entropy_kernel(y, r=0.35))
       theoretical_vals.append(0.5 * np.log(2 * np.pi * np.e * sigma**2))

   fig = go.Figure()
   fig.add_trace(go.Scatter(x=sigmas, y=theoretical_vals, mode="lines", name="Theoretical"))
   fig.add_trace(go.Scatter(x=sigmas, y=linear_vals, mode="lines+markers", name="Gaussian helper"))
   fig.add_trace(go.Scatter(x=sigmas, y=kernel_vals, mode="lines+markers", name="Kernel helper"))
   fig.update_layout(
       title="Univariate entropy estimates across signal variance",
       xaxis_title="Standard deviation",
       yaxis_title="Entropy (nats)",
       template="plotly_white",
       height=420,
       margin=dict(l=40, r=20, t=60, b=40),
   )
