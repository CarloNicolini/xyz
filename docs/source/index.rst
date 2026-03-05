xyz documentation
=================

Information-theoretic estimators for continuous and time-series data.

The library provides entropy, mutual information, transfer entropy, and related estimators
with both parametric (Gaussian) and non-parametric (KSG/kNN, kernel) approaches.

Interactive finance demo
------------------------

The chart below is generated at build time with executable code and embedded
as interactive Plotly HTML.

.. plotly-exec::

   import numpy as np
   import plotly.graph_objects as go

   rng = np.random.default_rng(123)
   n = 420
   market = 0.0008 + 0.010 * rng.standard_normal(n)
   factor = 0.0004 + 0.20 * np.roll(market, 1) + 0.0095 * rng.standard_normal(n)
   factor[0] = 0.0004 + 0.0095 * rng.standard_normal()

   x = np.arange(n)
   fig = go.Figure()
   fig.add_trace(go.Scatter(x=x, y=market, mode="lines", name="Market returns"))
   fig.add_trace(go.Scatter(x=x, y=factor, mode="lines", name="Factor returns"))
   fig.update_layout(
       title="Synthetic daily returns used in documentation examples",
       xaxis_title="Time",
       yaxis_title="Return",
       template="plotly_white",
       height=380,
       margin=dict(l=40, r=20, t=60, b=40),
   )

The finance examples in this documentation frame ``xyz`` in two practical ways:
as a hedge-fund research toolkit for testing incremental predictive content, and
as a market microstructure toolkit for studying directional information flow in
order-book and trade-derived state variables.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart
   theory
   estimators/index
   examples/index
   development

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/index
