Interactive finance examples (Plotly)
=====================================

This page runs real Python code at documentation build time and embeds
interactive Plotly charts directly in the generated HTML.

Synthetic market and strategy returns
-------------------------------------

The following chart simulates a market return stream with one strategy that
contains a small lagged predictive component from the market.

.. plotly-exec::

   import numpy as np
   import plotly.graph_objects as go

   rng = np.random.default_rng(42)
   n = 500
   market = 0.001 + 0.01 * rng.standard_normal(n)
   strategy = 0.0005 + 0.25 * np.roll(market, 1) + 0.009 * rng.standard_normal(n)
   strategy[0] = 0.0005 + 0.009 * rng.standard_normal()

   market_nav = np.cumprod(1.0 + market)
   strategy_nav = np.cumprod(1.0 + strategy)

   x = np.arange(n)
   fig = go.Figure()
   fig.add_trace(go.Scatter(x=x, y=market_nav, mode="lines", name="Market NAV"))
   fig.add_trace(go.Scatter(x=x, y=strategy_nav, mode="lines", name="Strategy NAV"))
   fig.update_layout(
       title="Synthetic cumulative returns",
       xaxis_title="Time",
       yaxis_title="Cumulative growth",
       template="plotly_white",
       height=420,
       margin=dict(l=40, r=20, t=60, b=40),
   )

Transfer entropy sensitivity to lag choice
------------------------------------------

The chart below computes ``TE(market -> strategy)`` with ``xyz`` over different
lag values using ``GaussianTransferEntropy`` and ``KSGTransferEntropy``.

.. plotly-exec::

   import numpy as np
   import plotly.graph_objects as go
   from xyz import GaussianTransferEntropy, KSGTransferEntropy

   rng = np.random.default_rng(7)
   n = 700
   market = 0.0 + 0.012 * rng.standard_normal(n)
   strategy = (
       0.15 * np.roll(market, 1)
       + 0.08 * np.roll(market, 2)
       + 0.011 * rng.standard_normal(n)
   )
   strategy[:2] = 0.011 * rng.standard_normal(2)
   data = np.column_stack([strategy, market])

   lags = [1, 2, 3, 4, 5]
   te_gauss = []
   te_ksg = []
   for lag in lags:
       g = GaussianTransferEntropy(driver_indices=[1], target_indices=[0], lags=lag).fit(data)
       k = KSGTransferEntropy(driver_indices=[1], target_indices=[0], lags=lag, k=3, metric="chebyshev").fit(data)
       te_gauss.append(g.transfer_entropy_)
       te_ksg.append(k.transfer_entropy_)

   fig = go.Figure()
   fig.add_trace(go.Scatter(x=lags, y=te_gauss, mode="lines+markers", name="Gaussian TE"))
   fig.add_trace(go.Scatter(x=lags, y=te_ksg, mode="lines+markers", name="KSG TE"))
   fig.update_layout(
       title="TE sensitivity across lag choice (synthetic data)",
       xaxis_title="Lag",
       yaxis_title="Estimated transfer entropy (nats)",
       template="plotly_white",
       height=420,
       margin=dict(l=40, r=20, t=60, b=40),
   )
