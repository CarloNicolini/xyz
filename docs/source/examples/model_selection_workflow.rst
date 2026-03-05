Model selection workflow
========================

This page demonstrates how the sklearn-style meta-estimators in ``xyz`` can be
used to search embedding settings and interaction delays before running a final
TE analysis.

Why this workflow exists
------------------------

TRENTOOL-style TE analysis is not only about the low-level estimator. It also
depends on:

- choosing a sensible embedding dimension,
- choosing an embedding spacing,
- and choosing a plausible interaction delay.

The ``xyz`` search classes make these choices explicit and reproducible in a
Pythonic, scikit-learn-like form.

Example: embedding and delay search
-----------------------------------

.. code-block:: python

   import numpy as np
   from xyz import (
       GaussianTransferEntropy,
       InteractionDelaySearchCV,
       RagwitzEmbeddingSearchCV,
   )

   rng = np.random.default_rng(123)
   n = 700
   driver = rng.normal(size=n)
   target = np.zeros(n)
   for t in range(2, n):
       target[t] = 0.45 * target[t - 1] + 0.20 * target[t - 2] + 0.35 * driver[t - 2] + 0.1 * rng.normal()

   data = np.column_stack([target, driver])

   base = GaussianTransferEntropy(driver_indices=[1], target_indices=[0], lags=1)

   embedding = RagwitzEmbeddingSearchCV(
       base,
       target_index=0,
       dimensions=(1, 2, 3),
       taus=(1, 2, 3),
   ).fit(data)

   delay = InteractionDelaySearchCV(
       base.set_params(**embedding.best_params_),
       delays=(1, 2, 3, 4, 5),
   ).fit(data)

   print(embedding.best_params_, embedding.best_score_)
   print(delay.best_delay_, delay.best_score_)

Interactive example
-------------------

The two figures below show:

1. a heatmap of the Ragwitz-style embedding search surface,
2. a delay profile after fixing the best embedding.

.. plotly-exec::

   import numpy as np
   import plotly.graph_objects as go
   from plotly.subplots import make_subplots
   from xyz import GaussianTransferEntropy, InteractionDelaySearchCV, RagwitzEmbeddingSearchCV

   rng = np.random.default_rng(22)
   n = 800
   true_delay = 3
   driver = rng.normal(size=n)
   target = np.zeros(n)
   for t in range(true_delay, n):
       target[t] = (
           0.45 * target[t - 1]
           + 0.18 * target[t - 2]
           + 0.30 * driver[t - true_delay]
           + 0.10 * rng.normal()
       )

   data = np.column_stack([target, driver])
   base = GaussianTransferEntropy(driver_indices=[1], target_indices=[0], lags=1)

   dimensions = (1, 2, 3, 4)
   taus = (1, 2, 3)
   embedding = RagwitzEmbeddingSearchCV(
       base,
       target_index=0,
       dimensions=dimensions,
       taus=taus,
   ).fit(data)

   error_grid = np.asarray(embedding.cv_results_["mean_prediction_error"]).reshape(len(dimensions), len(taus))

   delay = InteractionDelaySearchCV(
       base.set_params(**embedding.best_params_),
       delays=(1, 2, 3, 4, 5, 6),
   ).fit(data)

   delay_x = delay.te_by_delay_[:, 0]
   delay_y = delay.te_by_delay_[:, 1]

   fig = make_subplots(
       rows=1,
       cols=2,
       subplot_titles=("Embedding search surface", "Delay reconstruction"),
   )

   fig.add_trace(
       go.Heatmap(
           x=list(taus),
           y=list(dimensions),
           z=error_grid,
           colorbar=dict(title="Prediction error"),
           hovertemplate="tau=%{x}<br>lags=%{y}<br>error=%{z:.4f}<extra></extra>",
       ),
       row=1,
       col=1,
   )

   fig.add_trace(
       go.Scatter(
           x=delay_x,
           y=delay_y,
           mode="lines+markers",
           name="TE by delay",
       ),
       row=1,
       col=2,
   )

   fig.add_vline(
       x=true_delay,
       line_dash="dash",
       annotation_text="True delay",
       row=1,
       col=2,
   )

   fig.update_xaxes(title_text="tau", row=1, col=1)
   fig.update_yaxes(title_text="lags", row=1, col=1)
   fig.update_xaxes(title_text="delay", row=1, col=2)
   fig.update_yaxes(title_text="transfer entropy (nats)", row=1, col=2)
   fig.update_layout(
       title="Model selection workflow for transfer entropy",
       template="plotly_white",
       height=460,
       margin=dict(l=40, r=20, t=70, b=40),
   )

Interpretation
--------------

- A smooth embedding surface is usually easier to trust than a highly erratic
  one.
- Delay reconstruction is most convincing when the TE profile has a clear and
  interpretable maximum.
- In real data, do not rely on model selection alone; combine it with
  significance testing and domain knowledge.
