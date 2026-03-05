Finance workflow example
========================

This example illustrates a practical workflow for return-based information
dynamics. In finance, ``xyz`` is often most useful when framed as either:

- a **hedge-fund research** toolkit for testing whether a signal contains
  incremental predictive information about future returns;
- a **market microstructure** toolkit for measuring directional information flow
  between order-book state variables, trading activity, and short-horizon price
  changes.

Problem setup
-------------

- ``Y``: target return (asset or portfolio).
- ``X``: candidate driver (factor, market index, signal).
- ``Z``: controls/confounders (other factors, volatility proxies, sector index).

Hedge-fund research framing
---------------------------

For discretionary or systematic research, the core objects often look like:

- ``Y``: next-period return for an asset, spread, or portfolio sleeve.
- ``X``: candidate alpha source such as carry, momentum, revisions, macro
  surprises, option-implied variables, or cross-asset moves.
- ``Z``: benchmark risk controls such as market beta, sector indices, rates,
  credit spreads, volatility proxies, or competing signals.

Typical questions include:

- Does a candidate signal still add information after controlling for known
  factors?
- Is an apparent lead-lag effect still present after conditioning on the
  target's own past?
- Are two features redundant, uniquely informative, or synergistic when
  predicting the same target?
- Does a relationship persist across rolling windows or disappear outside of a
  specific regime?

Suggested sequence
------------------

1. **Baseline linear inference**

   - Fit ``GaussianTransferEntropy`` and ``GaussianPartialTransferEntropy``.
   - Use ``p_value_`` for quick significance screening.

2. **Nonparametric confirmation**

   - Fit ``KSGTransferEntropy`` / ``KSGPartialTransferEntropy``.
   - Check robustness over ``k`` (for example 3, 5, 8).

3. **Storage diagnostics**

   - Fit ``KSGSelfEntropy`` (or Gaussian/Kernel variants) for persistence in ``Y``.

4. **Sensitivity analysis**

   - Vary lags/embedding size.
   - Compare metrics (Chebyshev vs Euclidean if needed).
   - Validate on rolling windows for nonstationarity.
   - Treat significance testing as part of the workflow, not an optional extra.

Useful interpretations
----------------------

- ``GaussianTransferEntropy`` / ``KSGTransferEntropy``:
  "Does signal ``X`` improve prediction of future ``Y`` beyond ``Y``'s own
  history?"
- ``GaussianPartialTransferEntropy`` / ``KSGPartialTransferEntropy``:
  "Does that directed relationship survive after controlling for known risk
  drivers ``Z``?"
- ``KSGSelfEntropy``:
  "How much of the target is explained by its own lagged state?"
- ``MVKSGCondMutualInformation``:
  "Is this feature still informative after controlling for existing factors?"
- ``MVKSGPartialInformationDecomposition``:
  "Are two signals redundant, unique, or synergistic?"

Code sketch: hedge-fund research
--------------------------------

.. code-block:: python

   import numpy as np
   from xyz import (
       GaussianPartialTransferEntropy,
       KSGPartialTransferEntropy,
       KSGSelfEntropy,
   )

   # data columns: [target_return, candidate_signal, control_factor]
   data = np.random.randn(3000, 3)

   gpte = GaussianPartialTransferEntropy(
       driver_indices=[1],
       target_indices=[0],
       conditioning_indices=[2],
       lags=1,
   ).fit(data)

   kpte = KSGPartialTransferEntropy(
       driver_indices=[1],
       target_indices=[0],
       conditioning_indices=[2],
       lags=1,
       k=3,
       metric="chebyshev",
   ).fit(data)

   kse = KSGSelfEntropy(target_indices=[0], lags=1, k=3).fit(data[:, [0]])

   print("Gaussian PTE:", gpte.transfer_entropy_, "p=", gpte.p_value_)
   print("KSG PTE:", kpte.transfer_entropy_)
   print("KSG SE:", kse.self_entropy_)

Market microstructure framing
-----------------------------

At higher frequency, replace asset-level factors with order-book and trade state
variables:

- ``Y``: next mid-price move, short-horizon return, spread change, or volatility
  burst.
- ``X``: order-flow imbalance, trade sign, queue depletion, cancellation bursts,
  depth imbalance, or venue-specific activity.
- ``Z``: own-price history, spread regime, market state, auction flags, or
  broad benchmark features.

This framing is useful for questions such as:

- whether order-flow imbalance contains incremental predictive information about
  the next price move;
- whether one venue systematically leads another in a fragmented market;
- whether a richer book feature adds unique information beyond simple trade
  imbalance;
- whether naturally bucketed states are more transparent to study with discrete
  estimators before using continuous KSG variants.

Code sketch: discrete microstructure states
-------------------------------------------

.. code-block:: python

   import numpy as np
   from xyz import DiscretePartialTransferEntropy, DiscreteSelfEntropy

   # data columns: [midprice_state, imbalance_state, spread_state]
   # each column can already be integer-coded, e.g. 0=down, 1=flat, 2=up
   states = np.random.randint(0, 3, size=(5000, 3))

   dpte = DiscretePartialTransferEntropy(
       driver_indices=[1],
       target_indices=[0],
       conditioning_indices=[2],
       lags=1,
       tau=1,
       delay=1,
       quantize=False,
   ).fit(states)

   dse = DiscreteSelfEntropy(
       target_indices=[0],
       lags=1,
       tau=1,
       quantize=False,
   ).fit(states[:, [0]])

   print("Discrete PTE:", dpte.transfer_entropy_)
   print("Discrete SE:", dse.self_entropy_)

Practical notes
---------------

- Start simple: Gaussian estimators are often the best first screen when sample
  size is limited and the goal is ranking relationships quickly.
- Use KSG estimators when you suspect nonlinear, heavy-tailed, or regime-shaped
  dependencies.
- Use ``InteractionDelaySearchCV`` and ``RagwitzEmbeddingSearchCV`` when the
  timing of the interaction is part of the research question.
- Use ``SurrogatePermutationTest`` before trusting a weak TE edge in a noisy
  financial system.
- Prefer rolling or event-conditioned analyses over a single full-sample fit on
  nonstationary market data.
