from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone

from .preprocessing import (
    estimate_trial_acts,
    ragwitz_prediction_error,
    select_trials_by_act,
)


@dataclass(frozen=True)
class _SearchResult:
    params: dict[str, int]
    score: float
    raw_score: float


def _evaluate_ragwitz_candidate(
    trials: np.ndarray,
    *,
    target_index: int,
    dim: int,
    tau: int,
    k_neighbors: int,
    theiler_t: int,
    prediction_horizon: int,
    metric: str,
) -> _SearchResult:
    errors = [
        ragwitz_prediction_error(
            trial[:, target_index],
            dim=int(dim),
            tau=int(tau),
            k_neighbors=k_neighbors,
            theiler_t=theiler_t,
            prediction_horizon=prediction_horizon,
            metric=metric,
        )
        for trial in trials
    ]
    mean_error = float(np.mean(errors))
    return _SearchResult(
        params={"lags": int(dim), "tau": int(tau)},
        score=-mean_error,
        raw_score=mean_error,
    )


def _evaluate_delay_candidate(estimator, delay: int, X, y=None):
    est = clone(estimator).set_params(delay=int(delay))
    est.fit(X, y)
    return int(delay), float(est.score()), est


def _evaluate_source_candidate(estimator, sources: list[int], X, y=None):
    est = clone(estimator).set_params(driver_indices=list(sources))
    est.fit(X, y)
    return float(est.score()), est


class RagwitzEmbeddingSearchCV(MetaEstimatorMixin, BaseEstimator):
    """Search embedding (dimension, tau) using the Ragwitz prediction-error criterion.

    Evaluates (dim, tau) candidates via :func:`~xyz.preprocessing.ragwitz_prediction_error`
    and selects the pair that minimizes mean prediction error across trials.
    Optionally filters trials by autocorrelation decay time (ACT).

    Parameters
    ----------
    estimator : object
        TE estimator to tune (e.g. :class:`xyz.KSGTransferEntropy`).
    target_index : int
        Column index of the target variable.
    dimensions : tuple of int, optional
        Embedding dimensions to try. Default is (1, 2, 3).
    taus : tuple of int, optional
        Embedding delays (samples) to try. Default is (1, 2, 3).
    k_neighbors : int, optional
        k for local prediction in Ragwitz criterion. Default is 4.
    theiler_t : int, optional
        Theiler window. Default is 0.
    prediction_horizon : int, optional
        Steps ahead for prediction. Default is 1.
    metric : str, optional
        Distance metric. Default is ``"chebyshev"``.
    act_threshold : int or None, optional
        If set, keep only trials with ACT <= this. Default is None.
    max_act_lag : int, optional
        Max lag for ACT estimation. Default is 1000.
    min_trials : int, optional
        Minimum trials after ACT filtering. Default is 1.
    refit : bool, optional
        If True, fit best_estimator_ with best params. Default is True.
    n_jobs : int or None, optional
        Parallel jobs. Default is 1.

    Attributes
    ----------
    best_params_ : dict
        Best ``lags`` and ``tau``.
    best_score_ : float
        Best criterion score (negative prediction error).
    best_estimator_ : estimator
        Fitted estimator with best params (if refit=True).
    cv_results_ : dict
        ``params``, ``mean_test_score``, ``mean_prediction_error``.

    Examples
    --------
    >>> import numpy as np
    >>> from xyz import KSGTransferEntropy, RagwitzEmbeddingSearchCV
    >>> rng = np.random.default_rng(7)
    >>> trials = []
    >>> for _ in range(4):
    ...     driver = rng.normal(size=250)
    ...     target = np.zeros(250)
    ...     for t in range(3, 250):
    ...         target[t] = 0.55 * target[t-1] + 0.25 * target[t-3] + 0.2 * driver[t-1] + 0.1 * rng.normal()
    ...     trials.append(np.column_stack([target, driver]))
    >>> X = np.stack(trials)
    >>> search = RagwitzEmbeddingSearchCV(
    ...     KSGTransferEntropy(driver_indices=[1], target_indices=[0], k=3),
    ...     target_index=0, dimensions=(1, 2, 3), taus=(1, 2),
    ... ).fit(X)
    >>> "lags" in search.best_params_ and "tau" in search.best_params_
    True
    """

    def __init__(
        self,
        estimator,
        *,
        target_index: int,
        dimensions=(1, 2, 3),
        taus=(1, 2, 3),
        k_neighbors: int = 4,
        theiler_t: int = 0,
        prediction_horizon: int = 1,
        metric: str = "chebyshev",
        act_threshold: int | None = None,
        max_act_lag: int = 1000,
        min_trials: int = 1,
        refit: bool = True,
        n_jobs: int | None = 1,
    ):
        self.estimator = estimator
        self.target_index = target_index
        self.dimensions = dimensions
        self.taus = taus
        self.k_neighbors = k_neighbors
        self.theiler_t = theiler_t
        self.prediction_horizon = prediction_horizon
        self.metric = metric
        self.act_threshold = act_threshold
        self.max_act_lag = max_act_lag
        self.min_trials = min_trials
        self.refit = refit
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        selected_trials, acts = select_trials_by_act(
            X,
            target_index=self.target_index,
            max_lag=self.max_act_lag,
            act_threshold=self.act_threshold,
            min_trials=self.min_trials,
        )
        self.act_values_ = acts
        self.selected_trial_count_ = selected_trials.shape[0]

        candidates = [(int(dim), int(tau)) for dim in self.dimensions for tau in self.taus]
        results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_evaluate_ragwitz_candidate)(
                selected_trials,
                target_index=self.target_index,
                dim=dim,
                tau=tau,
                k_neighbors=self.k_neighbors,
                theiler_t=self.theiler_t,
                prediction_horizon=self.prediction_horizon,
                metric=self.metric,
            )
            for dim, tau in candidates
        )

        best = max(results, key=lambda item: (item.score, -item.params["lags"], -item.params["tau"]))
        self.cv_results_ = {
            "params": [result.params for result in results],
            "mean_test_score": np.array([result.score for result in results], dtype=float),
            "mean_prediction_error": np.array([result.raw_score for result in results], dtype=float),
        }
        self.best_params_ = dict(best.params)
        self.best_score_ = float(best.score)
        self.best_prediction_error_ = float(best.raw_score)

        if self.refit:
            self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
            self.best_estimator_.fit(selected_trials if selected_trials.shape[0] > 1 else selected_trials[0], y)
        return self

    def score(self, X=None, y=None):
        if X is not None:
            self.fit(X, y)
        return float(self.best_score_)


class InteractionDelaySearchCV(MetaEstimatorMixin, BaseEstimator):
    """Search interaction delay for a TE estimator over a set of candidate delays.

    Fits the estimator for each delay and selects the delay that maximizes the
    TE score (or minimizes, depending on estimator). Optionally refits the
    best estimator.

    Parameters
    ----------
    estimator : object
        TE estimator with a ``delay`` parameter.
    delays : array-like
        Candidate delay values (samples) to try.
    refit : bool, optional
        If True, fit best_estimator_ with best delay. Default is True.
    tie_break : str, optional
        ``"smallest"`` or ``"largest"`` when multiple delays tie. Default is ``"smallest"``.
    n_jobs : int or None, optional
        Parallel jobs. Default is 1.

    Attributes
    ----------
    best_delay_ : int
        Selected delay.
    best_score_ : float
        TE score at best delay.
    best_estimator_ : estimator
        Fitted estimator at best delay (if refit=True).
    delay_curve_ : dict
        Mapping delay -> score.
    cv_results_ : dict
        ``params``, ``mean_test_score``.
    """

    def __init__(
        self,
        estimator,
        *,
        delays,
        refit: bool = True,
        tie_break: str = "smallest",
        n_jobs: int | None = 1,
    ):
        self.estimator = estimator
        self.delays = delays
        self.refit = refit
        self.tie_break = tie_break
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_evaluate_delay_candidate)(self.estimator, int(delay), X, y) for delay in self.delays
        )

        self.delay_curve_ = {delay: score for delay, score, _ in results}
        scores = np.array([score for _, score, _ in results], dtype=float)
        delays = np.array([delay for delay, _, _ in results], dtype=int)
        best_score = np.max(scores)
        candidate_delays = delays[np.isclose(scores, best_score)]
        if self.tie_break == "smallest":
            best_delay = int(np.min(candidate_delays))
        elif self.tie_break == "largest":
            best_delay = int(np.max(candidate_delays))
        else:
            raise ValueError("tie_break must be 'smallest' or 'largest'")

        best_idx = next(i for i, (delay, _, _) in enumerate(results) if delay == best_delay)
        self.best_delay_ = best_delay
        self.best_params_ = {"delay": best_delay}
        self.best_score_ = float(best_score)
        self.cv_results_ = {
            "params": [{"delay": delay} for delay, _, _ in results],
            "mean_test_score": scores,
        }
        self.te_by_delay_ = np.array([[delay, score] for delay, score, _ in results], dtype=float)

        if self.refit:
            self.best_estimator_ = clone(self.estimator).set_params(delay=best_delay)
            self.best_estimator_.fit(X, y)
        else:
            self.best_estimator_ = results[best_idx][2]
        return self

    def score(self, X=None, y=None):
        if X is not None:
            self.fit(X, y)
        return float(self.best_score_)


class EnsembleTransferEntropy(MetaEstimatorMixin, BaseEstimator):
    """Wrapper that fits a TE estimator on multi-trial data.

    Passes trial-shaped data to the underlying estimator so it can respect
    trial boundaries (e.g. for KSG within-trial neighbor search).

    Parameters
    ----------
    estimator : object
        TE estimator with ``fit(X, y=None)`` and ``score()``.

    Attributes
    ----------
    estimator_ : estimator
        Fitted clone of the wrapped estimator.
    score_ : float
        TE score from the fitted estimator.

    Examples
    --------
    >>> import numpy as np
    >>> from xyz import EnsembleTransferEntropy, KSGTransferEntropy
    >>> X = np.random.randn(3, 200, 2)  # 3 trials
    >>> meta = EnsembleTransferEntropy(
    ...     KSGTransferEntropy(driver_indices=[1], target_indices=[0], lags=1),
    ... ).fit(X)
    >>> np.isfinite(meta.score())
    True
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        self.score_ = float(self.estimator_.score())
        return self

    def score(self, X=None, y=None):
        if X is not None:
            self.fit(X, y)
        return float(self.score_)


class GroupTEAnalysis(MetaEstimatorMixin, BaseEstimator):
    """Group-level TE: Ragwitz search per subject, then common embedding and aggregate.

    For each dataset in ``datasets``, runs :class:`RagwitzEmbeddingSearchCV` to find
    best (lags, tau). Then takes the maximum dimension and tau across subjects,
    refits each subject with that common embedding, and aggregates scores (mean or median).

    Parameters
    ----------
    estimator : object
        TE estimator to use (e.g. :class:`xyz.KSGTransferEntropy`).
    target_index : int
        Column index of the target.
    dimensions : tuple of int, optional
        Ragwitz dimension candidates. Default is (1, 2, 3).
    taus : tuple of int, optional
        Ragwitz tau candidates. Default is (1, 2, 3).
    aggregation : str, optional
        ``"mean"`` or ``"median"`` for group score. Default is ``"mean"``.

    Attributes
    ----------
    common_params_ : dict
        Common ``lags`` and ``tau`` used for all subjects.
    subject_scores_ : np.ndarray
        TE score per subject.
    group_score_ : float
        Aggregated (mean or median) group score.
    embedding_searches_ : list
        RagwitzEmbeddingSearchCV result per subject.
    """

    def __init__(
        self,
        estimator,
        *,
        target_index: int,
        dimensions=(1, 2, 3),
        taus=(1, 2, 3),
        aggregation: str = "mean",
    ):
        self.estimator = estimator
        self.target_index = target_index
        self.dimensions = dimensions
        self.taus = taus
        self.aggregation = aggregation

    def fit(self, datasets, y=None):
        searches = []
        max_dim = 1
        max_tau = 1
        for dataset in datasets:
            search = RagwitzEmbeddingSearchCV(
                self.estimator,
                target_index=self.target_index,
                dimensions=self.dimensions,
                taus=self.taus,
            ).fit(dataset)
            searches.append(search)
            max_dim = max(max_dim, search.best_params_["lags"])
            max_tau = max(max_tau, search.best_params_["tau"])

        self.embedding_searches_ = searches
        self.common_params_ = {"lags": max_dim, "tau": max_tau}
        self.subject_estimators_ = []
        self.subject_scores_ = []
        for dataset in datasets:
            est = clone(self.estimator).set_params(**self.common_params_)
            est.fit(dataset)
            self.subject_estimators_.append(est)
            self.subject_scores_.append(float(est.score()))

        self.subject_scores_ = np.asarray(self.subject_scores_, dtype=float)
        if self.aggregation == "mean":
            self.group_score_ = float(np.mean(self.subject_scores_))
        elif self.aggregation == "median":
            self.group_score_ = float(np.median(self.subject_scores_))
        else:
            raise ValueError("aggregation must be 'mean' or 'median'")
        return self

    def score(self, X=None, y=None):
        if X is not None:
            self.fit(X, y)
        return float(self.group_score_)


class GreedySourceSelectionTransferEntropy(MetaEstimatorMixin, BaseEstimator):
    """Greedy forward selection of driver sources for partial TE.

    Starts from the estimator's existing conditioning set and adds driver
    sources one at a time from ``candidate_sources``, keeping the one that
    increases the TE score most. Stops when no improvement or max_sources reached.

    Parameters
    ----------
    estimator : object
        Partial TE estimator with ``driver_indices`` and ``conditioning_indices``.
    candidate_sources : array-like
        Column indices of candidate driver sources.
    max_sources : int or None, optional
        Maximum number of sources to add. None = no limit. Default is None.
    min_improvement : float, optional
        Stop if improvement is <= this. Default is 0.0.
    n_jobs : int or None, optional
        Parallel jobs for evaluating candidate sets. Default is 1.
    refit : bool, optional
        If True, best_estimator_ is fitted with selected sources. Default is True.

    Attributes
    ----------
    selected_sources_ : list of int
        Indices of selected driver sources.
    selection_history_ : list
        History of (sources, score) along the greedy path.
    best_estimator_ : estimator
        Fitted estimator with selected sources (if refit=True).
    """

    def __init__(
        self,
        estimator,
        *,
        candidate_sources,
        max_sources: int | None = None,
        min_improvement: float = 0.0,
        n_jobs: int | None = 1,
        refit: bool = True,
    ):
        self.estimator = estimator
        self.candidate_sources = candidate_sources
        self.max_sources = max_sources
        self.min_improvement = min_improvement
        self.n_jobs = n_jobs
        self.refit = refit

    def fit(self, X, y=None):
        if not hasattr(self.estimator, "driver_indices") or not hasattr(self.estimator, "conditioning_indices"):
            raise ValueError(
                "GreedySourceSelectionTransferEntropy requires a partial TE estimator "
                "with driver_indices and conditioning_indices parameters"
            )

        target_indices = [int(index) for index in getattr(self.estimator, "target_indices")]
        base_conditioning = [int(index) for index in getattr(self.estimator, "conditioning_indices", [])]
        selected_sources: list[int] = []
        current_score = 0.0
        self.selection_history_ = []

        remaining = []
        for source in self.candidate_sources:
            source = int(source)
            if source in target_indices or source in base_conditioning or source in remaining:
                continue
            remaining.append(source)

        max_sources = len(remaining) if self.max_sources is None else int(self.max_sources)

        while remaining and len(selected_sources) < max_sources:
            candidate_sets = [selected_sources + [candidate] for candidate in remaining]
            results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(_evaluate_source_candidate)(self.estimator, sources, X, y)
                for sources in candidate_sets
            )
            candidate_scores = [score for score, _ in results]
            best_idx = int(np.argmax(candidate_scores))
            best_sources = candidate_sets[best_idx]
            best_candidate = best_sources[-1]
            best_score = float(candidate_scores[best_idx])
            improvement = best_score - current_score

            if improvement <= float(self.min_improvement):
                break

            selected_sources = best_sources
            current_score = best_score
            self.selection_history_.append(
                {
                    "source": best_candidate,
                    "selected_sources": list(selected_sources),
                    "score": best_score,
                    "improvement": improvement,
                }
            )
            remaining.remove(best_candidate)

        self.selected_sources_ = list(selected_sources)
        self.best_score_ = float(current_score)
        self.best_params_ = {"driver_indices": list(self.selected_sources_)}

        if self.refit and self.selected_sources_:
            self.best_estimator_ = clone(self.estimator).set_params(driver_indices=self.selected_sources_)
            self.best_estimator_.fit(X, y)
        else:
            self.best_estimator_ = None
        return self

    def score(self, X=None, y=None):
        if X is not None:
            self.fit(X, y)
        return float(self.best_score_)
