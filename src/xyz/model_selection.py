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
    """Search embedding settings using a Ragwitz-style prediction criterion."""

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
    """Search interaction delays for a TE estimator."""

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
    """Fit a TE estimator on multi-trial data without concatenation artifacts."""

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
    """Group-level TE workflow with common embedding harmonization."""

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
    """Greedy forward source selection for partial transfer entropy estimators."""

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
