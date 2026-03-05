from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone

from .preprocessing import as_trial_array


def _ensure_rng(random_state=None):
    return np.random.default_rng(random_state)


def fdr_bh(p_values, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg false-discovery-rate correction."""
    p_values = np.asarray(p_values, dtype=float)
    flat = p_values.ravel()
    order = np.argsort(flat)
    ranked = flat[order]
    thresholds = alpha * (np.arange(1, ranked.size + 1) / ranked.size)
    passed = ranked <= thresholds
    if not np.any(passed):
        return np.zeros_like(p_values, dtype=bool)
    cutoff = ranked[np.max(np.where(passed))]
    return p_values <= cutoff


def bonferroni(p_values, alpha: float = 0.05) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    return p_values <= alpha / p_values.size


def generate_surrogates(
    X,
    *,
    method: str = "trial_shuffle",
    n_surrogates: int = 100,
    block_length: int | None = None,
    random_state=None,
    driver_index: int | None = None,
) -> list[np.ndarray]:
    """Generate surrogate datasets for TE null testing."""
    rng = _ensure_rng(random_state)
    trials = as_trial_array(X)
    surrogates = []
    driver_index = 0 if driver_index is None else int(driver_index)

    for _ in range(int(n_surrogates)):
        surrogate = np.array(trials, copy=True)
        if method == "trial_shuffle":
            perm = rng.permutation(surrogate.shape[0])
            surrogate[:, :, driver_index] = surrogate[perm, :, driver_index]
        elif method == "block_resample":
            for trial_idx in range(surrogate.shape[0]):
                n_samples = surrogate.shape[1]
                if n_samples < 4:
                    continue
                cut = rng.integers(1, n_samples - 1)
                surrogate[trial_idx] = np.vstack(
                    [surrogate[trial_idx, cut:], surrogate[trial_idx, :cut]]
                )
        elif method == "block_reverse":
            for trial_idx in range(surrogate.shape[0]):
                n_samples = surrogate.shape[1]
                if n_samples < 4:
                    continue
                cut = rng.integers(1, n_samples - 1)
                surrogate[trial_idx] = np.vstack(
                    [surrogate[trial_idx, cut:][::-1], surrogate[trial_idx, :cut][::-1]]
                )
        elif method == "swap_neighbors":
            for trial_idx in range(0, surrogate.shape[0] - 1, 2):
                temp = surrogate[trial_idx, :, driver_index].copy()
                surrogate[trial_idx, :, driver_index] = surrogate[trial_idx + 1, :, driver_index]
                surrogate[trial_idx + 1, :, driver_index] = temp
        elif method == "time_shift":
            for trial_idx in range(surrogate.shape[0]):
                n_samples = surrogate.shape[1]
                if block_length is None:
                    shift = max(1, n_samples // 10)
                else:
                    shift = int(block_length)
                shift = max(1, min(shift, n_samples - 1))
                surrogate[trial_idx, :, driver_index] = np.roll(
                    surrogate[trial_idx, :, driver_index], shift
                )
        else:
            raise ValueError(
                "method must be one of 'trial_shuffle', 'block_resample', "
                "'block_reverse', 'swap_neighbors', or 'time_shift'"
            )

        surrogates.append(surrogate if surrogate.shape[0] > 1 else surrogate[0])
    return surrogates


class SurrogatePermutationTest(MetaEstimatorMixin, BaseEstimator):
    """Permutation-based significance testing for TE estimators."""

    def __init__(
        self,
        estimator,
        *,
        n_permutations: int = 100,
        surrogate_method: str = "trial_shuffle",
        alpha: float = 0.05,
        correction: str = "fdr_bh",
        shift_test: bool = False,
        shift_method: str = "time_shift",
        random_state=None,
    ):
        self.estimator = estimator
        self.n_permutations = n_permutations
        self.surrogate_method = surrogate_method
        self.alpha = alpha
        self.correction = correction
        self.shift_test = shift_test
        self.shift_method = shift_method
        self.random_state = random_state

    def fit(self, X, y=None):
        if self.shift_test and getattr(self.estimator, "extra_conditioning", None) in {"Faes_Method", "faes"}:
            raise ValueError("Shift tests and Faes-style extra conditioning are mutually exclusive")

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        self.observed_score_ = float(self.estimator_.score())

        driver_index = None
        if hasattr(self.estimator_, "driver_indices"):
            driver_index = int(self.estimator_.driver_indices[0])

        surrogates = generate_surrogates(
            X,
            method=self.surrogate_method,
            n_surrogates=self.n_permutations,
            random_state=self.random_state,
            driver_index=driver_index,
        )
        surrogate_scores = []
        for surrogate in surrogates:
            est = clone(self.estimator)
            est.fit(surrogate, y)
            surrogate_scores.append(float(est.score()))
        self.null_distribution_ = np.asarray(surrogate_scores, dtype=float)
        self.surrogate_scores_ = self.null_distribution_.copy()
        self.p_values_ = np.asarray(
            [(1 + np.sum(self.null_distribution_ >= self.observed_score_)) / (self.n_permutations + 1)],
            dtype=float,
        )
        self.significant_ = self.p_values_ <= self.alpha

        if self.correction == "fdr_bh":
            self.corrected_significant_ = fdr_bh(self.p_values_, alpha=self.alpha)
        elif self.correction == "bonferroni":
            self.corrected_significant_ = bonferroni(self.p_values_, alpha=self.alpha)
        elif self.correction in {None, "none"}:
            self.corrected_significant_ = self.significant_.copy()
        else:
            raise ValueError("correction must be 'fdr_bh', 'bonferroni' or 'none'")

        if self.shift_test:
            shift_surrogate = generate_surrogates(
                X,
                method=self.shift_method,
                n_surrogates=1,
                random_state=self.random_state,
                driver_index=driver_index,
            )[0]
            shift_estimator = clone(self.estimator)
            shift_estimator.fit(shift_surrogate, y)
            self.shift_test_ = {
                "shift_score": float(shift_estimator.score()),
                "passed": bool(self.observed_score_ > float(shift_estimator.score())),
            }
        else:
            self.shift_test_ = None

        return self

    def score(self, X=None, y=None):
        if X is not None:
            self.fit(X, y)
        return float(self.observed_score_)
