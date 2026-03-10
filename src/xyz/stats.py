from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone

from .preprocessing import as_trial_array


def _ensure_rng(random_state=None):
    return np.random.default_rng(random_state)


def fdr_bh(p_values, alpha: float = 0.05) -> np.ndarray:
    """Benjamini–Hochberg false-discovery-rate correction.

    Rejects hypotheses with p-value :math:`\\le` the adaptive threshold so that
    the expected FDR is controlled at level ``alpha``.

    Parameters
    ----------
    p_values : array-like
        P-values (any shape).
    alpha : float, optional
        Target FDR level. Default is 0.05.

    Returns
    -------
    np.ndarray
        Boolean array of same shape as ``p_values``: True where the null is
        rejected.

    Examples
    --------
    >>> import numpy as np
    >>> from xyz.stats import fdr_bh
    >>> p = np.array([0.001, 0.02, 0.04, 0.15])
    >>> fdr_bh(p, alpha=0.05)
    array([ True,  True,  True, False])
    """
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
    """Bonferroni correction for multiple testing.

    Rejects where :math:`p_i \\le \\alpha / m` with :math:`m` the number of tests.
    Controls family-wise error rate at level ``alpha``.

    Parameters
    ----------
    p_values : array-like
        P-values (any shape).
    alpha : float, optional
        Family-wise error rate. Default is 0.05.

    Returns
    -------
    np.ndarray
        Boolean array: True where the null is rejected.

    Examples
    --------
    >>> import numpy as np
    >>> from xyz.stats import bonferroni
    >>> p = np.array([0.001, 0.02, 0.04])
    >>> bonferroni(p, alpha=0.05)
    array([ True, False, False])
    """
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
    """Generate surrogate datasets for transfer-entropy null testing.

    Surrogates break the driver–target relationship while preserving marginal
    structure. Used with :class:`SurrogatePermutationTest` to assess significance.

    Parameters
    ----------
    X : array-like
        Data, shape ``(n_trials, n_samples, n_features)`` or equivalent
        (see :func:`xyz.preprocessing.as_trial_array`).
    method : str, optional
        One of ``"trial_shuffle"`` (shuffle driver across trials), ``"block_resample"``,
        ``"block_reverse"``, ``"swap_neighbors"``, ``"time_shift"``. Default
        is ``"trial_shuffle"``.
    n_surrogates : int, optional
        Number of surrogate datasets. Default is 100.
    block_length : int or None, optional
        Used by ``block_*`` and ``time_shift`` methods. Default is None.
    random_state : int, array-like or None, optional
        Random seed or generator.
    driver_index : int or None, optional
        Column index of the driver variable (for methods that permute the driver).
        Default is 0 if None.

    Returns
    -------
    list of np.ndarray
        List of surrogate arrays; each has the same shape as the trial
        representation of ``X``.

    Examples
    --------
    >>> import numpy as np
    >>> from xyz import generate_surrogates
    >>> rng = np.random.default_rng(5)
    >>> X = rng.normal(size=(3, 40, 2))
    >>> surrogates = generate_surrogates(X, method="trial_shuffle", n_surrogates=5, random_state=0, driver_index=1)
    >>> len(surrogates)
    5
    >>> surrogates[0].shape == X.shape
    True
    """
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


def _score_surrogate(estimator, surrogate, y=None) -> float:
    est = clone(estimator)
    est.fit(surrogate, y)
    return float(est.score())


def _restore_trial_shape(trials: np.ndarray, original_ndim: int):
    if original_ndim == 1:
        return trials[0, :, 0]
    if original_ndim == 2:
        return trials[0]
    return trials


def _sample_block_indices(n_samples: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    indices = []
    while len(indices) < n_samples:
        start = int(rng.integers(0, n_samples))
        block = (start + np.arange(block_length)) % n_samples
        indices.extend(block.tolist())
    return np.asarray(indices[:n_samples], dtype=int)


def _bootstrap_sample(X, y, *, method: str, block_length: int | None, rng: np.random.Generator):
    X_array = np.asarray(X)
    y_array = None if y is None else np.asarray(y)

    if method == "iid":
        n_samples = X_array.shape[0]
        indices = rng.integers(0, n_samples, size=n_samples)
        X_sample = X_array[indices]
        y_sample = None if y is None else y_array[indices]
        return X_sample, y_sample

    if method == "trial":
        X_trials = as_trial_array(X_array)
        if X_trials.shape[0] < 2:
            raise ValueError("Trial bootstrap requires at least two trials")
        indices = rng.integers(0, X_trials.shape[0], size=X_trials.shape[0])
        X_sample = _restore_trial_shape(X_trials[indices], X_array.ndim)
        if y is None:
            return X_sample, None
        y_trials = as_trial_array(y_array)
        y_sample = _restore_trial_shape(y_trials[indices], y_array.ndim)
        return X_sample, y_sample

    if method == "block":
        X_trials = as_trial_array(X_array)
        block_length = max(2, int(block_length or max(2, X_trials.shape[1] // 10)))
        X_boot = np.empty_like(X_trials)
        y_trials = None if y is None else as_trial_array(y_array)
        y_boot = None if y is None else np.empty_like(y_trials)
        for trial_idx in range(X_trials.shape[0]):
            indices = _sample_block_indices(X_trials.shape[1], block_length, rng)
            X_boot[trial_idx] = X_trials[trial_idx, indices]
            if y_boot is not None:
                y_boot[trial_idx] = y_trials[trial_idx, indices]
        X_sample = _restore_trial_shape(X_boot, X_array.ndim)
        y_sample = None if y_boot is None else _restore_trial_shape(y_boot, y_array.ndim)
        return X_sample, y_sample

    raise ValueError("method must be one of 'iid', 'trial', or 'block'")


def _score_bootstrap_sample(estimator, X, y, *, method: str, block_length: int | None, seed: int) -> float:
    rng = _ensure_rng(seed)
    X_sample, y_sample = _bootstrap_sample(X, y, method=method, block_length=block_length, rng=rng)
    est = clone(estimator)
    est.fit(X_sample, y_sample)
    return float(est.score())


class BootstrapEstimate(MetaEstimatorMixin, BaseEstimator):
    """Bootstrap confidence intervals for information-theoretic estimators.

    Fits the wrapped estimator on the original data and on bootstrap resamples
    to obtain a distribution of the score and a confidence interval.

    Parameters
    ----------
    estimator : object
        An xyz estimator with ``fit`` and ``score`` (e.g. transfer entropy,
        mutual information).
    n_bootstrap : int, optional
        Number of bootstrap samples. Default is 100.
    method : str, optional
        Resampling: ``"iid"`` (sample rows with replacement), ``"trial"``
        (resample trials), ``"block"`` (block bootstrap within trials).
        Default is ``"iid"``.
    block_length : int or None, optional
        Block length for ``method="block"``. Default is None (auto).
    ci : float, optional
        Confidence level for the interval (e.g. 0.95). Default is 0.95.
    random_state : int, array-like or None, optional
        Random seed or generator.
    n_jobs : int or None, optional
        Number of parallel jobs. Default is 1.

    Attributes
    ----------
    estimate_ : float
        Point estimate from the original data.
    ci_low_, ci_high_ : float
        Lower and upper bounds of the confidence interval.
    bootstrap_distribution_ : np.ndarray
        Bootstrap scores.
    standard_error_ : float
        Standard error of the bootstrap distribution.

    Examples
    --------
    >>> import numpy as np
    >>> from xyz import BootstrapEstimate, GaussianCopulaMutualInformation
    >>> rng = np.random.default_rng(404)
    >>> x = rng.normal(size=(500, 1))
    >>> y = 0.6 * x + 0.3 * rng.normal(size=(500, 1))
    >>> bootstrap = BootstrapEstimate(
    ...     GaussianCopulaMutualInformation(),
    ...     n_bootstrap=24, method="iid", random_state=0, n_jobs=2,
    ... ).fit(x, y)
    >>> bootstrap.ci_low_ <= bootstrap.estimate_ <= bootstrap.ci_high_
    True
    """

    def __init__(
        self,
        estimator,
        *,
        n_bootstrap: int = 100,
        method: str = "iid",
        block_length: int | None = None,
        ci: float = 0.95,
        random_state=None,
        n_jobs: int | None = 1,
    ):
        self.estimator = estimator
        self.n_bootstrap = n_bootstrap
        self.method = method
        self.block_length = block_length
        self.ci = ci
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        self.estimate_ = float(self.estimator_.score())

        rng = _ensure_rng(self.random_state)
        seeds = rng.integers(0, np.iinfo(np.uint32).max, size=int(self.n_bootstrap), dtype=np.uint32)
        bootstrap_scores = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_score_bootstrap_sample)(
                self.estimator,
                X,
                y,
                method=self.method,
                block_length=self.block_length,
                seed=int(seed),
            )
            for seed in seeds
        )
        self.bootstrap_distribution_ = np.asarray(bootstrap_scores, dtype=float)
        alpha = (1.0 - float(self.ci)) / 2.0
        self.ci_low_, self.ci_high_ = np.quantile(
            self.bootstrap_distribution_,
            [alpha, 1.0 - alpha],
        )
        self.standard_error_ = float(np.std(self.bootstrap_distribution_, ddof=1))
        return self

    def score(self, X=None, y=None):
        if X is not None:
            self.fit(X, y)
        return float(self.estimate_)


class SurrogatePermutationTest(MetaEstimatorMixin, BaseEstimator):
    """Permutation-based significance testing for transfer-entropy estimators.

    Fits the estimator on the observed data and on surrogate data (driver
    shuffled/perturbed) to build a null distribution and compute a p-value.

    Parameters
    ----------
    estimator : object
        TE estimator with ``fit`` and ``score`` (e.g. :class:`xyz.KSGTransferEntropy`).
    n_permutations : int, optional
        Number of surrogates. Default is 100.
    surrogate_method : str, optional
        Method passed to :func:`generate_surrogates`. Default is ``"trial_shuffle"``.
    alpha : float, optional
        Significance level. Default is 0.05.
    correction : str or None, optional
        Multiple-test correction: ``"fdr_bh"``, ``"bonferroni"``, or ``"none"``.
        Default is ``"fdr_bh"``.
    shift_test : bool, optional
        If True, also run a time-shift test. Default is False.
    shift_method : str, optional
        Surrogate method for the shift test. Default is ``"time_shift"``.
    random_state : int, array-like or None, optional
        Random seed or generator.
    n_jobs : int or None, optional
        Number of parallel jobs. Default is 1.

    Attributes
    ----------
    observed_score_ : float
        Score on the original data.
    null_distribution_ : np.ndarray
        Scores on surrogates.
    p_values_ : np.ndarray
        P-value(s).
    significant_ : bool or np.ndarray
        True where p_value <= alpha.
    corrected_significant_ : bool or np.ndarray
        After multiple-test correction.

    Examples
    --------
    >>> import numpy as np
    >>> from xyz import SurrogatePermutationTest, KSGTransferEntropy
    >>> rng = np.random.default_rng(42)
    >>> trials = []
    >>> for _ in range(4):
    ...     driver = rng.normal(size=120)
    ...     target = np.zeros(120)
    ...     for t in range(1, 120):
    ...         target[t] = 0.4 * target[t-1] + 0.5 * driver[t-1] + 0.1 * rng.normal()
    ...     trials.append(np.column_stack([target, driver]))
    >>> X = np.stack(trials)
    >>> test = SurrogatePermutationTest(
    ...     KSGTransferEntropy(driver_indices=[1], target_indices=[0], lags=1, k=3),
    ...     n_permutations=12, surrogate_method="trial_shuffle", alpha=0.1, random_state=0,
    ... ).fit(X)
    >>> np.isfinite(test.observed_score_)
    True
    """

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
        n_jobs: int | None = 1,
    ):
        self.estimator = estimator
        self.n_permutations = n_permutations
        self.surrogate_method = surrogate_method
        self.alpha = alpha
        self.correction = correction
        self.shift_test = shift_test
        self.shift_method = shift_method
        self.random_state = random_state
        self.n_jobs = n_jobs

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
        surrogate_scores = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_score_surrogate)(self.estimator, surrogate, y) for surrogate in surrogates
        )
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
