from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.spatial import cKDTree


def as_2d_array(X) -> np.ndarray:
    """Coerce input to a 2D array with shape (n_samples, n_features).

    Parameters
    ----------
    X : array-like
        1D or 2D array.

    Returns
    -------
    np.ndarray
        Shape ``(n_samples, n_features)``; 1D input becomes ``(n, 1)``.

    Raises
    ------
    ValueError
        If ``X.ndim`` is not 1 or 2.

    Examples
    --------
    >>> import numpy as np
    >>> from xyz.preprocessing import as_2d_array
    >>> as_2d_array(np.array([1, 2, 3])).shape
    (3, 1)
    >>> as_2d_array(np.random.randn(10, 2)).shape
    (10, 2)
    """
    X = np.asarray(X)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    if X.ndim == 2:
        return X
    raise ValueError(f"Expected a 1D or 2D array, got shape {X.shape}")


def as_trial_array(X) -> np.ndarray:
    """Coerce input to trial format (n_trials, n_samples, n_features).

    Parameters
    ----------
    X : array-like
        1D, 2D, or 3D array. 1D → (1, n, 1); 2D → (1, n_samples, n_features).

    Returns
    -------
    np.ndarray
        Shape ``(n_trials, n_samples, n_features)``.

    Raises
    ------
    ValueError
        If ``X.ndim`` is not 1, 2, or 3.

    Examples
    --------
    >>> import numpy as np
    >>> from xyz.preprocessing import as_trial_array
    >>> X = np.random.randn(50, 2)
    >>> T = as_trial_array(X)
    >>> T.shape
    (1, 50, 2)
    """
    X = np.asarray(X)
    if X.ndim == 1:
        return X.reshape(1, -1, 1)
    if X.ndim == 2:
        return X[np.newaxis, ...]
    if X.ndim == 3:
        return X
    raise ValueError(f"Expected a 1D, 2D or 3D array, got shape {X.shape}")


def iter_trials(X):
    """Iterate over trials, yielding arrays of shape (n_samples, n_features).

    Parameters
    ----------
    X : array-like
        Data in any format accepted by :func:`as_trial_array`.

    Yields
    ------
    np.ndarray
        One array per trial.

    Examples
    --------
    >>> import numpy as np
    >>> from xyz.preprocessing import iter_trials
    >>> X = np.random.randn(2, 30, 2)  # 2 trials
    >>> for trial in iter_trials(X):
    ...     print(trial.shape)
    (30, 2)
    (30, 2)
    """
    for trial in as_trial_array(X):
        yield np.asarray(trial)


def estimate_autocorrelation_decay(x, max_lag: int = 1000, threshold: float | None = None) -> int:
    """Estimate autocorrelation decay time (ACT) in samples.

    Returns the first positive lag :math:`\\tau` at which the normalized
    autocorrelation :math:`r(\\tau) \\le` threshold. Default threshold is
    :math:`e^{-1}`, a common proxy for the decay time.

    Parameters
    ----------
    x : array-like
        1D time series.
    max_lag : int, optional
        Maximum lag to consider. Default is 1000.
    threshold : float or None, optional
        Stop when autocorrelation falls below this. Default is :math:`e^{-1}`.

    Returns
    -------
    int
        Estimated ACT (samples).

    Examples
    --------
    >>> import numpy as np
    >>> from xyz.preprocessing import estimate_autocorrelation_decay
    >>> x = np.cumsum(np.random.randn(500))
    >>> act = estimate_autocorrelation_decay(x, max_lag=100)
    >>> 1 <= act <= 101
    True
    """
    x = np.asarray(x).reshape(-1)
    if x.size < 3:
        return 1

    x = x - x.mean()
    denom = np.dot(x, x)
    if denom <= 0:
        return 1

    threshold = float(np.exp(-1)) if threshold is None else float(threshold)
    max_lag = max(1, min(int(max_lag), x.size - 1))

    for lag in range(1, max_lag + 1):
        acf = np.dot(x[:-lag], x[lag:]) / denom
        if acf <= threshold:
            return lag
    return max_lag


def estimate_trial_acts(X, target_index: int, max_lag: int = 1000) -> np.ndarray:
    """Estimate autocorrelation decay time per trial for one target column.

    Parameters
    ----------
    X : array-like
        Data in trial format (see :func:`as_trial_array`).
    target_index : int
        Column index of the target variable.
    max_lag : int, optional
        Maximum lag for :func:`estimate_autocorrelation_decay`. Default is 1000.

    Returns
    -------
    np.ndarray
        One ACT value per trial, shape ``(n_trials,)``.

    Examples
    --------
    >>> import numpy as np
    >>> from xyz.preprocessing import as_trial_array, estimate_trial_acts
    >>> X = np.random.randn(3, 200, 2)
    >>> acts = estimate_trial_acts(X, target_index=0, max_lag=50)
    >>> acts.shape
    (3,)
    """
    return np.asarray(
        [
            estimate_autocorrelation_decay(trial[:, target_index], max_lag=max_lag)
            for trial in iter_trials(X)
        ],
        dtype=int,
    )


def select_trials_by_act(
    X,
    target_index: int,
    *,
    max_lag: int = 1000,
    act_threshold: int | None = None,
    min_trials: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Select trials whose autocorrelation decay time is below a threshold.

    Parameters
    ----------
    X : array-like
        Data in trial format.
    target_index : int
        Column index of the target for ACT estimation.
    max_lag : int, optional
        Max lag for ACT. Default is 1000.
    act_threshold : int or None, optional
        Keep only trials with ACT <= this. If None, no filtering (all trials returned).
    min_trials : int, optional
        Minimum number of trials that must remain after filtering. Default is 1.

    Returns
    -------
    selected_trials : np.ndarray
        Subset of trials with ACT <= act_threshold (or all if act_threshold is None).
    acts : np.ndarray
        ACT value for each original trial.

    Raises
    ------
    ValueError
        If filtering would leave fewer than min_trials.

    Examples
    --------
    >>> import numpy as np
    >>> from xyz.preprocessing import as_trial_array, select_trials_by_act
    >>> X = np.random.randn(4, 300, 2)
    >>> trials, acts = select_trials_by_act(X, 0, act_threshold=50, min_trials=2)
    >>> trials.shape[0] <= 4 and len(acts) == 4
    True
    """
    trials = as_trial_array(X)
    acts = estimate_trial_acts(trials, target_index=target_index, max_lag=max_lag)
    if act_threshold is None:
        return trials, acts

    mask = acts <= int(act_threshold)
    if np.sum(mask) < int(min_trials):
        raise ValueError(
            "ACT-based trial selection removed too many trials; "
            f"kept {np.sum(mask)} but require at least {min_trials}"
        )
    return trials[mask], acts


def _normalize_conditioning_indices(conditioning_indices: Iterable[int] | None) -> list[int]:
    if conditioning_indices is None:
        return []
    return [int(index) for index in conditioning_indices]


def _normalize_driver_indices(
    driver_indices: Iterable[int] | None = None,
    driver_index: int | None = None,
) -> list[int]:
    if driver_indices is not None:
        return [int(index) for index in driver_indices]
    if driver_index is None:
        return []
    return [int(driver_index)]


def build_te_observations(
    X,
    *,
    target_index: int,
    lags: int,
    tau: int = 1,
    delay: int = 1,
    driver_index: int | None = None,
    driver_indices: Iterable[int] | None = None,
    conditioning_indices: Iterable[int] | None = None,
    extra_conditioning: str | None = None,
) -> dict[str, np.ndarray]:
    """Build transfer-entropy state-space matrices from trial data.

    Constructs present/past blocks for target, driver(s), and optional
    conditioning variables across trials. Used internally by TE estimators.

    Parameters
    ----------
    X : array-like
        Data in trial format (n_trials, n_samples, n_features).
    target_index : int
        Column index of the target.
    lags : int
        Number of past lags (embedding dimension).
    tau : int, optional
        Lag step (samples). Default is 1.
    delay : int, optional
        Delay from driver to target (samples). Default is 1.
    driver_index, driver_indices : int or iterable of int, optional
        Driver column index(s).
    conditioning_indices : iterable of int or None, optional
        Column indices for conditioning (e.g. for PTE).
    extra_conditioning : str or None, optional
        If ``"Faes_Method"`` or ``"faes"``, include current driver in conditioning.

    Returns
    -------
    dict
        Keys: ``"y_present"``, ``"y_past"``, ``"x_past"``, ``"z_past"``,
        ``"faes_current"``, ``"trial_ids"``. Values are concatenated over trials.

    Raises
    ------
    ValueError
        If lags, tau, or delay < 1, or no valid samples remain.

    Examples
    --------
    >>> import numpy as np
    >>> from xyz.preprocessing import build_te_observations, as_trial_array
    >>> X = np.random.randn(1, 200, 2)
    >>> out = build_te_observations(X, target_index=0, lags=2, driver_index=1)
    >>> out["y_present"].shape[0] == out["x_past"].shape[0]
    True
    """
    if lags < 1:
        raise ValueError("lags must be >= 1")
    if tau < 1:
        raise ValueError("tau must be >= 1")
    if delay < 1:
        raise ValueError("delay must be >= 1")

    conditioning_indices = _normalize_conditioning_indices(conditioning_indices)
    driver_indices = _normalize_driver_indices(driver_indices, driver_index)
    trials = as_trial_array(X)

    y_present_all = []
    y_past_all = []
    x_past_all = []
    z_past_all = []
    faes_all = []
    trial_ids = []

    target_offsets = [1 + i * tau for i in range(lags)]
    driver_offsets = [delay + i * tau for i in range(lags)] if driver_indices else []
    cond_offsets = [1 + i * tau for i in range(lags)]
    max_offset = max(target_offsets + driver_offsets + cond_offsets + [0])

    use_faes = extra_conditioning in {"Faes_Method", "faes"}

    for trial_id, trial in enumerate(trials):
        trial = as_2d_array(trial)
        n_samples = trial.shape[0]
        if max_offset >= n_samples:
            continue

        rows = np.arange(max_offset, n_samples)
        y_present = trial[rows, target_index : target_index + 1]
        y_past = np.column_stack(
            [trial[rows - offset, target_index] for offset in target_offsets]
        )

        if not driver_indices:
            x_past = np.empty((rows.size, 0))
        else:
            x_blocks = []
            for driver_idx in driver_indices:
                x_blocks.extend(trial[rows - offset, driver_idx] for offset in driver_offsets)
            x_past = np.column_stack(x_blocks)

        if conditioning_indices:
            z_blocks = []
            for conditioning_index in conditioning_indices:
                z_blocks.extend(
                    trial[rows - offset, conditioning_index] for offset in cond_offsets
                )
            z_past = np.column_stack(z_blocks)
        else:
            z_past = np.empty((rows.size, 0))

        if use_faes and driver_indices:
            faes = np.column_stack([trial[rows, driver_idx] for driver_idx in driver_indices])
        else:
            faes = np.empty((rows.size, 0))

        y_present_all.append(y_present)
        y_past_all.append(y_past)
        x_past_all.append(x_past)
        z_past_all.append(z_past)
        faes_all.append(faes)
        trial_ids.append(np.full(rows.size, trial_id, dtype=int))

    if not y_present_all:
        raise ValueError("No valid samples remain after applying lag and delay settings")

    y_present = np.vstack(y_present_all)
    y_past = np.vstack(y_past_all)
    x_past = np.vstack(x_past_all)
    z_past = np.vstack(z_past_all)
    faes = np.vstack(faes_all)
    trial_ids = np.concatenate(trial_ids)

    return {
        "y_present": y_present,
        "y_past": y_past,
        "x_past": x_past,
        "z_past": z_past,
        "faes_current": faes,
        "trial_ids": trial_ids,
    }


def ragwitz_prediction_error(
    x,
    *,
    dim: int,
    tau: int,
    k_neighbors: int = 4,
    theiler_t: int = 0,
    prediction_horizon: int = 1,
    metric: str = "chebyshev",
) -> float:
    """Ragwitz criterion: local prediction error for embedding (dim, tau).

    Embeds the 1D series with dimension ``dim`` and spacing ``tau``, finds
    :math:`k` nearest neighbors in embedding space, and returns the mean
    squared error of predicting the future value from neighbors (Theiler
    window and metric as specified).

    Parameters
    ----------
    x : array-like
        1D time series.
    dim : int
        Embedding dimension.
    tau : int
        Embedding delay (samples).
    k_neighbors : int, optional
        Number of neighbors for local prediction. Default is 4.
    theiler_t : int, optional
        Theiler window (exclude neighbors within this time index). Default is 0.
    prediction_horizon : int, optional
        Steps ahead to predict. Default is 1.
    metric : str, optional
        Distance metric (e.g. ``"chebyshev"``, ``"euclidean"``). Default is ``"chebyshev"``.

    Returns
    -------
    float
        Mean squared prediction error.

    Raises
    ------
    ValueError
        If dim, tau, or prediction_horizon < 1, or series too short.

    Examples
    --------
    >>> import numpy as np
    >>> from xyz.preprocessing import ragwitz_prediction_error
    >>> rng = np.random.default_rng(7)
    >>> x = np.cumsum(rng.normal(size=300))
    >>> err = ragwitz_prediction_error(x, dim=2, tau=1, k_neighbors=4)
    >>> err >= 0
    True
    """
    if dim < 1:
        raise ValueError("dim must be >= 1")
    if tau < 1:
        raise ValueError("tau must be >= 1")
    if prediction_horizon < 1:
        raise ValueError("prediction_horizon must be >= 1")

    x = np.asarray(x).reshape(-1)
    max_offset = 1 + (dim - 1) * tau
    last_future_index = x.size - prediction_horizon
    if max_offset >= last_future_index:
        raise ValueError("Time series is too short for the requested embedding")

    rows = np.arange(max_offset, last_future_index)
    states = np.column_stack(
        [x[rows - (1 + i * tau)] for i in range(dim)]
    )
    futures = x[rows + prediction_horizon]

    tree = cKDTree(states)
    p = np.inf if metric == "chebyshev" else 2
    distances, indices = tree.query(states, k=min(k_neighbors + 1, states.shape[0]), p=p)

    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    preds = np.empty_like(futures)
    for i in range(states.shape[0]):
        neighbor_ids = []
        for candidate in indices[i]:
            if candidate == i:
                continue
            if abs(rows[candidate] - rows[i]) <= theiler_t:
                continue
            neighbor_ids.append(candidate)
            if len(neighbor_ids) == k_neighbors:
                break
        if not neighbor_ids:
            preds[i] = np.mean(futures)
        else:
            preds[i] = np.mean(futures[neighbor_ids])

    residual = futures - preds
    return float(np.mean(residual**2))
