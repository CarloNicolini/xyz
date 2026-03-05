from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.spatial import cKDTree


def as_2d_array(X) -> np.ndarray:
    """Return an array with shape ``(n_samples, n_features)``."""
    X = np.asarray(X)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    if X.ndim == 2:
        return X
    raise ValueError(f"Expected a 1D or 2D array, got shape {X.shape}")


def as_trial_array(X) -> np.ndarray:
    """Return an array with shape ``(n_trials, n_samples, n_features)``."""
    X = np.asarray(X)
    if X.ndim == 1:
        return X.reshape(1, -1, 1)
    if X.ndim == 2:
        return X[np.newaxis, ...]
    if X.ndim == 3:
        return X
    raise ValueError(f"Expected a 1D, 2D or 3D array, got shape {X.shape}")


def iter_trials(X):
    """Yield per-trial arrays of shape ``(n_samples, n_features)``."""
    for trial in as_trial_array(X):
        yield np.asarray(trial)


def estimate_autocorrelation_decay(x, max_lag: int = 1000, threshold: float | None = None) -> int:
    """Estimate the autocorrelation decay time in samples.

    The estimate is the first positive lag whose normalized autocorrelation
    drops below ``threshold``. By default, the threshold is ``exp(-1)``, a
    simple proxy for the decay time used in ACT-based workflows.
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
    """Estimate trial-wise ACT values for a target variable."""
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
    """Select trials whose ACT does not exceed a threshold.

    Returns
    -------
    selected_trials : ndarray
        Subset of the original trials.
    acts : ndarray
        Estimated ACT values for all trials before filtering.
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


def build_te_observations(
    X,
    *,
    target_index: int,
    lags: int,
    tau: int = 1,
    delay: int = 1,
    driver_index: int | None = None,
    conditioning_indices: Iterable[int] | None = None,
    extra_conditioning: str | None = None,
) -> dict[str, np.ndarray]:
    """Build TE/PTE/SE state-space matrices from one or more trials."""
    if lags < 1:
        raise ValueError("lags must be >= 1")
    if tau < 1:
        raise ValueError("tau must be >= 1")
    if delay < 1:
        raise ValueError("delay must be >= 1")

    conditioning_indices = _normalize_conditioning_indices(conditioning_indices)
    trials = as_trial_array(X)

    y_present_all = []
    y_past_all = []
    x_past_all = []
    z_past_all = []
    faes_all = []
    trial_ids = []

    target_offsets = [1 + i * tau for i in range(lags)]
    driver_offsets = [delay + i * tau for i in range(lags)] if driver_index is not None else []
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

        if driver_index is None:
            x_past = np.empty((rows.size, 0))
        else:
            x_past = np.column_stack(
                [trial[rows - offset, driver_index] for offset in driver_offsets]
            )

        if conditioning_indices:
            z_blocks = []
            for conditioning_index in conditioning_indices:
                z_blocks.extend(
                    trial[rows - offset, conditioning_index] for offset in cond_offsets
                )
            z_past = np.column_stack(z_blocks)
        else:
            z_past = np.empty((rows.size, 0))

        if use_faes and driver_index is not None:
            faes = trial[rows, driver_index : driver_index + 1]
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
    """Estimate local prediction error for a 1D time series.

    This is a lightweight Python approximation of the Ragwitz criterion: embed
    the target time series with dimension ``dim`` and spacing ``tau``, predict
    the future value from nearby states, and return the mean squared error.
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
