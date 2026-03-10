from __future__ import annotations

from abc import ABC

import numpy as np

from .base import InfoTheoryEstimator, InfoTheoryMixin
from .preprocessing import build_te_observations


class DiscreteInfoTheoryEstimator(InfoTheoryMixin, InfoTheoryEstimator, ABC):
    """Base class for discrete (binned) information-theoretic estimators.

    Subclasses operate on discretized time series and use histogram-based
    entropy and conditional entropy to compute transfer entropy, partial
    transfer entropy, or self-entropy.
    """


def _quantize_matlab(y: np.ndarray, c: int) -> np.ndarray:
    """Quantize a 1D signal using MATLAB-compatible uniform bins."""
    y = np.asarray(y).reshape(-1)
    if c <= 0:
        raise ValueError("c must be > 0")
    ma = np.max(y)
    mi = np.min(y)
    q = (ma - mi) / c
    if q == 0:
        return np.ones_like(y, dtype=int)
    levels = np.array([mi + (i + 1) * q for i in range(c)])
    x = np.zeros_like(y, dtype=int)
    for i, value in enumerate(y):
        j = 0
        while j < c - 1 and value >= levels[j]:
            j += 1
        x[i] = j + 1
    return x


def _prepare_discrete_trials(X, *, c: int, quantize: bool) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1, 1)
    elif X.ndim == 2:
        X = X[np.newaxis, ...]
    elif X.ndim != 3:
        raise ValueError(f"Expected a 1D, 2D or 3D array, got shape {X.shape}")

    if not quantize:
        return X.astype(int, copy=False)

    Xq = np.empty_like(X, dtype=int)
    for trial_idx in range(X.shape[0]):
        for feature_idx in range(X.shape[2]):
            Xq[trial_idx, :, feature_idx] = _quantize_matlab(X[trial_idx, :, feature_idx], c) - 1
    return Xq


def _entropy_binning(Y: np.ndarray) -> float:
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    _, counts = np.unique(Y, axis=0, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _conditional_entropy_binning(Y: np.ndarray, X: np.ndarray) -> float:
    Y = np.asarray(Y)
    X = np.asarray(X)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[1] == 0:
        return _entropy_binning(Y)

    _, inv = np.unique(X, axis=0, return_inverse=True)
    ce = 0.0
    for group_id in range(inv.max() + 1):
        idx = inv == group_id
        p_group = idx.mean()
        ce += p_group * _entropy_binning(Y[idx])
    return float(ce)


class DiscreteTransferEntropy(DiscreteInfoTheoryEstimator):
    """Discrete (binned) bivariate transfer entropy.

    Estimates :math:`TE_{X \\to Y} = H(Y_t | Y_{t-1:t-l}) - H(Y_t | Y_{t-1:t-l}, X_{t-d:t-d-l})`
    using histogram-based entropy after binning into ``c`` bins.

    Parameters
    ----------
    driver_indices : array-like
        Column index(es) of the driver variable(s).
    target_indices : array-like
        Column index(es) of the target variable(s).
    lags : int, optional
        Number of past lags. Default is 1.
    tau : int, optional
        Lag step (samples). Default is 1.
    delay : int, optional
        Delay from driver to target. Default is 1.
    c : int, optional
        Number of bins for discretization. Default is 8.
    quantize : bool, optional
        If True, bin continuous data; if False, assume input is already discrete. Default is True.
    extra_conditioning : str or None, optional
        Optional extra conditioning (e.g. Faes method). Default is None.

    Attributes
    ----------
    transfer_entropy_ : float
        Fitted transfer entropy estimate (after :meth:`fit`).

    Examples
    --------
    >>> import numpy as np
    >>> from xyz import DiscreteTransferEntropy
    >>> rng = np.random.default_rng(42)
    >>> X = np.column_stack([rng.rand(200), rng.rand(200)])  # (n_samples, 2): target, driver
    >>> est = DiscreteTransferEntropy(driver_indices=[1], target_indices=[0], lags=1, c=6, quantize=True)
    >>> est.fit(X)
    >>> np.isfinite(est.transfer_entropy_)
    True
    """

    score_attr_ = "transfer_entropy_"

    def __init__(
        self,
        driver_indices,
        target_indices,
        lags: int = 1,
        tau: int = 1,
        delay: int = 1,
        c: int = 8,
        quantize: bool = True,
        extra_conditioning: str | None = None,
    ):
        self.driver_indices = driver_indices
        self.target_indices = target_indices
        self.lags = lags
        self.tau = tau
        self.delay = delay
        self.c = c
        self.quantize = quantize
        self.extra_conditioning = extra_conditioning

    def fit(self, X, y=None):
        Xq = _prepare_discrete_trials(X, c=self.c, quantize=self.quantize)
        parts = build_te_observations(
            Xq,
            target_index=self.target_indices[0],
            lags=self.lags,
            tau=self.tau,
            delay=self.delay,
            driver_index=self.driver_indices[0],
            extra_conditioning=self.extra_conditioning,
        )
        restricted = np.hstack([parts["y_past"], parts["faes_current"]])
        full = np.hstack([parts["y_past"], parts["x_past"], parts["faes_current"]])
        self.hy_y_ = _conditional_entropy_binning(parts["y_present"], restricted)
        self.hy_xy_ = _conditional_entropy_binning(parts["y_present"], full)
        self.transfer_entropy_ = float(self.hy_y_ - self.hy_xy_)
        return self


class DiscretePartialTransferEntropy(DiscreteInfoTheoryEstimator):
    """Discrete partial transfer entropy (conditioning on side variables).

    Estimates PTE from driver to target conditioned on ``conditioning_indices``
    using binned entropy, i.e. the information transfer excluding that explained
    by the conditioning variable(s).

    Parameters
    ----------
    driver_indices : array-like
        Column index(es) of the driver.
    target_indices : array-like
        Column index(es) of the target.
    conditioning_indices : array-like
        Column index(es) to condition on.
    lags : int, optional
        Number of past lags. Default is 1.
    tau : int, optional
        Lag step. Default is 1.
    delay : int, optional
        Delay from driver to target. Default is 1.
    c : int, optional
        Number of bins. Default is 8.
    quantize : bool, optional
        If True, bin continuous data. Default is True.
    extra_conditioning : str or None, optional
        Optional extra conditioning. Default is None.

    Attributes
    ----------
    transfer_entropy_ : float
        Fitted partial transfer entropy (after :meth:`fit`).

    Examples
    --------
    >>> import numpy as np
    >>> from xyz import DiscretePartialTransferEntropy
    >>> rng = np.random.default_rng(42)
    >>> X = np.column_stack([rng.rand(200), rng.rand(200), rng.rand(200)])  # target, driver, conditioning
    >>> est = DiscretePartialTransferEntropy(
    ...     driver_indices=[1], target_indices=[0], conditioning_indices=[2],
    ...     lags=1, c=6, quantize=True,
    ... )
    >>> est.fit(X)
    >>> np.isfinite(est.transfer_entropy_)
    True
    """

    score_attr_ = "transfer_entropy_"

    def __init__(
        self,
        driver_indices,
        target_indices,
        conditioning_indices,
        lags: int = 1,
        tau: int = 1,
        delay: int = 1,
        c: int = 8,
        quantize: bool = True,
        extra_conditioning: str | None = None,
    ):
        self.driver_indices = driver_indices
        self.target_indices = target_indices
        self.conditioning_indices = conditioning_indices
        self.lags = lags
        self.tau = tau
        self.delay = delay
        self.c = c
        self.quantize = quantize
        self.extra_conditioning = extra_conditioning

    def fit(self, X, y=None):
        Xq = _prepare_discrete_trials(X, c=self.c, quantize=self.quantize)
        parts = build_te_observations(
            Xq,
            target_index=self.target_indices[0],
            lags=self.lags,
            tau=self.tau,
            delay=self.delay,
            driver_index=self.driver_indices[0],
            conditioning_indices=self.conditioning_indices,
            extra_conditioning=self.extra_conditioning,
        )
        restricted = np.hstack([parts["y_past"], parts["z_past"], parts["faes_current"]])
        full = np.hstack([parts["y_past"], parts["x_past"], parts["z_past"], parts["faes_current"]])
        self.hy_yz_ = _conditional_entropy_binning(parts["y_present"], restricted)
        self.hy_xyz_ = _conditional_entropy_binning(parts["y_present"], full)
        self.transfer_entropy_ = float(self.hy_yz_ - self.hy_xyz_)
        return self


class DiscreteSelfEntropy(DiscreteInfoTheoryEstimator):
    """Discrete information storage (self-entropy).

    Estimates :math:`S_Y = H(Y_t) - H(Y_t | Y_{t-1:t-l})`, the information
    in the target's past about its present, using binned entropy.

    Parameters
    ----------
    target_indices : array-like
        Column index(es) of the target variable.
    lags : int, optional
        Number of past lags. Default is 1.
    tau : int, optional
        Lag step. Default is 1.
    c : int, optional
        Number of bins. Default is 8.
    quantize : bool, optional
        If True, bin continuous data. Default is True.

    Attributes
    ----------
    self_entropy_ : float
        Fitted self-entropy (after :meth:`fit`).

    Examples
    --------
    >>> import numpy as np
    >>> from xyz import DiscreteSelfEntropy
    >>> rng = np.random.default_rng(42)
    >>> X = rng.rand(200, 1)  # single series
    >>> est = DiscreteSelfEntropy(target_indices=[0], lags=2, c=6, quantize=True)
    >>> est.fit(X)
    >>> np.isfinite(est.self_entropy_)
    True
    """

    score_attr_ = "self_entropy_"

    def __init__(self, target_indices, lags: int = 1, tau: int = 1, c: int = 8, quantize: bool = True):
        self.target_indices = target_indices
        self.lags = lags
        self.tau = tau
        self.c = c
        self.quantize = quantize

    def fit(self, X, y=None):
        Xq = _prepare_discrete_trials(X, c=self.c, quantize=self.quantize)
        parts = build_te_observations(
            Xq,
            target_index=self.target_indices[0],
            lags=self.lags,
            tau=self.tau,
        )
        full_target = Xq[:, :, self.target_indices[0]].reshape(-1, 1)
        self.hy_ = _entropy_binning(full_target)
        self.hy_y_ = _conditional_entropy_binning(parts["y_present"], parts["y_past"])
        self.self_entropy_ = float(self.hy_ - self.hy_y_)
        return self


__all__ = [
    "DiscreteInfoTheoryEstimator",
    "DiscretePartialTransferEntropy",
    "DiscreteSelfEntropy",
    "DiscreteTransferEntropy",
]
