from abc import ABC

import numpy as np

from .base import InfoTheoryEstimator, InfoTheoryMixin
from .utils import buildvectors


class DiscreteInfoTheoryEstimator(InfoTheoryMixin, InfoTheoryEstimator, ABC):
    """Base class for discrete estimators.

    Parameters
    ----------
    alphabet : array-like, optional
        Optional explicit alphabet for discrete states. Current implementations
        infer states from data when this is not provided.
    """

    def __init__(self, alphabet=None):
        self.alphabet = alphabet


def _quantize_matlab(y: np.ndarray, c: int) -> np.ndarray:
    """Quantize a 1D signal using MATLAB-compatible uniform bins.

    The implementation mirrors the ITS toolbox quantization convention:
    integer labels in ``{1, ..., c}``, with saturation at the highest level.
    """
    y = np.asarray(y).reshape(-1)
    n = y.shape[0]
    x = np.zeros(n, dtype=int)
    ma = np.max(y)
    mi = np.min(y)
    if c <= 0:
        raise ValueError("c must be > 0")
    q = (ma - mi) / c
    if q == 0:
        return np.ones(n, dtype=int)
    levels = np.array([mi + (i + 1) * q for i in range(c)])
    for i in range(n):
        j = 0
        while j < c - 1 and y[i] >= levels[j]:
            j += 1
        x[i] = j + 1
    return x


def _entropy_binning(Y: np.ndarray, c: int, quantize: bool = True) -> float:
    """Estimate entropy from empirical frequencies on discretized states.

    Parameters
    ----------
    Y : ndarray
        Samples, shape ``(n_samples, n_features)``.
    c : int
        Number of quantization bins if ``quantize=True``.
    quantize : bool, default=True
        If True, apply MATLAB-like quantization before counting states.
    """
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if quantize:
        Yq = np.column_stack([_quantize_matlab(Y[:, m], c) - 1 for m in range(Y.shape[1])])
    else:
        Yq = Y
    n = Yq.shape[0]
    _, counts = np.unique(Yq, axis=0, return_counts=True)
    p = counts / n
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _conditional_entropy_binning(B: np.ndarray) -> float:
    """Estimate conditional entropy ``H(y|A)`` from an observation matrix.

    The first column of ``B`` is interpreted as the current target ``y`` and
    remaining columns as conditioning variables ``A``.
    """
    B = np.asarray(B)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    y = B[:, :1]
    A = B[:, 1:]
    n = B.shape[0]
    if A.shape[1] == 0:
        return _entropy_binning(y, c=1, quantize=False)
    uniq_A, inv = np.unique(A, axis=0, return_inverse=True)
    ce = 0.0
    for g in range(uniq_A.shape[0]):
        idx = inv == g
        pg = idx.mean()
        y_g = y[idx]
        _, cnt = np.unique(y_g, axis=0, return_counts=True)
        p = cnt / cnt.sum()
        h = -(p[p > 0] * np.log(p[p > 0])).sum()
        ce += pg * h
    return float(ce)


class DiscreteTransferEntropy(InfoTheoryEstimator):
    """Discrete bivariate transfer entropy estimator.

    This estimator implements:

    ``TE(X->Y) = H(Y_n | Y_n^-) - H(Y_n | Y_n^-, X_n^-)``

    where past vectors are built with uniform lags using ``buildvectors``.
    The implementation is aligned with the ITS binning workflow.
    """

    def __init__(self, driver_indices, target_indices, lags=1, c=8, quantize=True):
        self.driver_indices = driver_indices
        self.target_indices = target_indices
        self.lags = lags
        self.c = c
        self.quantize = quantize

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.quantize:
            Xq = np.column_stack([_quantize_matlab(X[:, m], self.c) - 1 for m in range(X.shape[1])])
        else:
            Xq = X
        V = np.array(
            [[self.target_indices[0], l] for l in range(1, self.lags + 1)]
            + [[self.driver_indices[0], l] for l in range(1, self.lags + 1)]
        )
        B = buildvectors(Xq, self.target_indices[0], V)
        n_t = self.lags
        y_present = B[:, :1]
        y_past = B[:, 1 : 1 + n_t]
        xy_past = B[:, 1:]
        hy_y = _conditional_entropy_binning(np.hstack([y_present, y_past])) if y_past.size else _entropy_binning(y_present, self.c, False)
        hy_xy = _conditional_entropy_binning(np.hstack([y_present, xy_past])) if xy_past.size else hy_y
        self.transfer_entropy_ = float(hy_y - hy_xy)
        self.hy_y_ = float(hy_y)
        self.hy_xy_ = float(hy_xy)
        return self


class DiscretePartialTransferEntropy(InfoTheoryEstimator):
    """Discrete partial transfer entropy estimator.

    Implements conditional transfer entropy:

    ``PTE(X->Y|Z) = H(Y_n | Y_n^-, Z_n^-) - H(Y_n | Y_n^-, X_n^-, Z_n^-)``.
    """

    def __init__(self, driver_indices, target_indices, conditioning_indices, lags=1, c=8, quantize=True):
        self.driver_indices = driver_indices
        self.target_indices = target_indices
        self.conditioning_indices = conditioning_indices
        self.lags = lags
        self.c = c
        self.quantize = quantize

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.quantize:
            Xq = np.column_stack([_quantize_matlab(X[:, m], self.c) - 1 for m in range(X.shape[1])])
        else:
            Xq = X
        V = np.array(
            [[self.target_indices[0], l] for l in range(1, self.lags + 1)]
            + [[self.driver_indices[0], l] for l in range(1, self.lags + 1)]
            + [[self.conditioning_indices[0], l] for l in range(1, self.lags + 1)]
        )
        B = buildvectors(Xq, self.target_indices[0], V)
        y_present = B[:, :1]
        A = B[:, 1:]
        vars_all = V[:, 0]
        ii = self.driver_indices[0]
        yz = A[:, vars_all != ii]
        hy_xyz = _conditional_entropy_binning(np.hstack([y_present, A]))
        hy_yz = _conditional_entropy_binning(np.hstack([y_present, yz])) if yz.size else _entropy_binning(y_present, self.c, False)
        self.transfer_entropy_ = float(hy_yz - hy_xyz)
        self.hy_yz_ = float(hy_yz)
        self.hy_xyz_ = float(hy_xyz)
        return self


class DiscreteSelfEntropy(InfoTheoryEstimator):
    """Discrete self-entropy (information storage) estimator.

    Implements:

    ``SE(Y) = I(Y_n ; Y_n^-) = H(Y_n) - H(Y_n | Y_n^-)``.
    """

    def __init__(self, target_indices, lags=1, c=8, quantize=True):
        self.target_indices = target_indices
        self.lags = lags
        self.c = c
        self.quantize = quantize

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        x = X[:, self.target_indices]
        if self.quantize:
            xq = np.column_stack([_quantize_matlab(x[:, m], self.c) - 1 for m in range(x.shape[1])])
        else:
            xq = x
        hy = _entropy_binning(xq[:, :1], self.c, quantize=False)
        V = np.array([[0, l] for l in range(1, self.lags + 1)])
        B = buildvectors(xq, 0, V)
        hy_y = _conditional_entropy_binning(B)
        self.self_entropy_ = float(hy - hy_y)
        self.hy_ = float(hy)
        self.hy_y_ = float(hy_y)
        return self
