import numpy as np
from scipy.spatial.distance import pdist


def entropy_linear(A: np.ndarray) -> float:
    """Linear Gaussian estimate of differential (Shannon) entropy.

    Assumes the data are multivariate Gaussian. For covariance :math:`C`,
    the differential entropy in nats is:

    .. math::

        H = \\frac{1}{2} \\log \\det(C) + \\frac{M}{2} \\log(2\\pi e)

    where :math:`M` is the number of variables.

    Parameters
    ----------
    A : np.ndarray
        Multivariate data, shape ``(n_samples, n_features)`` (N×M).

    Returns
    -------
    float
        Estimated differential entropy in nats.

    Examples
    --------
    >>> import numpy as np
    >>> from xyz.univariate import entropy_linear
    >>> rng = np.random.default_rng(42)
    >>> A = rng.normal(size=(500, 3))
    >>> h = entropy_linear(A)
    >>> np.isfinite(h)
    True
    """
    C = np.cov(A.T)

    # Entropy for the multivariate Gaussian case:
    N, M = A.shape
    # e.g., Barnett PRL 2009
    e = 0.5 * np.log(np.linalg.det(C)) + 0.5 * M * np.log(2 * np.pi * np.exp(1))
    return e


def entropy_kernel(Y: np.ndarray, r: float, metric: str = "chebyshev") -> float:
    """Kernel (step-kernel) estimate of differential entropy.

    Uses the mean log-probability of pairs within radius :math:`r` under the
    chosen distance. By default uses a step kernel with Chebyshev (max-norm)
    distance.

    Parameters
    ----------
    Y : np.ndarray
        Data, shape ``(n_samples, n_features)``.
    r : float
        Radius for the step kernel.
    metric : str, optional
        Distance metric for pairwise distances (e.g. ``"chebyshev"`` or
        ``"euclidean"``). Default is ``"chebyshev"``.

    Returns
    -------
    float
        Estimated differential entropy in nats.

    Examples
    --------
    >>> import numpy as np
    >>> from xyz.univariate import entropy_kernel
    >>> rng = np.random.default_rng(42)
    >>> Y = rng.normal(size=(500, 2))
    >>> h = entropy_kernel(Y, 0.1)
    >>> np.isfinite(h)
    True
    """
    return -np.log((pdist(Y, metric=metric) < r).mean())


# for details on Kraskov estimator
# https://lizliz.github.io/teaspoon/_modules/teaspoon/parameter_selection/MI_delay.html
# also read this for the case of mutual information and why the KSG estimator may return negative results
# https://github.com/paulbrodersen/entropy_estimators/issues/11#issuecomment-2109577671
# with the paper here
# https://arxiv.org/abs/2405.04980
# and a python implementation here too
# https://github.com/moldyn/NorMI


def entropy_binning(Y, c, quantize, log_base: str = "nat"):
    """Binning (histogram) estimate of Shannon entropy.

    Discretizes each column into ``c`` bins and computes entropy from the
    empirical distribution. If ``quantize`` is False, data are binned by
    equal-width bins; otherwise ``Y`` is assumed already quantized.

    Parameters
    ----------
    Y : array-like
        Data matrix, shape ``(n_samples, n_features)``.
    c : int
        Number of bins per dimension.
    quantize : bool
        If True, treat ``Y`` as already quantized (values in ``0..c-1``).
        If False, bin continuous values with :func:`xyz.utils.quantize`.
    log_base : str, optional
        Logarithm base; currently only ``"nat"`` is supported.

    Returns
    -------
    float
        Estimated entropy (implementation may return from internal state).

    Examples
    --------
    Used internally by discrete estimators; for continuous entropy prefer
    :class:`xyz.KSGEntropy` or :class:`xyz.MVNEntropy`.
    """
    # TODO write version with entropy in bits

    from .utils import quantize

    if quantize:
        Yq = Y
    else:
        Yq = Y
        for j in range(Y.shape[1]):
            Yq[:, j] = quantize(Y[:, j], c) - 1
    Q: np.ndarray = Yq
