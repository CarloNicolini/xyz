import numpy as np


def quantize(y, c):
    """Bin continuous values into ``c`` equal-width bins.

    Uses the range of ``y`` to define bin edges; returns 1-based bin indices
    (as from :func:`numpy.digitize`).

    Parameters
    ----------
    y : array-like
        1D array of continuous values.
    c : int
        Number of bins.

    Returns
    -------
    np.ndarray
        Integer bin index for each element (1 to c).

    Examples
    --------
    >>> import numpy as np
    >>> from xyz.utils import quantize
    >>> y = np.array([0.1, 0.5, 1.0, 1.5, 2.0])
    >>> quantize(y, 3)
    array([1, 1, 2, 2, 3])
    """
    M, m = np.max(y), np.min(y)
    bins = np.arange(m, M, (M - m) / c)
    return np.digitize(y, bins)


def cov(X):
    """Compute the sample covariance matrix, always as a 2D array.

    Parameters
    ----------
    X : array-like
        Data, shape ``(n_samples,)`` or ``(n_samples, n_features)``.
        For 1D input, treated as a single feature.

    Returns
    -------
    np.ndarray
        Covariance matrix, shape ``(n_features, n_features)`` (at least 2D).

    Examples
    --------
    >>> import numpy as np
    >>> from xyz.utils import cov
    >>> X = np.random.randn(100, 2)
    >>> C = cov(X)
    >>> C.shape
    (2, 2)
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    # Return 2D covariance matrix. For 1D input reshaped to 2D, np.cov returns 0D array.
    # To be consistent, ensure we return 2D.
    return np.atleast_2d(np.cov(X, rowvar=False))


def buildvectors(Y, j, V=None):
    """Build observation matrix for entropy computation from lagged variables.

    First column is the current target :math:`Y_{\\cdot,j}`; subsequent columns
    are lagged variables specified by ``V`` (variable index, lag).

    Parameters
    ----------
    Y : np.ndarray
        Multivariate time series, shape ``(n_samples, n_features)`` (N×M).
    j : int
        Column index of the target variable (0-based).
    V : np.ndarray or None, optional
        Candidate lags, shape ``(n_candidates, 2)``. Column 0: variable index
        (0-based); column 1: lag in samples. If None, returns only the target
        column.

    Returns
    -------
    np.ndarray
        Matrix with current target as first column and lagged variables as
        subsequent columns. Rows start at index :math:`L_{max}` so all lags
        are valid.

    Examples
    --------
    >>> import numpy as np
    >>> from xyz.utils import buildvectors
    >>> Y = np.random.randn(100, 3)
    >>> # Target column 1, with lags: (var 0, lag 1), (var 2, lag 2)
    >>> B = buildvectors(Y, 1, np.array([[0, 1], [2, 2]]))
    >>> B.shape[1]
    3
    """
    Y = np.asarray(Y)
    
    if V is None or len(V) == 0:
        if Y.ndim == 1:
            return Y.reshape(-1, 1)
        return Y[:, j:j+1]
        
    V = np.asarray(V, dtype=int)
    
    N = Y.shape[0]
    Lmax = np.max(V[:, 1])
    
    # Vectorized construction of lagged matrix A
    # A[row, i] = Y[row + Lmax - lag, var_idx]
    row_indices = np.arange(Lmax, N)[:, None]
    lags = V[:, 1][None, :]
    var_indices = V[:, 0][None, :]
    
    A = Y[row_indices - lags, var_indices]
    
    # B = [Y[Lmax:N, j], A]
    B = np.hstack([Y[Lmax:N, j:j+1], A])
    
    return B
