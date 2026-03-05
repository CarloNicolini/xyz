import numpy as np


def quantize(y, c):
    M, m = np.max(y), np.min(y)
    bins = np.arange(m, M, (M - m) / c)
    return np.digitize(y, bins)


def cov(X):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    # Return 2D covariance matrix. For 1D input reshaped to 2D, np.cov returns 0D array.
    # To be consistent, ensure we return 2D.
    return np.atleast_2d(np.cov(X, rowvar=False))


def buildvectors(Y, j, V=None):
    """
    Form observation matrix for entropy computation.
    
    Parameters
    ----------
    Y : np.ndarray
        Input multiple time series, dimension N*M
    j : int
        Index of the target variable (0-based)
    V : np.ndarray, optional
        List of candidates, dimension Nc*2, where Nc is number of candidates;
        1st column: index of the signal (0-based)
        2nd column: index of the lag
        
    Returns
    -------
    B : np.ndarray
        Complete matrix with the current samples of target j as first column,
        and lagged variables as subsequent columns.
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
