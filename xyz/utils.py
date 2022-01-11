import numpy as np


def quantize(y, c):
    M, m = np.max(y), np.min(y)
    bins = np.arange(m, M, (M - m) / c)
    return np.digitize(y, bins)


def cov(X):
    if X.shape[1] == 1:
        return np.cov(X, rowvar=False).reshape(-1, 1)
    else:
        return np.cov(X, rowvar=False).reshape(-1, 1)
