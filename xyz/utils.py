import numpy as np


def quantize(y, c):
    M, m = np.max(y), np.min(y)
    bins = np.arange(m, M, (M - m) / c)
    return np.digitize(y, bins)
