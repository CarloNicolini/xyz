import numpy as np
from scipy.spatial.distance import pdist


def entropy_linear(A: np.ndarray) -> float:
    """
    Linear Gaussian Estimation of the Shannon Entropy
    Computes the shannon entropy of a multivariate dataset A
    A is N*M multivariate data (N observations, M variables)
    """
    C = np.cov(A.T)

    # Entropy for the multivariate Gaussian case:
    N, M = A.shape
    # e.g., Barnett PRL 2009
    e = 0.5 * np.log(np.linalg.det(C)) + 0.5 * N * np.log(2 * np.pi * np.exp(1))
    return e


def entropy_kernel(Y: np.ndarray, r: float, metric: str = "chebyshev") -> float:
    """
    Computes the entropy of the M-dimensional variable Y.
    By default uses step kernel (Chebyshev distance).

    Look here for the details
    https://www.youtube.com/watch?v=aDlv5rn0938&list=PLOfPLLxr5gsVLSlmzcMnsFANb-uWkArby&index=21
    """
    return -np.log((pdist(Y, metric=metric) < r).mean())


# for details on Kraskov estimator
# https://lizliz.github.io/teaspoon/_modules/teaspoon/parameter_selection/MI_delay.html


def entropy_binning(Y, c, quantize, log_base: str = "nat"):
    """
    Binning Estimation of the Shannon Entropy
    Computes the shannon entropy of the columns of the matrix A
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
