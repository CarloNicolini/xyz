from typing import Optional
import numpy as np
from .base import InfoTheoryMixin, InfoTheoryEstimator


class MVNInfoTheoryEstimator(InfoTheoryMixin, InfoTheoryEstimator):
    """
    Base class for information theory estimators based on multivariate normal assumption.
    Implements these measures

    Entropy expressions and their estimators for multivariate distributions
    """

    def __init__(
        self, cov: Optional[np.ndarray] = None, mean: Optional[np.ndarray] = None
    ):
        self.cov = cov
        self.mu = mean

    def fit(self, X, y=None):
        raise NotImplementedError("Abstract base class")
        pass

    def score(self, X, y=None):
        raise NotImplementedError("Abstract base class")
        pass


class MVNEntropy(MVNInfoTheoryEstimator):
    """
    Computes the differential entropy of a given multivariate set of observations
    assuming that the probability distribution function for these observations is Gaussian
    Theorem 8.4.1 Cover & Thomas
    """

    def fit(self, X, y=None):
        if self.cov is None:
            self.cov = np.cov(X, rowvar=False)
        return self

    def score(self, X, y=None):
        C = self.cov
        # Entropy for the multivariate Gaussian case:
        n_dims = C.shape[0]
        # e.g., Barnett PRL 2009
        s, logdet = np.linalg.slogdet(C)
        e = 0.5 * (s * logdet) + 0.5 * n_dims * np.log(2 * np.pi * np.exp(1))
        return e


class MVLNEntropy(MVNInfoTheoryEstimator):
    """
    Computes the differential entropy of a given multivariate set of observations
    assuming that the probability distribution function for these observations is LogNormal
    "Entropy Expressions and Their Estimators for Multivariate Distributions", Ahmed and Gokhale 1989
    Eq. 1.2
    """

    def fit(self, X, y=None):
        if self.cov is None:
            self.cov = np.cov(X, rowvar=False)
        if self.mu is None:
            self.mu = X.mean(axis=1)

        return self

    def score(self, X, y=None):
        return MVLNEntropy(cov=self.cov, mu=self.mu).fit(X).score(X, y) + self.mu.sum()


class MVParetoEntropy(MVNInfoTheoryEstimator):
    def fit(self, X, y=None):
        pass

    def score(self, X, y=None):
        # TODO equation 1.4 "Entropy Expressions and Their Estimators for Multivariate Distributions", Ahmed and Gokhale 1989
        raise NotImplementedError("To be implemented")
        return


class MVExponentialEntropy(MVNInfoTheoryEstimator):
    def fit(self, X, y=None):
        pass

    def score(self, X, y=None):
        # TODO equation 1.5 "Entropy Expressions and Their Estimators for Multivariate Distributions", Ahmed and Gokhale 1989
        raise NotImplementedError("To be implemented")
        return


class MVNCondEntropy(MVNInfoTheoryEstimator):
    def __init__(
        self,
        partial_cov: Optional[np.ndarray] = None,
        cov_x: Optional[np.ndarray] = None,
        cov_xy: Optional[np.ndarray] = None,
        cov_y: Optional[np.ndarray] = None,
    ):
        """
        See Barnett PRL 2009, Eq.1
        Parameters
        ----------
        partial_cov
        cov_x
        cov_xy
        """
        self.partial_cov = partial_cov
        self.cov_x = cov_x
        self.cov_y = cov_y
        self.cov_xy = cov_xy

    def fit(self, X, y):
        # if self.cov_x is None:
        #     self.cov_x = np.cov(X, rowvar=False)
        # if self.cov_y is None:
        #     self.cov_y = np.cov(y, rowvar=False)
        # if self.cov_xy is None:
        #     self.cov_xy = np.cov(X, y, rowvar=False)

        # TODO sistemare questo calcolo dove si fornisce la matrice di cross-covarianza nxm dove n è la
        #  dimensione di X e m  la dimensione di y (nel esempio matlab m=1)
        # self.partial_cov = self.cov_x - (
        #     np.linalg.multi_dot(self.cov_xy, np.linalg.inv(self.cov_y), self.cov_xy.T)
        # )
        alpha = np.linalg.lstsq(X, y)[0]
        res = y - X @ alpha

        self.partial_cov = np.cov(res, rowvar=False)
        if self.partial_cov.ndim < 2:
            self.partial_cov = self.partial_cov.reshape(-1, 1)
        return self

    def score(self, X, y=None):
        sign, log_det_part_cov = np.linalg.slogdet(self.partial_cov)
        n_dims = self.partial_cov.shape[0]
        return 0.5 * (sign * log_det_part_cov + n_dims * np.log(2 * np.pi * np.exp(1)))
