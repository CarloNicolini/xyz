from typing import Optional
import numpy as np
from .base import InfoTheoryMixin, InfoTheoryEstimator
from .utils import cov as covariance
from abc import ABC


class MVInfoTheoryEstimator(InfoTheoryMixin, InfoTheoryEstimator, ABC):
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


class MVNEntropy(MVInfoTheoryEstimator):
    """
    Computes the differential entropy of a given multivariate set of observations
    assuming that the probability distribution function for these observations is Gaussian
    Theorem 8.4.1 Cover & Thomas
    """

    def fit(self, X, y=None):
        if self.cov is None:
            self.cov = covariance(X)
        return self

    def score(self, X, y=None):
        C = self.cov
        # Entropy for the multivariate Gaussian case:
        n_dims = C.shape[0]
        # e.g., Barnett PRL 2009
        s, logdet = np.linalg.slogdet(C)
        e = 0.5 * (s * logdet) + 0.5 * n_dims * np.log(2 * np.pi * np.exp(1))
        return e


class MVLNEntropy(MVInfoTheoryEstimator):
    """
    Computes the differential entropy of a given multivariate set of observations
    assuming that the probability distribution function for these observations is LogNormal
    "Entropy Expressions and Their Estimators for Multivariate Distributions", Ahmed and Gokhale 1989
    Eq. 1.2
    """

    def fit(self, X, y=None):
        if self.cov is None:
            self.cov = covariance(X)
        if self.mu is None:
            self.mu = X.mean(axis=1)

        return self

    def score(self, X, y=None):
        return MVLNEntropy(cov=self.cov).fit(X).score(X, y) + self.mu.sum()


class MVParetoEntropy(MVInfoTheoryEstimator):
    def fit(self, X, y=None):
        pass

    def score(self, X, y=None):
        # TODO equation 1.4 "Entropy Expressions and Their Estimators for Multivariate Distributions", Ahmed and Gokhale 1989
        raise NotImplementedError("To be implemented")
        pass


class MVExponentialEntropy(MVInfoTheoryEstimator):
    def fit(self, X, y=None):
        pass

    def score(self, X, y=None):
        # TODO equation 1.5 "Entropy Expressions and Their Estimators for Multivariate Distributions", Ahmed and Gokhale 1989
        raise NotImplementedError("To be implemented")
        pass


class MVCondEntropy(MVInfoTheoryEstimator):
    def __init__(
        self,
        partial_cov: Optional[np.ndarray] = None,
        cov_x: Optional[np.ndarray] = None,
        cov_xy: Optional[np.ndarray] = None,
        cov_y: Optional[np.ndarray] = None,
    ):
        """
        Conditional entropy H(Y;X) = H(X,Y) - H(X)
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
        #     self.cov_x = cov(X)
        # if self.cov_y is None:
        #     self.cov_y = cov(y)
        # if self.cov_xy is None:
        #     self.cov_xy = cov(X,y)????

        # TODO sistemare questo calcolo dove si fornisce la matrice di cross-covarianza nxm dove n ?? la
        #  dimensione di X e m  la dimensione di y (nel esempio matlab m=1)
        # self.partial_cov = self.cov_x - (
        #     np.linalg.multi_dot(self.cov_xy, np.linalg.inv(self.cov_y), self.cov_xy.T)
        # )
        alpha = np.linalg.lstsq(X, y)[0]
        # computes the residuals epsilon
        epsilon = y - X @ alpha

        self.partial_cov = covariance(epsilon)
        return self

    def score(self, X, y=None):
        sign, log_det_part_cov = np.linalg.slogdet(self.partial_cov)
        n_dims = self.partial_cov.shape[0]
        return 0.5 * (sign * log_det_part_cov + n_dims * np.log(2 * np.pi * np.exp(1)))


class MVNMutualInformation(MVCondEntropy):
    def __init__(
        self,
    ):
        self.hy = None
        self.chxy = None

    def fit(self, X, y):
        self.hy = MVNEntropy().fit(X=y).score(X)
        self.chxy = MVCondEntropy().fit(X, y).score(X, y)
        return self

    def score(self, X, y=None):
        return self.hy - self.chxy
