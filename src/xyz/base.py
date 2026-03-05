from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator


class InfoTheoryEstimator(BaseEstimator, ABC):
    """Base class for all information-theoretic estimators in ``xyz``.

    The class follows the scikit-learn estimator conventions and is intended to
    expose fitted quantities as attributes (for example ``transfer_entropy_``).
    """


class InfoTheoryMixin:
    """Mixin defining the minimal estimator protocol used across the package."""

    @abstractmethod
    def fit(self, X, y=None):
        """Fit estimator-specific internal state from data."""
        pass

    @abstractmethod
    def score(self, X, y=None):
        """
        Parameters
        ----------
        X : array-like
            Input data.
        y : array-like, optional
            Optional second input variable.

        Returns
        -------
        score : float
            Scalar estimate (entropy, mutual information, etc.).
        """
        pass
