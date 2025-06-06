from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator


class InfoTheoryEstimator(BaseEstimator, ABC):
    pass


class InfoTheoryMixin:
    @abstractmethod
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def score(self, X, y=None):
        """

        Parameters
        ----------
        X
        y

        Returns
        -------
        score: float
        """
        pass
