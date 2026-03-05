from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


class InfoTheoryEstimator(BaseEstimator, ABC):
    """Base class for all information-theoretic estimators in ``xyz``.

    Subclasses are expected to follow scikit-learn conventions:
    parameter-only ``__init__``, ``fit(...) -> self``, fitted attributes with a
    trailing underscore, and a stable ``score`` method that returns the primary
    scalar quantity for the estimator.
    """

    score_attr_: str | None = None

    @staticmethod
    def _as_2d_array(X: Any, *, name: str = "X") -> np.ndarray:
        """Coerce an input array to shape ``(n_samples, n_features)``."""
        X = np.asarray(X)
        if X.ndim == 0:
            raise ValueError(f"{name} must be at least 1-dimensional")
        if X.ndim == 1:
            return X.reshape(-1, 1)
        if X.ndim == 2:
            return X
        raise ValueError(f"{name} must be 1D or 2D, got shape {X.shape}")

    @staticmethod
    def _as_trial_array(X: Any, *, name: str = "X") -> np.ndarray:
        """Coerce a time-series input to ``(trials, samples, features)``."""
        X = np.asarray(X)
        if X.ndim == 1:
            return X.reshape(1, -1, 1)
        if X.ndim == 2:
            return X[np.newaxis, ...]
        if X.ndim == 3:
            return X
        raise ValueError(
            f"{name} must be 1D, 2D or 3D for time-series estimators, got shape {X.shape}"
        )

    @staticmethod
    def _validate_matching_samples(*arrays: np.ndarray) -> None:
        """Ensure all provided arrays share the same number of rows."""
        n_samples = {np.asarray(array).shape[0] for array in arrays}
        if len(n_samples) > 1:
            raise ValueError("All inputs must have the same number of samples")

    def _score_from_attr(self, attr_name: str | None = None) -> float:
        """Return a scalar fitted attribute as the estimator score."""
        attr_name = attr_name or self.score_attr_
        if not attr_name:
            raise AttributeError(
                "This estimator does not define a default fitted score attribute."
            )
        check_is_fitted(self, attr_name)
        return float(getattr(self, attr_name))


class InfoTheoryMixin(ABC):
    """Mixin defining the minimal estimator protocol used across the package."""

    @abstractmethod
    def fit(self, X, y=None):
        """Fit estimator-specific internal state from data."""

    def score(self, *args, **kwargs):
        """Return the estimator's primary fitted scalar quantity."""
        if args or kwargs:
            self.fit(*args, **kwargs)
        if not isinstance(self, InfoTheoryEstimator):
            raise TypeError("InfoTheoryMixin.score requires InfoTheoryEstimator")
        return self._score_from_attr()
