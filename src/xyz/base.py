from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


class InfoTheoryEstimator(BaseEstimator, ABC):
    """Base class for all information-theoretic estimators in xyz.

    Subclasses follow scikit-learn conventions: parameter-only ``__init__``,
    ``fit(...) -> self``, fitted attributes with a trailing underscore, and
    a stable ``score`` method that returns the primary scalar quantity
    (e.g. entropy, mutual information, transfer entropy).

    Attributes
    ----------
    score_attr_ : str or None
        Name of the fitted attribute used as the default score.

    Examples
    --------
    Concrete estimators (e.g. :class:`xyz.KSGMutualInformation`) are fitted
    on data and expose the estimate via ``score()``:

    >>> import numpy as np
    >>> from xyz import KSGMutualInformation
    >>> rng = np.random.default_rng(42)
    >>> X = rng.normal(size=(500, 1))
    >>> y = 0.7 * X + 0.3 * rng.normal(size=(500, 1))
    >>> est = KSGMutualInformation(k=3).fit(X, y)
    >>> mi = est.score()
    >>> np.isfinite(mi)
    True
    """

    score_attr_: str | None = None

    @staticmethod
    def _validate_matching_samples(*arrays: np.ndarray) -> None:
        """Ensure all provided arrays share the same number of rows.

        Raises
        ------
        ValueError
            If any array has a different number of rows.
        """
        n_samples = {np.asarray(array).shape[0] for array in arrays}
        if len(n_samples) > 1:
            raise ValueError("All inputs must have the same number of samples")

    def _score_from_attr(self, attr_name: str | None = None) -> float:
        """Return the scalar fitted attribute used as the estimator score.

        Parameters
        ----------
        attr_name : str or None, optional
            Attribute to use. Defaults to :attr:`score_attr_`.

        Returns
        -------
        float
            The fitted score value.

        Raises
        ------
        AttributeError
            If no default score attribute is defined or estimator is not fitted.
        """
        attr_name = attr_name or self.score_attr_
        if not attr_name:
            raise AttributeError(
                "This estimator does not define a default fitted score attribute."
            )
        check_is_fitted(self, attr_name)
        return float(getattr(self, attr_name))


class InfoTheoryMixin(ABC):
    """Mixin defining the minimal estimator protocol used across the package.

    Requires :meth:`fit`; :meth:`score` may optionally refit when given data.
    """

    @abstractmethod
    def fit(self, X, y=None):
        """Fit estimator-specific internal state from data.

        Parameters
        ----------
        X : array-like
            Input data (e.g. driver/target time series or observations).
        y : array-like or None, optional
            Optional second argument (e.g. target for MI, or conditioning).

        Returns
        -------
        self
            Fitted estimator.
        """

    def score(self, *args, **kwargs):
        """Return the estimator's primary fitted scalar quantity.

        If additional positional/keyword arguments are passed, refits the
        estimator on that data and then returns the score.
        """
        if args or kwargs:
            self.fit(*args, **kwargs)
        if not isinstance(self, InfoTheoryEstimator):
            raise TypeError("InfoTheoryMixin.score requires InfoTheoryEstimator")
        return self._score_from_attr()
