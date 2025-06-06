from abc import ABC
from typing import Any, Iterable, Optional

import numpy as np

from .base import InfoTheoryEstimator, InfoTheoryMixin
from .utils import cov as covariance


class DiscreteInfoTheoryEstimator(InfoTheoryMixin, InfoTheoryEstimator, ABC):
    """
    Base class for information theory estimators based on discrete variables
    """

    def __init__(self, alphabet: Iterable[Any] = None):
        self.alphabet = alphabet
