from typing import Optional, Iterable, Any
import numpy as np
from .base import InfoTheoryMixin, InfoTheoryEstimator
from .utils import cov as covariance
from abc import ABC


class DiscreteInfoTheoryEstimator(InfoTheoryMixin, InfoTheoryEstimator, ABC):
    """
    Base class for information theory estimators based on discrete variables
    """

    def __init__(self, alphabet: Iterable[Any] = None):
        self.alphabet = alphabet
