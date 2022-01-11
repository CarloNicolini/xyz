import pytest
import numpy as np
from xyz.univariate import entropy_linear, entropy_kernel
from xyz._continuos import *

A = np.loadtxt("../xyz/r.csv")


def test_entropy_linear():
    print(entropy_linear(A))


def test_entropy_kernel():
    print(entropy_kernel(A, 0.1))


def test_entropy_mvn():
    """
    Compares with its_Elin
    """
    assert np.allclose(MVNEntropy().fit(A).score(A), 14.174, rtol=1e-3)


def test_condentropy_mvn():
    """
    Compares with its_CElin
    """
    X = A[:, 1:]
    y = A[:, 0].reshape(-1, 1)
    correct_result = 1.4168
    assert np.allclose(
        MVNCondEntropy().fit(X, y).score(X, y), correct_result, rtol=1e-3
    )


def test_mi_mvn():
    X = A[:, 1:]
    y = A[:, 0].reshape(-1, 1)
    MVNMutualInformation().fit(X, y).score(X, y)
