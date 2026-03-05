import numpy as np
from sklearn.base import clone

from xyz import (
    GaussianTransferEntropy,
    KSGEntropy,
    KSGTransferEntropy,
    RagwitzEmbeddingSearchCV,
)


def test_root_exports_are_cloneable():
    estimator = KSGTransferEntropy(
        driver_indices=[1],
        target_indices=[0],
        lags=2,
        tau=2,
        delay=1,
        k=4,
    )
    cloned = clone(estimator)
    assert cloned.get_params()["lags"] == 2
    assert cloned.get_params()["tau"] == 2
    assert cloned.get_params()["delay"] == 1
    assert cloned.get_params()["k"] == 4


def test_fit_score_contract_for_time_series_estimators():
    rng = np.random.default_rng(42)
    n = 300
    driver = rng.normal(size=n)
    target = np.zeros(n)
    for t in range(1, n):
        target[t] = 0.5 * target[t - 1] + 0.4 * driver[t - 1] + 0.1 * rng.normal()
    data = np.column_stack([target, driver])

    estimator = GaussianTransferEntropy(driver_indices=[1], target_indices=[0], lags=1)
    estimator.fit(data)
    assert np.isfinite(estimator.transfer_entropy_)
    assert estimator.score() == estimator.transfer_entropy_


def test_entropy_estimator_score_after_fit():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(1000, 2))
    estimator = KSGEntropy(k=3)
    estimator.fit(X)
    assert np.isfinite(estimator.score())


def test_search_estimator_is_cloneable():
    estimator = RagwitzEmbeddingSearchCV(
        KSGTransferEntropy(driver_indices=[1], target_indices=[0]),
        target_index=0,
        dimensions=(1, 2),
        taus=(1, 2),
    )
    cloned = clone(estimator)
    assert cloned.get_params()["dimensions"] == (1, 2)
