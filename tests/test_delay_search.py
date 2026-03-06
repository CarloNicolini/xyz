import numpy as np

from xyz import GaussianTransferEntropy, InteractionDelaySearchCV


def test_interaction_delay_search_recovers_known_delay():
    rng = np.random.default_rng(21)
    n_samples = 700
    true_delay = 2
    driver = rng.normal(size=n_samples)
    target = np.zeros(n_samples)
    for t in range(true_delay, n_samples):
        target[t] = (
            0.5 * target[t - 1]
            + 0.45 * driver[t - true_delay]
            + 0.1 * rng.normal()
        )

    X = np.column_stack([target, driver])
    search = InteractionDelaySearchCV(
        GaussianTransferEntropy(driver_indices=[1], target_indices=[0], lags=1),
        delays=(1, 2, 3, 4),
    ).fit(X)

    assert search.best_delay_ == true_delay
    assert search.best_score_ == search.best_estimator_.transfer_entropy_
    assert search.te_by_delay_.shape == (4, 2)


def test_interaction_delay_search_parallel_matches_serial():
    rng = np.random.default_rng(22)
    n_samples = 500
    driver = rng.normal(size=n_samples)
    target = np.zeros(n_samples)
    for t in range(2, n_samples):
        target[t] = 0.45 * target[t - 1] + 0.4 * driver[t - 2] + 0.1 * rng.normal()

    X = np.column_stack([target, driver])
    estimator = GaussianTransferEntropy(driver_indices=[1], target_indices=[0], lags=1)
    serial = InteractionDelaySearchCV(estimator, delays=(1, 2, 3, 4), n_jobs=1).fit(X)
    parallel = InteractionDelaySearchCV(estimator, delays=(1, 2, 3, 4), n_jobs=2).fit(X)

    assert parallel.best_delay_ == serial.best_delay_
    assert np.isclose(parallel.best_score_, serial.best_score_)
    assert np.allclose(parallel.te_by_delay_, serial.te_by_delay_)
