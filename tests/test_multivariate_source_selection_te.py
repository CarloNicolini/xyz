import numpy as np

from xyz import GaussianPartialTransferEntropy, GaussianTransferEntropy, GreedySourceSelectionTransferEntropy


def test_gaussian_transfer_entropy_supports_multivariate_drivers():
    rng = np.random.default_rng(501)
    n_samples = 800
    driver_1 = rng.normal(size=n_samples)
    driver_2 = rng.normal(size=n_samples)
    noise_driver = rng.normal(size=n_samples)
    target = np.zeros(n_samples)

    for t in range(1, n_samples):
        target[t] = (
            0.5 * target[t - 1]
            + 0.35 * driver_1[t - 1]
            + 0.2 * driver_2[t - 1]
            + 0.1 * rng.normal()
        )

    X = np.column_stack([target, driver_1, driver_2, noise_driver])
    te_1 = GaussianTransferEntropy(driver_indices=[1], target_indices=[0], lags=1).fit(X)
    te_2 = GaussianTransferEntropy(driver_indices=[2], target_indices=[0], lags=1).fit(X)
    te_12 = GaussianTransferEntropy(driver_indices=[1, 2], target_indices=[0], lags=1).fit(X)

    assert te_12.transfer_entropy_ > te_1.transfer_entropy_
    assert te_12.transfer_entropy_ > te_2.transfer_entropy_


def test_greedy_source_selection_te_recovers_true_sources():
    rng = np.random.default_rng(502)
    n_samples = 900
    driver_1 = rng.normal(size=n_samples)
    driver_2 = rng.normal(size=n_samples)
    noise_driver = rng.normal(size=n_samples)
    target = np.zeros(n_samples)

    for t in range(1, n_samples):
        target[t] = (
            0.45 * target[t - 1]
            + 0.30 * driver_1[t - 1]
            + 0.18 * driver_2[t - 1]
            + 0.1 * rng.normal()
        )

    X = np.column_stack([target, driver_1, driver_2, noise_driver])
    selector = GreedySourceSelectionTransferEntropy(
        GaussianPartialTransferEntropy(
            driver_indices=[1],
            target_indices=[0],
            conditioning_indices=[],
            lags=1,
        ),
        candidate_sources=[1, 2, 3],
        max_sources=3,
        min_improvement=0.01,
    ).fit(X)

    assert selector.selected_sources_ == [1, 2]
    assert selector.best_estimator_.driver_indices == [1, 2]
    assert selector.best_score_ > 0
    assert len(selector.selection_history_) == 2
