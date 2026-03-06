import numpy as np

from xyz import KSGTransferEntropy, RagwitzEmbeddingSearchCV


def test_ragwitz_embedding_search_returns_best_params():
    rng = np.random.default_rng(7)
    n_trials = 4
    n_samples = 250
    trials = []
    for _ in range(n_trials):
        driver = rng.normal(size=n_samples)
        target = np.zeros(n_samples)
        for t in range(3, n_samples):
            target[t] = (
                0.55 * target[t - 1]
                + 0.25 * target[t - 3]
                + 0.2 * driver[t - 1]
                + 0.1 * rng.normal()
            )
        trials.append(np.column_stack([target, driver]))

    X = np.stack(trials)
    search = RagwitzEmbeddingSearchCV(
        KSGTransferEntropy(driver_indices=[1], target_indices=[0], k=3),
        target_index=0,
        dimensions=(1, 2, 3),
        taus=(1, 2),
    ).fit(X)

    assert search.best_params_["lags"] in {1, 2, 3}
    assert search.best_params_["tau"] in {1, 2}
    assert np.isfinite(search.best_score_)
    assert len(search.cv_results_["params"]) == 6
    assert hasattr(search, "best_estimator_")


def test_ragwitz_embedding_search_parallel_matches_serial():
    rng = np.random.default_rng(11)
    trials = []
    for _ in range(3):
        n_samples = 220
        driver = rng.normal(size=n_samples)
        target = np.zeros(n_samples)
        for t in range(2, n_samples):
            target[t] = 0.5 * target[t - 1] + 0.2 * target[t - 2] + 0.2 * driver[t - 1]
        trials.append(np.column_stack([target, driver]))

    X = np.stack(trials)
    estimator = KSGTransferEntropy(driver_indices=[1], target_indices=[0], k=3)
    serial = RagwitzEmbeddingSearchCV(
        estimator,
        target_index=0,
        dimensions=(1, 2, 3),
        taus=(1, 2),
        n_jobs=1,
    ).fit(X)
    parallel = RagwitzEmbeddingSearchCV(
        estimator,
        target_index=0,
        dimensions=(1, 2, 3),
        taus=(1, 2),
        n_jobs=2,
    ).fit(X)

    assert parallel.best_params_ == serial.best_params_
    assert np.isclose(parallel.best_score_, serial.best_score_)
    assert np.allclose(
        parallel.cv_results_["mean_prediction_error"],
        serial.cv_results_["mean_prediction_error"],
    )
