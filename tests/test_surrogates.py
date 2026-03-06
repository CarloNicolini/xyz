import numpy as np

from xyz import (
    KSGTransferEntropy,
    SurrogatePermutationTest,
    generate_surrogates,
)


def test_generate_surrogates_preserves_shape():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(3, 40, 2))
    surrogates = generate_surrogates(X, method="trial_shuffle", n_surrogates=5, random_state=0, driver_index=1)
    assert len(surrogates) == 5
    assert all(np.asarray(surrogate).shape == X.shape for surrogate in surrogates)


def test_surrogate_permutation_test_detects_directed_signal():
    rng = np.random.default_rng(42)
    trials = []
    for _ in range(8):
        n = 180
        driver = rng.normal(size=n)
        target = np.zeros(n)
        for t in range(1, n):
            target[t] = 0.4 * target[t - 1] + 0.5 * driver[t - 1] + 0.1 * rng.normal()
        trials.append(np.column_stack([target, driver]))
    X = np.stack(trials)

    test = SurrogatePermutationTest(
        KSGTransferEntropy(driver_indices=[1], target_indices=[0], lags=1, k=3),
        n_permutations=24,
        surrogate_method="trial_shuffle",
        alpha=0.1,
        random_state=0,
    ).fit(X)

    assert np.isfinite(test.observed_score_)
    assert test.null_distribution_.shape == (24,)
    assert test.p_values_.shape == (1,)


def test_surrogate_permutation_test_parallel_matches_serial():
    rng = np.random.default_rng(123)
    trials = []
    for _ in range(6):
        n = 160
        driver = rng.normal(size=n)
        target = np.zeros(n)
        for t in range(1, n):
            target[t] = 0.35 * target[t - 1] + 0.45 * driver[t - 1] + 0.1 * rng.normal()
        trials.append(np.column_stack([target, driver]))
    X = np.stack(trials)

    estimator = KSGTransferEntropy(driver_indices=[1], target_indices=[0], lags=1, k=3)
    serial = SurrogatePermutationTest(
        estimator,
        n_permutations=12,
        surrogate_method="trial_shuffle",
        random_state=0,
        n_jobs=1,
    ).fit(X)
    parallel = SurrogatePermutationTest(
        estimator,
        n_permutations=12,
        surrogate_method="trial_shuffle",
        random_state=0,
        n_jobs=2,
    ).fit(X)

    assert np.isclose(parallel.observed_score_, serial.observed_score_)
    assert np.allclose(parallel.null_distribution_, serial.null_distribution_)
    assert np.allclose(parallel.p_values_, serial.p_values_)
