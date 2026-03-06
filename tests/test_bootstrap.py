import numpy as np
import pytest

from xyz import BootstrapEstimate, GaussianCopulaMutualInformation, GaussianTransferEntropy


def test_bootstrap_estimate_iid_returns_interval():
    rng = np.random.default_rng(404)
    x = rng.normal(size=(500, 1))
    y = 0.6 * x + 0.3 * rng.normal(size=(500, 1))

    bootstrap = BootstrapEstimate(
        GaussianCopulaMutualInformation(),
        n_bootstrap=24,
        method="iid",
        random_state=0,
        n_jobs=2,
    ).fit(x, y)

    assert np.isfinite(bootstrap.estimate_)
    assert bootstrap.bootstrap_distribution_.shape == (24,)
    assert bootstrap.ci_low_ <= bootstrap.estimate_ <= bootstrap.ci_high_
    assert bootstrap.score() == bootstrap.estimate_


def test_bootstrap_estimate_trial_resampling_for_te():
    rng = np.random.default_rng(405)
    trials = []
    for _ in range(6):
        n = 180
        driver = rng.normal(size=n)
        target = np.zeros(n)
        for t in range(1, n):
            target[t] = 0.4 * target[t - 1] + 0.45 * driver[t - 1] + 0.1 * rng.normal()
        trials.append(np.column_stack([target, driver]))
    X = np.stack(trials)

    bootstrap = BootstrapEstimate(
        GaussianTransferEntropy(driver_indices=[1], target_indices=[0], lags=1),
        n_bootstrap=20,
        method="trial",
        random_state=0,
        n_jobs=2,
    ).fit(X)

    assert np.isfinite(bootstrap.estimate_)
    assert bootstrap.bootstrap_distribution_.shape == (20,)
    assert bootstrap.ci_low_ < bootstrap.ci_high_


def test_bootstrap_estimate_block_resampling_for_single_trial_te():
    rng = np.random.default_rng(406)
    n = 320
    driver = rng.normal(size=n)
    target = np.zeros(n)
    for t in range(1, n):
        target[t] = 0.45 * target[t - 1] + 0.35 * driver[t - 1] + 0.1 * rng.normal()
    X = np.column_stack([target, driver])

    bootstrap = BootstrapEstimate(
        GaussianTransferEntropy(driver_indices=[1], target_indices=[0], lags=1),
        n_bootstrap=16,
        method="block",
        block_length=32,
        random_state=0,
    ).fit(X)

    assert np.isfinite(bootstrap.estimate_)
    assert bootstrap.bootstrap_distribution_.shape == (16,)
    assert bootstrap.ci_low_ < bootstrap.ci_high_


def test_bootstrap_estimate_trial_method_requires_multiple_trials():
    rng = np.random.default_rng(407)
    X = rng.normal(size=(200, 2))

    with pytest.raises(ValueError, match="requires at least two trials"):
        BootstrapEstimate(
            GaussianTransferEntropy(driver_indices=[1], target_indices=[0], lags=1),
            n_bootstrap=8,
            method="trial",
            random_state=0,
        ).fit(X)
