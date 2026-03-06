import numpy as np

from xyz import (
    GaussianCopulaConditionalMutualInformation,
    GaussianCopulaMutualInformation,
    GaussianCopulaTransferEntropy,
    GaussianTransferEntropy,
    MVNMutualInformation,
)


def test_gaussian_copula_mi_matches_gaussian_on_gaussian_data():
    rng = np.random.default_rng(101)
    data = rng.multivariate_normal(
        mean=[0.0, 0.0],
        cov=[[1.0, 0.7], [0.7, 1.0]],
        size=1200,
    )
    x = data[:, :1]
    y = data[:, 1:]

    gaussian_mi = MVNMutualInformation().fit(x, y).score()
    copula_mi = GaussianCopulaMutualInformation().fit(x, y).score()

    assert np.isclose(copula_mi, gaussian_mi, atol=0.05)


def test_gaussian_copula_cmi_handles_conditional_independence():
    rng = np.random.default_rng(202)
    z = rng.normal(size=(1000, 1))
    x = z + 0.3 * rng.normal(size=(1000, 1))
    y_independent = z + 0.3 * rng.normal(size=(1000, 1))
    y_dependent = x + z + 0.1 * rng.normal(size=(1000, 1))

    estimator = GaussianCopulaConditionalMutualInformation()
    cmi_independent = estimator.fit(x, y_independent, z).score()
    cmi_dependent = estimator.fit(x, y_dependent, z).score()

    assert abs(cmi_independent) < 0.1
    assert cmi_dependent > 0.3


def test_gaussian_copula_te_detects_direction_after_monotone_transform():
    rng = np.random.default_rng(303)
    n_samples = 900
    driver = rng.normal(size=n_samples)
    target = np.zeros(n_samples)
    for t in range(1, n_samples):
        target[t] = 0.55 * target[t - 1] + 0.35 * driver[t - 1] + 0.1 * rng.normal()

    latent_data = np.column_stack([target, driver])
    observed_data = np.column_stack([np.exp(target / 4.0), np.sinh(driver / 3.0)])

    latent_te = GaussianTransferEntropy(driver_indices=[1], target_indices=[0], lags=1).fit(
        latent_data
    )
    forward = GaussianCopulaTransferEntropy(driver_indices=[1], target_indices=[0], lags=1).fit(
        observed_data
    )
    reverse = GaussianCopulaTransferEntropy(driver_indices=[0], target_indices=[1], lags=1).fit(
        observed_data
    )

    assert np.isfinite(forward.transfer_entropy_)
    assert forward.transfer_entropy_ > reverse.transfer_entropy_
    assert np.isclose(forward.transfer_entropy_, latent_te.transfer_entropy_, atol=0.08)
