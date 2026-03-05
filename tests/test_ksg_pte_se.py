import numpy as np

from xyz._continuos import KSGPartialTransferEntropy, KSGSelfEntropy


def test_ksg_pte():
    np.random.seed(42)
    n_samples = 1000

    driver = np.random.normal(0, 1, n_samples)
    z = np.random.normal(0, 1, n_samples)
    target = np.zeros(n_samples)
    for t in range(1, n_samples):
        target[t] = 0.8 * driver[t - 1] + np.random.normal(0, 0.1)

    data = np.column_stack([target, driver, z])

    kpte = KSGPartialTransferEntropy(
        driver_indices=[1],
        target_indices=[0],
        conditioning_indices=[2],
        lags=1,
        k=3,
    )
    kpte.fit(data)
    assert np.isfinite(kpte.transfer_entropy_)


def test_ksg_se():
    np.random.seed(42)
    n_samples = 500

    target = np.zeros(n_samples)
    for t in range(1, n_samples):
        target[t] = 0.8 * target[t - 1] + 0.2 * np.random.normal()

    data = np.column_stack([target])

    kse = KSGSelfEntropy(target_indices=[0], lags=1, k=3)
    kse.fit(data)

    assert np.isfinite(kse.self_entropy_)
    assert kse.self_entropy_ > 0.1
