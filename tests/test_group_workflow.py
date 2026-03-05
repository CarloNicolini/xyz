import numpy as np

from xyz import GaussianTransferEntropy, GroupTEAnalysis


def test_group_workflow_harmonizes_embedding():
    rng = np.random.default_rng(17)
    datasets = []
    for _ in range(3):
        trials = []
        for _ in range(3):
            n = 220
            driver = rng.normal(size=n)
            target = np.zeros(n)
            for t in range(2, n):
                target[t] = (
                    0.45 * target[t - 1]
                    + 0.15 * target[t - 2]
                    + 0.35 * driver[t - 1]
                    + 0.1 * rng.normal()
                )
            trials.append(np.column_stack([target, driver]))
        datasets.append(np.stack(trials))

    group = GroupTEAnalysis(
        GaussianTransferEntropy(driver_indices=[1], target_indices=[0], lags=1),
        target_index=0,
        dimensions=(1, 2),
        taus=(1, 2),
    ).fit(datasets)

    assert "lags" in group.common_params_
    assert "tau" in group.common_params_
    assert len(group.subject_estimators_) == 3
    assert np.isfinite(group.group_score_)
