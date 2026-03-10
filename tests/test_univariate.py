import pytest
import numpy as np
from xyz.univariate import entropy_linear, entropy_kernel
from xyz._continuous import *

A = np.loadtxt("tests/r.csv")


def test_entropy_linear():
    print(entropy_linear(A))


def test_entropy_kernel():
    print(entropy_kernel(A, 0.1))


def test_entropy_mvn(octave):
    """
    Compares with its_Elin
    """
    command = "A = dlmread('tests/r.csv', ' '); [e, covA] = its_Elin(A); disp(e);"
    output = octave(command)
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    expected = float(lines[-1])
    assert np.allclose(MVNEntropy().fit(A).score(A), expected, rtol=1e-3)

def test_condentropy_mvn(octave):
    """
    Compares with its_CElin
    """
    X = A[:, 1:]
    y = A[:, 0].reshape(-1, 1)
    
    # its_CElin takes B where first column is target, others are condition
    command = "A = dlmread('tests/r.csv', ' '); B = [A(:, 1), A(:, 2:end)]; [ce,S,Up,Am] = its_CElin(B); disp(ce);"
    output = octave(command)
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    expected = float(lines[-1])
    
    assert np.allclose(MVCondEntropy().fit(X, y).score(X, y), expected, rtol=1e-3)

def test_mi_mvn(octave):
    X = A[:, 1:]
    y = A[:, 0].reshape(-1, 1)
    
    # Mutual information for MVN is H(Y) - H(Y|X)
    command = "A = dlmread('tests/r.csv', ' '); B = [A(:, 1), A(:, 2:end)]; [ce,S,Up,Am] = its_CElin(B); [e, covA] = its_Elin(A(:, 1)); disp(e - ce);"
    output = octave(command)
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    expected = float(lines[-1])

    assert np.allclose(
        MVNMutualInformation().fit(X, y).score(X, y), expected, rtol=1e-3
    )


def test_ksg_mutual_information():
    """
    Test KSG mutual information estimator with synthetic data
    """
    # Generate correlated Gaussian data with known MI
    np.random.seed(42)
    n_samples = 1000

    # Independent variables should have MI ≈ 0
    X = np.random.normal(0, 1, (n_samples, 1))
    y = np.random.normal(0, 1, (n_samples, 1))

    ksg = KSGMutualInformation(k=3)
    mi_independent = ksg.fit(X, y).score(X, y)

    # MI should be close to 0 for independent variables
    assert abs(mi_independent) < 0.2, (
        f"MI for independent variables should be ~0, got {mi_independent}"
    )

    # Perfectly correlated variables should have high MI
    y_correlated = X + 0.1 * np.random.normal(0, 1, (n_samples, 1))
    mi_correlated = ksg.fit(X, y_correlated).score(X, y_correlated)

    # MI should be positive for correlated variables
    assert mi_correlated > 0.5, (
        f"MI for correlated variables should be > 0.5, got {mi_correlated}"
    )
    print(
        f"KSG MI - Independent: {mi_independent:.4f}, Correlated: {mi_correlated:.4f}"
    )


def test_ksg_entropy():
    """
    Test KSG entropy estimator with synthetic Gaussian data
    """
    np.random.seed(42)
    n_samples = 10000

    # 1D Gaussian with variance σ² has entropy = 0.5 * log(2πeσ²)
    sigma = 2.0
    X = np.random.normal(0, sigma, (n_samples, 1))

    ksg_entropy = KSGEntropy(k=3)
    estimated_entropy = ksg_entropy.fit(X).score(X)
    theoretical_entropy = 0.5 * np.log(2 * np.pi * np.e * sigma**2)

    # Allow for reasonable estimation error
    relative_error = abs(estimated_entropy - theoretical_entropy) / abs(
        theoretical_entropy
    )
    assert relative_error < 0.01, (
        f"Entropy estimation error too large: {relative_error:.3f}"
    )
    print(
        f"KSG Entropy - Estimated: {estimated_entropy:.4f}, Theoretical: {theoretical_entropy:.4f}"
    )


def test_ksg_consistency():
    """
    Test that KSG estimators are consistent with information theory identities
    """
    np.random.seed(42)
    n_samples = 500

    # Generate 2D Gaussian data
    mean = [0, 0]
    cov = [[1, 0.5], [0.5, 1]]
    data = np.random.multivariate_normal(mean, cov, n_samples)
    X = data[:, 0:1]
    y = data[:, 1:2]

    # Estimate entropies and MI
    entropy_est = KSGEntropy(k=3)
    mi_est = KSGMutualInformation(k=3)

    h_x = entropy_est.fit(X).score(X)
    h_y = entropy_est.fit(y).score(y)
    h_xy = entropy_est.fit(data).score(data)
    mi_xy = mi_est.fit(X, y).score(X, y)

    # Check information theory identity: MI(X,Y) = H(X) + H(Y) - H(X,Y)
    mi_from_entropies = h_x + h_y - h_xy
    relative_error = abs(mi_xy - mi_from_entropies) / max(
        abs(mi_xy), abs(mi_from_entropies)
    )

    assert relative_error < 0.8, (
        f"MI consistency check failed: error = {relative_error:.3f}"
    )
    print(
        f"KSG Consistency - MI direct: {mi_xy:.4f}, MI from entropies: {mi_from_entropies:.4f}"
    )


def test_ksg_entropy_regression_values():
    rng = np.random.default_rng(12345)
    x_1d = rng.normal(size=(256, 1))
    x_2d = rng.normal(size=(256, 2))

    assert np.isclose(
        KSGEntropy(k=3, metric="chebyshev").fit(x_1d).score(),
        1.3927752346455806,
    )
    assert np.isclose(
        KSGEntropy(k=3, metric="euclidean").fit(x_1d).score(),
        1.3927752346455806,
    )
    assert np.isclose(
        KSGEntropy(k=3, metric="chebyshev").fit(x_2d).score(),
        2.820724681678224,
    )


def test_ksg_mutual_information_regression_values():
    rng = np.random.default_rng(12345)
    x = rng.normal(size=(256, 1))
    y = 0.7 * x + 0.2 * rng.normal(size=(256, 1))

    assert np.isclose(
        KSGMutualInformation(k=3, algorithm=1, metric="chebyshev").fit(x, y).score(),
        1.2102575690004707,
    )
    assert np.isclose(
        KSGMutualInformation(k=3, algorithm=2, metric="chebyshev").fit(x, y).score(),
        1.128601195430562,
    )
    assert np.isclose(
        KSGMutualInformation(k=3, algorithm=1, metric="euclidean").fit(x, y).score(),
        0.9003241273631847,
    )


def test_mvksg_conditional_entropy():
    """
    Test multivariate KSG conditional entropy estimator
    """
    np.random.seed(42)
    n_samples = 1000

    # Generate correlated data: Y = X + noise
    X = np.random.normal(0, 1, (n_samples, 2))
    noise = np.random.normal(0, 0.5, (n_samples, 1))
    y = X[:, 0:1] + 0.5 * X[:, 1:2] + noise

    # Test conditional entropy
    cond_entropy_est = MVKSGCondEntropy(k=3)
    h_y_given_x = cond_entropy_est.fit(X, y).score(X, y)

    # Conditional entropy should be positive
    assert h_y_given_x > 0, f"Conditional entropy should be positive, got {h_y_given_x}"

    # For nearly perfect correlation, conditional entropy should be low
    X_nearly_perfect = np.random.normal(0, 1, (n_samples, 1))
    y_nearly_perfect = X_nearly_perfect + 0.01 * np.random.normal(
        0, 1, (n_samples, 1)
    )  # Nearly perfect correlation
    h_nearly_perfect = cond_entropy_est.fit(X_nearly_perfect, y_nearly_perfect).score(
        X_nearly_perfect, y_nearly_perfect
    )

    # Should be much lower than the noisy case
    assert h_nearly_perfect < h_y_given_x, (
        f"Conditional entropy for nearly perfect correlation should be < noisy case, got {h_nearly_perfect:.4f} vs {h_y_given_x:.4f}"
    )
    print(
        f"Conditional entropy - Noisy: {h_y_given_x:.4f}, Nearly Perfect: {h_nearly_perfect:.4f}"
    )


def test_mvksg_conditional_mutual_information():
    """
    Test conditional mutual information estimator
    """
    np.random.seed(42)
    n_samples = 800

    # Generate data where X and Y are conditionally independent given Z
    Z = np.random.normal(0, 1, (n_samples, 1))
    X = Z + 0.3 * np.random.normal(0, 1, (n_samples, 1))
    y = Z + 0.3 * np.random.normal(0, 1, (n_samples, 1))

    cmi_est = MVKSGCondMutualInformation(k=3)
    cmi_value = cmi_est.fit(X, y, Z).score(X, y, Z)

    # Should be close to 0 since X and Y are conditionally independent given Z
    assert abs(cmi_value) < 0.3, (
        f"CMI should be ~0 for conditionally independent variables, got {cmi_value}"
    )

    # Now test with conditionally dependent variables
    y_dependent = X + Z + 0.1 * np.random.normal(0, 1, (n_samples, 1))
    cmi_dependent = cmi_est.fit(X, y_dependent, Z).score(X, y_dependent, Z)

    # Should be positive for dependent variables
    assert cmi_dependent > 0.1, (
        f"CMI should be positive for dependent variables, got {cmi_dependent}"
    )
    print(f"CMI - Independent: {cmi_value:.4f}, Dependent: {cmi_dependent:.4f}")


def test_direct_ksg_conditional_mutual_information():
    np.random.seed(123)
    n_samples = 900

    z = np.random.normal(0, 1, (n_samples, 1))
    x = z + 0.3 * np.random.normal(0, 1, (n_samples, 1))
    y_independent = z + 0.3 * np.random.normal(0, 1, (n_samples, 1))
    y_dependent = x + z + 0.1 * np.random.normal(0, 1, (n_samples, 1))

    direct_est = DirectKSGConditionalMutualInformation(k=3)
    cmi_independent = direct_est.fit(x, y_independent, z).score(x, y_independent, z)
    cmi_dependent = direct_est.fit(x, y_dependent, z).score(x, y_dependent, z)

    assert abs(cmi_independent) < 0.25, (
        f"Direct KSG CMI should be ~0 for conditional independence, got {cmi_independent}"
    )
    assert cmi_dependent > 0.15, (
        f"Direct KSG CMI should be positive for conditional dependence, got {cmi_dependent}"
    )


def test_direct_ksg_cmi_tracks_existing_mvksg_estimator():
    rng = np.random.default_rng(321)
    n_samples = 700
    z = rng.normal(size=(n_samples, 2))
    x = np.column_stack([z[:, 0] + 0.2 * rng.normal(size=n_samples), rng.normal(size=n_samples)])
    y = np.column_stack([x[:, 0] + z[:, 1] + 0.2 * rng.normal(size=n_samples)])

    direct_value = DirectKSGConditionalMutualInformation(k=3).fit(x, y, z).score()
    identity_value = MVKSGCondMutualInformation(k=3).fit(x, y, z).score()

    assert np.isfinite(direct_value)
    assert np.isfinite(identity_value)
    assert abs(direct_value - identity_value) < 0.35


def test_mvksg_transfer_entropy():
    """
    Test transfer entropy estimator
    """
    np.random.seed(42)
    n_samples = 500
    lag = 1

    # Generate time series where X drives Y
    X = np.random.normal(0, 1, (n_samples, 1))
    y = np.zeros((n_samples, 1))

    # Y[t] = 0.7 * Y[t-1] + 0.3 * X[t-1] + noise
    for t in range(1, n_samples):
        y[t] = 0.7 * y[t - 1] + 0.3 * X[t - 1] + 0.2 * np.random.normal()

    te_est = MVKSGTransferEntropy(k=3, lag=lag)
    te_x_to_y = te_est.fit(X, y).score(X, y)
    te_y_to_x = te_est.fit(y, X).score(y, X)

    # Transfer entropy from X to Y should be higher than from Y to X
    assert te_x_to_y > te_y_to_x, (
        f"TE(X->Y) should be > TE(Y->X), got {te_x_to_y:.4f} vs {te_y_to_x:.4f}"
    )

    # TE(X->Y) should be positive, TE(Y->X) can be small/negative due to estimation variance
    assert te_x_to_y > 0, f"TE(X->Y) should be positive, got {te_x_to_y}"
    assert te_y_to_x > -0.2, f"TE(Y->X) should not be very negative, got {te_y_to_x}"

    print(f"Transfer Entropy - X->Y: {te_x_to_y:.4f}, Y->X: {te_y_to_x:.4f}")


def test_mvksg_partial_information_decomposition():
    """
    Test partial information decomposition
    """
    np.random.seed(42)
    n_samples = 600

    # Generate data with known information structure
    # X1 and X2 both predict Y, with some redundancy
    X1 = np.random.normal(0, 1, (n_samples, 1))
    X2 = 0.5 * X1 + np.random.normal(0, 1, (n_samples, 1))  # Correlated with X1
    y = 0.6 * X1 + 0.4 * X2 + 0.3 * np.random.normal(0, 1, (n_samples, 1))

    pid_est = MVKSGPartialInformationDecomposition(k=3)
    pid_result = pid_est.fit(X1, X2, y).score(X1, X2, y)

    # Check that all components are non-negative
    assert pid_result["unique_x1"] >= 0, (
        f"Unique X1 should be non-negative, got {pid_result['unique_x1']}"
    )
    assert pid_result["unique_x2"] >= 0, (
        f"Unique X2 should be non-negative, got {pid_result['unique_x2']}"
    )
    assert pid_result["redundant"] >= 0, (
        f"Redundant should be non-negative, got {pid_result['redundant']}"
    )
    assert pid_result["synergistic"] >= 0, (
        f"Synergistic should be non-negative, got {pid_result['synergistic']}"
    )

    # Total should equal sum of components
    total_components = (
        pid_result["unique_x1"]
        + pid_result["unique_x2"]
        + pid_result["redundant"]
        + pid_result["synergistic"]
    )
    total_error = abs(pid_result["total"] - total_components) / max(
        pid_result["total"], total_components
    )

    assert total_error < 0.3, (
        f"PID components should sum to total MI, error: {total_error:.3f}"
    )

    print(
        f"PID - Unique X1: {pid_result['unique_x1']:.4f}, Unique X2: {pid_result['unique_x2']:.4f}"
    )
    print(
        f"      Redundant: {pid_result['redundant']:.4f}, Synergistic: {pid_result['synergistic']:.4f}"
    )
    print(f"      Total: {pid_result['total']:.4f}")


def test_mvksg_multivariate_consistency():
    """
    Test consistency between different multivariate KSG estimators
    """
    np.random.seed(42)
    n_samples = 500

    # Generate 3D correlated data
    mean = [0, 0, 0]
    cov = [[1.0, 0.6, 0.3], [0.6, 1.0, 0.4], [0.3, 0.4, 1.0]]
    data = np.random.multivariate_normal(mean, cov, n_samples)

    X = data[:, 0:1]
    y = data[:, 1:2]
    Z = data[:, 2:3]

    # Test relationship: I(X;Y) >= I(X;Y|Z) (information processing inequality)
    mi_est = KSGMutualInformation(k=3)
    cmi_est = MVKSGCondMutualInformation(k=3)

    mi_xy = mi_est.fit(X, y).score(X, y)
    cmi_xy_given_z = cmi_est.fit(X, y, Z).score(X, y, Z)

    # Due to estimation noise, allow some tolerance
    assert mi_xy >= cmi_xy_given_z - 0.2, (
        f"I(X;Y) should be >= I(X;Y|Z), got {mi_xy:.4f} vs {cmi_xy_given_z:.4f}"
    )

    print(
        f"Information Processing - I(X;Y): {mi_xy:.4f}, I(X;Y|Z): {cmi_xy_given_z:.4f}"
    )
