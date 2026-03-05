import os

import numpy as np

from xyz._discrete import DiscretePartialTransferEntropy, DiscreteSelfEntropy, DiscreteTransferEntropy


def test_discrete_bte(octave):
    r_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "r.csv"))
    command = f"""
    Y = dlmread('{r_csv_path}', ' ');
    ii = 2; jj = 1; V = [1 1; 2 1];
    c = 6; quantizza = 'y';
    out = its_BTEbin(Y, V, ii, jj, c, quantizza);
    disp(out.Txy);
    """
    output = octave(command)
    expected = float([line.strip() for line in output.splitlines() if line.strip()][-1])

    X = np.loadtxt(r_csv_path)
    est = DiscreteTransferEntropy(driver_indices=[1], target_indices=[0], lags=1, c=6, quantize=True)
    est.fit(X)
    assert np.allclose(est.transfer_entropy_, expected, rtol=1e-3, atol=1e-5)


def test_discrete_pte(octave):
    r_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "r.csv"))
    command = f"""
    Y = dlmread('{r_csv_path}', ' ');
    ii = 2; jj = 1; V = [1 1; 2 1; 3 1];
    c = 6; quantizza = 'y';
    out = its_PTEbin(Y, V, ii, jj, c, quantizza);
    disp(out.Txy_z);
    """
    output = octave(command)
    expected = float([line.strip() for line in output.splitlines() if line.strip()][-1])

    X = np.loadtxt(r_csv_path)
    est = DiscretePartialTransferEntropy(
        driver_indices=[1],
        target_indices=[0],
        conditioning_indices=[2],
        lags=1,
        c=6,
        quantize=True,
    )
    est.fit(X)
    assert np.allclose(est.transfer_entropy_, expected, rtol=1e-3, atol=1e-5)


def test_discrete_se(octave):
    r_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "r.csv"))
    command = f"""
    Y = dlmread('{r_csv_path}', ' ');
    jj = 1; V = [1 1; 1 2];
    c = 6; quantizza = 'y';
    out = its_SEbin(Y, V, jj, c, quantizza);
    disp(out.Sy);
    """
    output = octave(command)
    expected = float([line.strip() for line in output.splitlines() if line.strip()][-1])

    X = np.loadtxt(r_csv_path)
    est = DiscreteSelfEntropy(target_indices=[0], lags=2, c=6, quantize=True)
    est.fit(X)
    assert np.allclose(est.self_entropy_, expected, rtol=1e-3, atol=1e-5)
