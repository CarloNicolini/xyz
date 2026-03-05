import os
import numpy as np
from xyz._continuos import GaussianTransferEntropy

def test_gaussian_bte(octave):
    r_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'r.csv'))
    
    # We test its_BTElin(data, ii, jj, V)
    # ii = driver (e.g., 2), jj = target (e.g., 1)
    # V = [1 1; 2 1] (target lag 1, driver lag 1)
    
    command = f"""
    data = dlmread('{r_csv_path}', ' ');
    ii = 2;
    jj = 1;
    V = [1 1; 2 1];
    out = its_BTElin(data, ii, jj, V);
    disp(out.Txy);
    disp(out.p_Txy);
    """
    output = octave(command)
    
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    expected_te = float(lines[-2])
    expected_p = float(lines[-1])
    
    # Python equivalent
    X = np.loadtxt(r_csv_path)
    # Note: the python interface should probably be GaussianTransferEntropy(driver_indices=[1], target_indices=[0], lags=1)
    # We will implement it exactly as the plan suggested
    gte = GaussianTransferEntropy(driver_indices=[1], target_indices=[0], lags=1)
    gte.fit(X)
    
    assert np.allclose(gte.transfer_entropy_, expected_te, rtol=1e-3, atol=1e-5)
    assert np.allclose(gte.p_value_, expected_p, rtol=1e-3, atol=1e-5)
