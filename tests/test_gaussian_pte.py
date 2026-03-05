import os
import numpy as np
from xyz._continuos import GaussianPartialTransferEntropy

def test_gaussian_pte(octave):
    r_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'r.csv'))
    
    # We test its_PTElin(data, ii, jj, V)
    # ii = driver (e.g., 2), jj = target (e.g., 1)
    # Z (conditioning) will be variable 3.
    # V specifies past components: V = [1 1; 2 1; 3 1]
    
    command = f"""
    data = dlmread('{r_csv_path}', ' ');
    ii = 2;
    jj = 1;
    V = [1 1; 2 1; 3 1];
    out = its_PTElin(data, ii, jj, V);
    disp(out.Txy_z);
    disp(out.p_Txy_z);
    """
    output = octave(command)
    
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    expected_te = float(lines[-2])
    expected_p = float(lines[-1])
    
    # Python equivalent
    X = np.loadtxt(r_csv_path)
    # target_indices=[0], driver_indices=[1], conditioning_indices=[2]
    gpte = GaussianPartialTransferEntropy(
        driver_indices=[1], 
        target_indices=[0], 
        conditioning_indices=[2], 
        lags=1
    )
    gpte.fit(X)
    
    assert np.allclose(gpte.transfer_entropy_, expected_te, rtol=1e-3, atol=1e-5)
    assert np.allclose(gpte.p_value_, expected_p, rtol=1e-3, atol=1e-5)
