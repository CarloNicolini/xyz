import os
import numpy as np
from xyz._continuos import GaussianSelfEntropy

def test_gaussian_se(octave):
    r_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'r.csv'))
    
    # We test its_SElin(data, jj, V)
    # jj = target (e.g., 1)
    # V specifies past components: V = [1 1; 1 2]
    
    command = f"""
    data = dlmread('{r_csv_path}', ' ');
    jj = 1;
    V = [1 1; 1 2];
    out = its_SElin(data, jj, V);
    disp(out.Sy);
    disp(out.p_Sy);
    """
    output = octave(command)
    
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    expected_se = float(lines[-2])
    expected_p = float(lines[-1])
    
    # Python equivalent
    X = np.loadtxt(r_csv_path)
    
    # Python estimator
    gse = GaussianSelfEntropy(target_indices=[0], lags=2)
    gse.fit(X)
    
    assert np.allclose(gse.self_entropy_, expected_se, rtol=1e-3, atol=1e-5)
    assert np.allclose(gse.p_value_, expected_p, rtol=1e-3, atol=1e-5)
