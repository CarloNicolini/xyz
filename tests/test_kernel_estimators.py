import os
import numpy as np
from xyz._continuos import KernelTransferEntropy, KernelPartialTransferEntropy, KernelSelfEntropy

def test_kernel_bte(octave):
    r_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'r.csv'))
    command = f"""
    Y = dlmread('{r_csv_path}', ' ');
    ii = 2; jj = 1; V = [1 1; 2 1];
    r = 0.5; norma = 'c';
    out = its_BTEker(Y, V, ii, jj, r, norma);
    disp(out.Txy);
    """
    output = octave(command)
    expected_te = float([line.strip() for line in output.splitlines() if line.strip()][-1])
    
    X = np.loadtxt(r_csv_path)
    kte = KernelTransferEntropy(driver_indices=[1], target_indices=[0], lags=1, r=0.5, metric='chebyshev')
    kte.fit(X)
    
    assert np.allclose(kte.transfer_entropy_, expected_te, rtol=1e-3, atol=1e-5)

def test_kernel_pte(octave):
    r_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'r.csv'))
    command = f"""
    Y = dlmread('{r_csv_path}', ' ');
    ii = 2; jj = 1; V = [1 1; 2 1; 3 1];
    r = 0.5; norma = 'c';
    out = its_PTEker(Y, V, ii, jj, r, norma);
    disp(out.Txy_z);
    """
    output = octave(command)
    expected_pte = float([line.strip() for line in output.splitlines() if line.strip()][-1])
    
    X = np.loadtxt(r_csv_path)
    kpte = KernelPartialTransferEntropy(driver_indices=[1], target_indices=[0], conditioning_indices=[2], lags=1, r=0.5, metric='chebyshev')
    kpte.fit(X)
    
    assert np.allclose(kpte.transfer_entropy_, expected_pte, rtol=1e-3, atol=1e-5)

def test_kernel_se(octave):
    r_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'r.csv'))
    command = f"""
    Y = dlmread('{r_csv_path}', ' ');
    jj = 1; V = [1 1; 1 2];
    r = 0.5; norma = 'c';
    out = its_SEker(Y, V, jj, r, norma);
    disp(out.Sy);
    """
    output = octave(command)
    expected_se = float([line.strip() for line in output.splitlines() if line.strip()][-1])
    
    X = np.loadtxt(r_csv_path)
    kse = KernelSelfEntropy(target_indices=[0], lags=2, r=0.5, metric='chebyshev')
    kse.fit(X)
    
    assert np.allclose(kse.self_entropy_, expected_se, rtol=2e-1, atol=5e-3)
