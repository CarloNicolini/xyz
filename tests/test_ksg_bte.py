import os
import numpy as np
from xyz._continuous import KSGTransferEntropy

def test_ksg_bte():
    r_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'r.csv'))
    X = np.loadtxt(r_csv_path)
    
    # We test KSGTransferEntropy on r.csv
    kte = KSGTransferEntropy(
        driver_indices=[1], 
        target_indices=[0], 
        lags=1,
        k=3,
        metric='chebyshev'
    )
    kte.fit(X)
    
    # Check that it produces a valid float
    assert np.isfinite(kte.transfer_entropy_)
    
    # Generate time series where driver causes target
    np.random.seed(42)
    n_samples = 500
    driver = np.random.normal(0, 1, n_samples)
    target = np.zeros(n_samples)
    for t in range(1, n_samples):
        target[t] = 0.5 * target[t-1] + 0.5 * driver[t-1] + 0.1 * np.random.normal()
        
    data = np.column_stack([target, driver])
    
    # TE driver -> target
    kte_d2t = KSGTransferEntropy(driver_indices=[1], target_indices=[0], lags=1, k=3)
    kte_d2t.fit(data)
    
    # TE target -> driver
    kte_t2d = KSGTransferEntropy(driver_indices=[0], target_indices=[1], lags=1, k=3)
    kte_t2d.fit(data)
    
    assert kte_d2t.transfer_entropy_ > kte_t2d.transfer_entropy_
    assert kte_d2t.transfer_entropy_ > 0.05

