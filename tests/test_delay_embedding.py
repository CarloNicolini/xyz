import os
import numpy as np
from xyz.utils import buildvectors

def test_buildvectors(octave):
    r_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'r.csv'))
    
    # Let's say we want to build vectors for target variable 2 (which is 1 in python 0-based indexing)
    # And V = [1 1; 1 3; 2 4; 3 1; 2 1] (in 1-based Matlab indexing)
    # In python, V would be [[0, 1], [0, 3], [1, 4], [2, 1], [1, 1]]
    command = f"""
    Y = dlmread('{r_csv_path}', ' ');
    j = 2;
    V = [1 1; 1 3; 2 4; 3 1; 2 1];
    B = its_buildvectors(Y, j, V);
    dlmwrite(stdout, B, 'delimiter', ' ', 'precision', '%.15f');
    """
    output = octave(command)
    
    # Parse output to numpy array
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    
    expected_B = []
    for line in lines:
        try:
            row = [float(x) for x in line.split()]
            expected_B.append(row)
        except ValueError:
            pass
    expected_B = np.array(expected_B)
    
    # Python equivalent
    Y = np.loadtxt(r_csv_path)
    j = 1
    V = np.array([[0, 1], [0, 3], [1, 4], [2, 1], [1, 1]])
    
    B_python = buildvectors(Y, j, V)
    
    print("Expected:")
    print(expected_B[:2])
    print("Python:")
    print(B_python[:2])
    
    assert B_python.shape == expected_B.shape
    assert np.allclose(B_python, expected_B)

