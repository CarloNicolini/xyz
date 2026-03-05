import os
import numpy as np

def test_octave_fixture_Elin(octave):
    """Test calling its_Elin.m via octave."""
    r_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'r.csv'))
    command = f"""
    A = dlmread('{r_csv_path}', ' ');
    [e, covA] = its_Elin(A);
    disp(e);
    """
    output = octave(command)
    
    # Parse the last line as float
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    e_val = float(lines[-1])
    
    assert np.allclose(e_val, 4.0644, rtol=1e-3)
