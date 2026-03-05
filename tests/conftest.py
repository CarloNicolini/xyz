import subprocess
import pytest
import numpy as np
import os

def run_octave(command: str) -> str:
    """
    Run an octave command and return the stdout.
    Adds the 'matlab' directory to the octave path.
    """
    # Ensure paths are correct based on the repo root
    matlab_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'matlab'))
    full_command = f"addpath('{matlab_dir}'); {command}"
    result = subprocess.run(
        ['octave-cli', '--eval', full_command],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout.strip()

@pytest.fixture
def octave():
    """Fixture that provides the run_octave function."""
    return run_octave
