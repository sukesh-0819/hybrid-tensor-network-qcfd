import numpy as np

import sys
import os

# Add root directory to module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.qcfd.burgers import setup_classical_burgers
from src.qcfd.block_encoding import block_encode

def test_classical_grid():
    x, dx, dt, n_steps, D1, D2, sol = setup_classical_burgers(N=8)
    assert sol.shape[1] == 8
    assert np.allclose(sol[0, 0], 0.0)

def test_block_encoding():
    matrix = np.eye(4) * 0.5
    gate, c = block_encode(matrix, "TestGate")
    assert gate.params[0].shape == (8, 8)