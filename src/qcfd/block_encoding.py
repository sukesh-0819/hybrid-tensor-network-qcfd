import numpy as np
from qiskit.circuit.library import UnitaryGate

def matrix_sqrt_psd(H):
    eigvals, eigvecs = np.linalg.eigh((H + H.conj().T) / 2)
    return (eigvecs * np.sqrt(np.clip(eigvals, 0, None))) @ eigvecs.conj().T

def block_encode(M, label):
    n = M.shape[0]
    c = np.linalg.norm(M, 2) * (1 + 1e-6)
    A = M / c
    I = np.eye(n)
 
    U = np.block([[A, matrix_sqrt_psd(I - A @ A.conj().T)],
                  [matrix_sqrt_psd(I - A.conj().T @ A), -A.conj().T]])
 
    assert np.allclose(U.conj().T @ U, np.eye(2 * n), atol=1e-9), "U is not unitary"
    assert np.allclose(c * U[:n, :n], M), "top-left block is not M"
    return UnitaryGate(U, label=label), c