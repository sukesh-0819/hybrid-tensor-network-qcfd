import numpy as np


def matrix_sqrt_psd(H):
    eigvals, eigvecs = np.linalg.eigh((H + H.conj().T) / 2)
    return (eigvecs * np.sqrt(np.clip(eigvals, 0, None))) @ eigvecs.conj().T


def halmos_dilation(M, margin=1e-6):
    n = M.shape[0]
    c = np.linalg.norm(M, 2) * (1 + margin)
    A = M / c
    I = np.eye(n)
    U = np.block([[A, matrix_sqrt_psd(I - A @ A.conj().T)],
                  [matrix_sqrt_psd(I - A.conj().T @ A), -A.conj().T]])
    if not np.allclose(U.conj().T @ U, np.eye(2 * n), atol=1e-9):
        raise AssertionError("dilation is not unitary")
    if not np.allclose(c * U[:n, :n], M):
        raise AssertionError("top-left block is not M")
    return U, c


def block_encode(M, label):
    from qiskit.circuit.library import UnitaryGate
    U, c = halmos_dilation(M)
    return UnitaryGate(U, label=label), c
