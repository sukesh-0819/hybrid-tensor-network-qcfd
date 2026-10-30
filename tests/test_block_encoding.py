import numpy as np

from qcfd.block_encoding import halmos_dilation


def test_random_matrix():
    M = np.random.default_rng(0).standard_normal((8, 8))
    U, c = halmos_dilation(M)
    assert np.allclose(U.conj().T @ U, np.eye(16), atol=1e-9)
    assert np.allclose(c * U[:8, :8], M)


def test_non_normal_matrix_with_zero_rows():
    M = np.random.default_rng(1).standard_normal((8, 8))
    M[0] = 0.0
    M[-1] = 0.0
    U, c = halmos_dilation(M)
    assert np.allclose(U.conj().T @ U, np.eye(16), atol=1e-9)
    assert np.allclose(c * U[:8, :8], M)


def test_subnormalization_bounds_spectral_norm():
    M = np.random.default_rng(2).standard_normal((4, 4))
    _, c = halmos_dilation(M)
    assert c >= np.linalg.norm(M, 2)
