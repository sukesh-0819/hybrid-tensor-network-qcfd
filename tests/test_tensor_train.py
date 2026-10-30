import numpy as np

from qcfd.tensor_train import (dense_ansatz_state, simulate_ansatz_mps, tt_bond_dims, tt_to_diagonal_tto,
                               tt_to_vector, tto_to_dense, vector_to_tt)


def random_theta(n, n_layers, seed):
    return np.random.default_rng(seed).uniform(0, 2 * np.pi, n * (n_layers + 1))


def test_dense_state_is_normalized():
    psi = dense_ansatz_state(random_theta(4, 5, 0), 4, 5)
    assert np.isclose(np.linalg.norm(psi), 1.0)


def test_mps_simulation_matches_dense_statevector():
    for n in (3, 4, 6):
        theta = random_theta(n, 5, n)
        psi = dense_ansatz_state(theta, n, 5)
        cores = simulate_ansatz_mps(theta, n, 5, chi_max=2 ** n)
        assert np.allclose(tt_to_vector(cores), psi, atol=1e-10)


def test_truncation_reduces_bond_dimension():
    n = 8
    theta = random_theta(n, 5, 3)
    cores = simulate_ansatz_mps(theta, n, 5, chi_max=4)
    assert max(tt_bond_dims(cores)) <= 4


def test_tt_svd_round_trip():
    psi = np.random.default_rng(4).standard_normal(32)
    cores = vector_to_tt(psi, 5)
    assert np.allclose(tt_to_vector(cores), psi, atol=1e-10)


def test_smooth_field_has_low_bond_dimension():
    xs = np.linspace(0, 1, 2 ** 8)
    cores = vector_to_tt(np.sin(np.pi * xs), 8, tol=1e-10)
    assert max(tt_bond_dims(cores)) == 2


def test_diagonal_tto_equals_diag_of_state():
    psi = dense_ansatz_state(random_theta(3, 5, 5), 3, 5)
    W = tt_to_diagonal_tto(vector_to_tt(psi, 3))
    assert np.allclose(tto_to_dense(W), np.diag(psi), atol=1e-10)
