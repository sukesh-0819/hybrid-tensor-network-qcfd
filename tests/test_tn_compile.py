import numpy as np

from qcfd.tensor_train import dense_ansatz_state, simulate_ansatz_mps, vector_to_tt
from qcfd.tn_compile import compile_from_cores, compiled_to_dense


def random_state(n, seed):
    theta = np.random.default_rng(seed).uniform(0, 2 * np.pi, n * 6)
    return theta, dense_ansatz_state(theta, n, 5)


def test_compiled_block_equals_diagonal():
    for n in (3, 4):
        theta, psi = random_state(n, n)
        compiled = compile_from_cores(simulate_ansatz_mps(theta, n, 5, chi_max=2 ** n))
        U = compiled_to_dense(compiled)
        assert np.allclose(U.conj().T @ U, np.eye(U.shape[0]), atol=1e-9)
        assert np.allclose(U[:2 ** n, :2 ** n] * compiled["scale"], np.diag(psi), atol=1e-9)


def test_scale_is_a_valid_subnormalization():
    for n in (3, 4, 5):
        _, psi = random_state(n, 10 + n)
        compiled = compile_from_cores(vector_to_tt(psi, n))
        assert compiled["scale"] >= np.max(np.abs(psi))


def test_larger_bond_register_still_correct():
    theta, psi = random_state(3, 7)
    compiled = compile_from_cores(simulate_ansatz_mps(theta, 3, 5, chi_max=8), n_bond=2)
    assert compiled["n_bond"] == 2
    U = compiled_to_dense(compiled)
    assert np.allclose(U[:8, :8] * compiled["scale"], np.diag(psi), atol=1e-9)


def test_smooth_field_scale_close_to_ideal():
    xs = np.linspace(0, 1, 2 ** 7)
    psi = np.sin(np.pi * xs)
    psi /= np.linalg.norm(psi)
    compiled = compile_from_cores(vector_to_tt(psi, 7, tol=1e-10))
    assert compiled["scale"] / np.max(np.abs(psi)) < 1.05


def test_ancilla_count_scales_with_sites_and_bond():
    theta, _ = random_state(4, 3)
    compiled = compile_from_cores(simulate_ansatz_mps(theta, 4, 5, chi_max=4))
    assert compiled["n_aux"] == compiled["n_bond"] + 4
    assert len(compiled["ops"]) == 8
