import numpy as np

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.qcfd.tensor_network import circuit_to_tt, ry_matrix, tn_compile, tt_to_vector


def reference_state(theta, n, n_layers):
    T = np.zeros((2,) * n)
    T[(0,) * n] = 1.0
    p = 0
    for layer in range(n_layers + 1):
        for q in range(n):
            T = np.moveaxis(np.tensordot(ry_matrix(theta[p]), T, axes=([1], [q])), 0, q)
            p += 1
        if layer < n_layers:
            for q in range(n - 1):
                index = [slice(None)] * n
                index[q] = 1
                T[tuple(index)] = np.flip(T[tuple(index)], axis=q)
    return T.transpose(list(range(n))[::-1]).reshape(-1)


def random_theta(n, n_layers, seed):
    return np.random.default_rng(seed).uniform(0, 2 * np.pi, n * (n_layers + 1))


def ops_to_unitary(ops, n, n_bond):
    n_qubits = 2 * n + n_bond
    U = np.eye(2 ** n_qubits, dtype=complex)
    for k, i, matrix in ops:
        targets = list(range(n, n + n_bond)) + [n + n_bond + k]
        F = np.zeros_like(U)
        for s in range(2 ** n_qubits):
            if ((s >> k) & 1) != i:
                F[s, s] = 1.0
                continue
            j = sum(((s >> q) & 1) << pos for pos, q in enumerate(targets))
            base = s
            for q in targets:
                base &= ~(1 << q)
            for jp in range(2 ** len(targets)):
                sp = base
                for pos, q in enumerate(targets):
                    sp |= ((jp >> pos) & 1) << q
                F[sp, s] = matrix[jp, j]
        U = F @ U
    return U


def test_tt_of_circuit_matches_statevector():
    for n in (3, 4, 6):
        theta = random_theta(n, 5, n)
        cores = circuit_to_tt(theta, n, 5, chi_max=2 ** n)
        assert np.allclose(tt_to_vector(cores), reference_state(theta, n, 5), atol=1e-10)


def test_truncation_limits_bond_dimension():
    cores = circuit_to_tt(random_theta(8, 5, 3), 8, 5, chi_max=4)
    assert max(core.shape[2] for core in cores) <= 4


def test_compiled_gates_encode_the_diagonal():
    for n in (3, 4):
        theta = random_theta(n, 5, 10 + n)
        psi = reference_state(theta, n, 5)
        ops, scale = tn_compile(circuit_to_tt(theta, n, 5, chi_max=2 ** (n // 2)), n_bond=max(1, n // 2))
        U = ops_to_unitary(ops, n, max(1, n // 2))
        assert np.allclose(U.conj().T @ U, np.eye(U.shape[0]), atol=1e-9)
        assert np.allclose(U[:2 ** n, :2 ** n] * scale, np.diag(psi), atol=1e-9)


def test_scale_is_a_valid_subnormalization():
    for n in (3, 4, 5):
        theta = random_theta(n, 5, 20 + n)
        _, scale = tn_compile(circuit_to_tt(theta, n, 5, chi_max=2 ** n), n_bond=n)
        assert scale >= np.max(np.abs(reference_state(theta, n, 5)))
