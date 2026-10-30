import numpy as np

from qcfd.block_encoding import matrix_sqrt_psd
from qcfd.tensor_train import tt_mixed_canonical, tt_to_diagonal_tto


def dilate(A):
    I = np.eye(A.shape[0])
    return np.block([[A, matrix_sqrt_psd(I - A @ A.conj().T)],
                     [matrix_sqrt_psd(I - A.conj().T @ A), -A.conj().T]])


def site_scale(w):
    return max(np.linalg.norm(w[:, i, i, :], 2) for i in range(2)) * (1 + 1e-6)


def diagonal_tto_scale(W):
    return float(np.prod([site_scale(w) for w in W]))


def best_canonical_center(cores):
    best = None
    for center in range(len(cores)):
        trial = tt_mixed_canonical(cores, center)
        scale = diagonal_tto_scale(tt_to_diagonal_tto(trial))
        if best is None or scale < best[0]:
            best = (scale, center, trial)
    return best[2], best[1], best[0]


def required_bond_qubits(W):
    max_bond = max(max(w.shape[0], w.shape[3]) for w in W)
    return max(1, int(np.ceil(np.log2(max_bond))))


def compile_diagonal_tto(W, n_bond=None):
    n = len(W)
    needed = required_bond_qubits(W)
    n_bond = needed if n_bond is None else n_bond
    if n_bond < needed:
        raise ValueError(f"bond register of {n_bond} qubits is too small, need {needed}")
    bond_qubits = list(range(n, n + n_bond))
    flag_qubits = list(range(n + n_bond, 2 * n + n_bond))
    dim_bond = 2 ** n_bond
    ops, scale = [], 1.0
    for k, w in enumerate(W):
        c_k = site_scale(w)
        if c_k == 0:
            c_k = 1.0
        scale *= c_k
        for i in range(2):
            A = np.zeros((dim_bond, dim_bond))
            block = w[:, i, i, :].T / c_k
            A[:block.shape[0], :block.shape[1]] = block
            ops.append(dict(matrix=dilate(A), targets=bond_qubits + [flag_qubits[k]],
                            control=k, ctrl_state=i))
    return dict(ops=ops, n_data=n, n_aux=n_bond + n, n_bond=n_bond, n_qubits=2 * n + n_bond,
                scale=scale, bond_qubits=bond_qubits, flag_qubits=flag_qubits)


def compile_from_cores(cores, n_bond=None):
    cores, _, _ = best_canonical_center(cores)
    return compile_diagonal_tto(tt_to_diagonal_tto(cores), n_bond=n_bond)


def op_to_dense(op, n_qubits):
    dim = 2 ** n_qubits
    F = np.zeros((dim, dim), dtype=complex)
    targets = op["targets"]
    for s in range(dim):
        if ((s >> op["control"]) & 1) != op["ctrl_state"]:
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
            F[sp, s] = op["matrix"][jp, j]
    return F


def compiled_to_dense(compiled):
    U = np.eye(2 ** compiled["n_qubits"], dtype=complex)
    for op in compiled["ops"]:
        U = op_to_dense(op, compiled["n_qubits"]) @ U
    return U


def ops_to_circuit(compiled):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    circuit = QuantumCircuit(compiled["n_qubits"])
    for op in compiled["ops"]:
        gate = UnitaryGate(op["matrix"]).control(1, ctrl_state=op["ctrl_state"])
        circuit.append(gate, [op["control"]] + op["targets"])
    return circuit


def build_U_phi_gate(cores, n_bond=None):
    compiled = compile_from_cores(cores, n_bond=n_bond)
    gate = ops_to_circuit(compiled).to_gate(label="U_phi_TN")
    return gate, compiled["scale"], compiled["n_aux"], compiled
