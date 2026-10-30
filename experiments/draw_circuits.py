import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
import sys
import os

# Add root directory to module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.qcfd.burgers import setup_classical_burgers
from src.qcfd.block_encoding import block_encode
from src.qcfd.tensor_network import circuit_to_tt, tn_compile

N, nu, L_dom, T = 8, 0.1, 1.0, 1.0
x, dx, dt, n_steps, D1, D2, classical_solution = setup_classical_burgers(N, L_dom, nu, T)

n_data = int(np.log2(N))
n_layers = max(5, int(np.ceil(N / n_data)))
chi_max = 2 ** (n_data // 2)
n_bond = max(1, int(np.ceil(np.log2(chi_max))))
C, Lq = 0, 1
DATA = list(range(2, 2 + n_data))
A1 = 2 + n_data

theta = ParameterVector("θ", n_data * (n_layers + 1))
qc = QuantumCircuit(n_data)
p = 0
for layer in range(n_layers + 1):
    for q in range(n_data):
        qc.ry(theta[p], q)
        p += 1
    if layer < n_layers:
        for q in range(n_data - 1):
            qc.cx(q, q + 1)

def ansatz_state(params):
    return Statevector(qc.assign_parameters({theta: params})).data.real

u_initial = classical_solution[0]
norm_j = np.linalg.norm(u_initial)
u_target = u_initial / norm_j

best = None
for seed in range(10):
    start = np.random.default_rng(seed).uniform(0, 2 * np.pi, qc.num_parameters)
    attempt = minimize(lambda params: 1.0 - u_target @ ansatz_state(params), start, method="L-BFGS-B",
                       options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 2000})
    if best is None or attempt.fun < best.fun:
        best = attempt
    if best.fun < 1e-9:
        break
theta_j = best.x
theta_next = theta_j
phi = np.pi / 2

O_L = np.eye(N) + dt * nu * D2
O_C = -dt * D1
U_L, c_L = block_encode(O_L, "U_L")
U_C, c_C = block_encode(O_C, "U_C")

def ansatz_gate(params, label, inverse=False):
    circuit = qc.assign_parameters(dict(zip(theta, params)))
    return (circuit.inverse() if inverse else circuit).to_gate(label=label)

def new_circuit(n_aux2):
    return QuantumCircuit(QuantumRegister(1, "ctrl"), QuantumRegister(1, "lcu"), QuantumRegister(n_data, "data"),
                          QuantumRegister(1, "a1"), QuantumRegister(n_aux2, "a2"))

def build_full_circuit(use_tn):
    n_aux2 = n_bond + n_data if use_tn else 1
    A2 = list(range(3 + n_data, 3 + n_data + n_aux2))
    if use_tn:
        ops, c_D = tn_compile(circuit_to_tt(theta_j, n_data, n_layers, chi_max), n_bond)
    else:
        ops = None
        field_gate, c_D = block_encode(np.diag(ansatz_state(theta_j)), "U_phi")
    beta2 = c_C * c_D * norm_j
    omega = 2 * np.arccos(np.sqrt(beta2 / (c_L + beta2)))

    c = new_circuit(n_aux2)
    c.h(C)
    c.append(ansatz_gate(theta_j, "U(θ_j)").control(1), [C] + DATA)
    c.cry(omega, C, Lq)
    c.append(U_L.control(2, ctrl_state=0b11, label="U_L"), [C, Lq] + DATA + [A1])
    c.append(U_C.control(2, ctrl_state=0b01, label="U_C"), [C, Lq] + DATA + [A1])
    if use_tn:
        for k, i, matrix in ops:
            gate = UnitaryGate(matrix, label=f"S{k}:{i}").control(3, ctrl_state=1 | (i << 2), label=f"S{k}:{i}")
            c.append(gate, [C, Lq, DATA[k]] + A2[:n_bond] + [A2[n_bond + k]])
    else:
        c.append(field_gate.control(2, ctrl_state=0b01, label="U_phi"), [C, Lq] + DATA + A2)
    c.cry(-omega, C, Lq)
    c.append(ansatz_gate(theta_next, "U†(θ_j+1)", inverse=True).control(1), [C] + DATA)
    c.ry(-phi, C)
    return c, ops

def build_tn_block(ops):
    c = QuantumCircuit(QuantumRegister(n_data, "data"), QuantumRegister(n_bond, "bond"), QuantumRegister(n_data, "flag"))
    bond = list(range(n_data, n_data + n_bond))
    for k, i, matrix in ops:
        c.append(UnitaryGate(matrix, label=f"S{k}:{i}").control(1, ctrl_state=i, label=f"S{k}:{i}"),
                 [k] + bond + [n_data + n_bond + k])
    return c

def save_circuit(circuit, name, title):
    figure = circuit.draw("mpl", style="clifford", scale=0.7, fold=-1)
    figure.suptitle(title)
    os.makedirs("results/figures", exist_ok=True)
    figure.savefig(f"results/figures/{name}.png", dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"{name:28s} qubits={circuit.num_qubits:2d}  depth={circuit.depth():3d}  gates={dict(circuit.count_ops())}")

dense_circuit, _ = build_full_circuit(use_tn=False)
tn_circuit, tn_ops = build_full_circuit(use_tn=True)

note = "one time step, angles θ_j+1 and φ are placeholders"
save_circuit(qc, "circuit_ansatz", f"Ansatz circuit U(θ), N={N}")
save_circuit(dense_circuit, "circuit_dense", f"Complete circuit, dense U_phi (run_simulation.py), {note}")
save_circuit(tn_circuit, "circuit_tn", f"Complete circuit, tensor-network U_phi (run_simulation_tn.py), {note}")
save_circuit(build_tn_block(tn_ops), "circuit_tn_field_encoding",
             "Tensor-network field encoding: one controlled gate per site and bit value")
print("Saved the four circuit images in results/figures/")
