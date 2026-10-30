import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
import sys
import os

# Add root directory to module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.qcfd.burgers import setup_classical_burgers
from src.qcfd.block_encoding import block_encode
from src.qcfd.circuit_vqa import normalization_factor, fidelity_eq4, z_from_probabilities
from src.qcfd.tensor_network import circuit_to_tt, tn_compile, tt_to_vector

# Parameters
N, nu, L_dom, T, SHOTS = 8, 0.1, 1.0, 1.0, None
CHI_MAX = None
x, dx, dt, n_steps, D1, D2, classical_solution = setup_classical_burgers(N, L_dom, nu, T)

n_data = int(np.log2(N))
n_layers = max(5, int(np.ceil(N / n_data)))
chi_max = CHI_MAX or 2 ** (n_data // 2)
n_bond = max(1, int(np.ceil(np.log2(chi_max))))

# Qubit layout: control, LCU, data, A1, A2 (A2 is a list of qubits)
n_aux2 = n_bond + n_data
n_total = n_data + 3 + n_aux2
dim_total = 2 ** n_total
C, Lq = 0, 1
DATA = list(range(2, 2 + n_data))
A1 = 2 + n_data
A2 = list(range(3 + n_data, 3 + n_data + n_aux2))

# Ansatz Circuit
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

# Initial state fit
u_initial = classical_solution[0]
norm_j = np.linalg.norm(u_initial)
u_target = u_initial / norm_j

def fit_cost(params):
    sv = Statevector(qc.assign_parameters({theta: params})).data.real
    return 1.0 - u_target @ sv

best = None
for seed in range(10):
    start = np.random.default_rng(seed).uniform(0, 2 * np.pi, qc.num_parameters)
    attempt = minimize(fit_cost, start, method="L-BFGS-B", options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 2000})
    if best is None or attempt.fun < best.fun:
        best = attempt
    if best.fun < 1e-9:
        break

theta_j = best.x

# Block Encodings of the time-independent operators
O_L = np.eye(N) + dt * nu * D2
O_C = -dt * D1
U_L, c_L = block_encode(O_L, "U_L")
U_C, c_C = block_encode(O_C, "U_C")

theta_p = ParameterVector("θ'", qc.num_parameters)
phi_p = Parameter("φ")

idx = np.arange(dim_total)
keep = (((idx >> Lq) & 1) == 0) & (((idx >> A1) & 1) == 0)
for q in A2:
    keep &= ((idx >> q) & 1) == 0
cbit = (idx >> C) & 1
branch1_idx = (1 << C) + (np.arange(N) << DATA[0])

rng = np.random.default_rng(0)

def ansatz_gate(params, label, inverse=False):
    circuit = qc.assign_parameters(dict(zip(theta, params)))
    return (circuit.inverse() if inverse else circuit).to_gate(label=label)

def build_suffix(th_next, phi):
    s = QuantumCircuit(n_total)
    s.append(ansatz_gate(th_next, "U†(θ_j+1)", inverse=True).control(1), [C] + DATA)
    s.ry(-phi, C)
    return s

def encode_field(th_j):
    cores = circuit_to_tt(th_j, n_data, n_layers, chi_max)
    ops, c_D_j = tn_compile(cores, n_bond)
    return ops, c_D_j, tt_to_vector(cores)

def add_field_encoding(circuit, encoding):
    bond = A2[:n_bond]
    for k, i, matrix in encoding:
        gate = UnitaryGate(matrix).control(3, ctrl_state=1 | (i << 2))
        circuit.append(gate, [C, Lq, DATA[k]] + bond + [A2[n_bond + k]])

def build_prefix(th_j, field_encoding, omega_j):
    c = QuantumCircuit(n_total)
    c.h(C)
    c.append(ansatz_gate(th_j, "U(θ_j)").control(1), [C] + DATA)
    c.cry(omega_j, C, Lq)
    c.append(U_L.control(2, ctrl_state=0b11), [C, Lq] + DATA + [A1])
    c.append(U_C.control(2, ctrl_state=0b01), [C, Lq] + DATA + [A1])
    add_field_encoding(c, field_encoding)
    c.cry(-omega_j, C, Lq)
    return c

suffix_template = build_suffix(theta_p, phi_p)

def sigma_z_of(params, sv_prefix, shots=None):
    th, phi = params[:-1], params[-1]
    bound = suffix_template.assign_parameters({**dict(zip(theta_p, th)), phi_p: phi})
    pr = sv_prefix.evolve(bound).probabilities()
    return z_from_probabilities(pr, keep, cbit, dim_total, shots, rng)

def train_next_step(x0, sv_prefix, shots):
    def cost_fn(params):
        return 1.0 - sigma_z_of(params, sv_prefix, shots)
    if shots is None:
        return minimize(cost_fn, x0, method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-9})
    return minimize(cost_fn, x0, method="COBYLA", options={"maxiter": 2000, "rhobeg": 0.1})

def ansatz_state(params):
    return Statevector(qc.assign_parameters({theta: params})).data.real

def rel_l2_error(approx, reference):
    return np.linalg.norm(approx - reference) / np.linalg.norm(reference)

def quantum_step(th_j, norm_cur, shots=None):
    psi_j = ansatz_state(th_j)
    field_encoding, c_D_j, d_j = encode_field(th_j)
    tt_error = np.max(np.abs(d_j - psi_j))
    if CHI_MAX is None:
        assert tt_error < 1e-8, "tensor train does not reproduce the circuit state"
    beta2_j = c_C * c_D_j * norm_cur
    beta_full_j = c_L + beta2_j
    omega_j = 2 * np.arccos(np.sqrt(beta2_j / beta_full_j))

    O_full_j = O_L - dt * norm_cur * np.diag(d_j) @ D1
    sv_prefix = Statevector(build_prefix(th_j, field_encoding, omega_j))
    target = O_full_j @ psi_j / (beta_full_j * np.sqrt(2))
    assert np.allclose(sv_prefix.data[branch1_idx], target, atol=1e-7), "circuit != O_full psi / beta_full"

    alpha0_j = 2 * sv_prefix.probabilities()[keep & (cbit == 1)].sum()
    x0 = np.append(th_j, 2 * np.arctan(np.sqrt(alpha0_j)))
    result = train_next_step(x0, sv_prefix, shots)
    th_next, phi_j = result.x[:-1], result.x[-1]
    ez_j = sigma_z_of(result.x, sv_prefix, shots)

    alpha_j = np.tan(phi_j / 2) ** 2
    norm_next = beta_full_j * normalization_factor(alpha_j, phi_j) * norm_cur
    psi_exact_j = O_full_j @ psi_j
    alpha_exact_j = np.linalg.norm(psi_exact_j) ** 2 / beta_full_j ** 2
    psi_exact_j /= np.linalg.norm(psi_exact_j)

    info = dict(ez=ez_j, alpha=alpha_j, alpha_exact=alpha_exact_j,
                F_eq4=fidelity_eq4(ez_j, alpha_j, phi_j),
                F_true=(ansatz_state(th_next) @ psi_exact_j) ** 2,
                beta_full=beta_full_j, c_ratio=c_D_j / np.max(np.abs(psi_j)), tt_error=tt_error)
    return th_next, norm_next, info

# Time loop execution
theta_cur, norm_cur = theta_j.copy(), norm_j
u_quantum_hist = [norm_cur * ansatz_state(theta_cur)]
infos = []

mode = "tensor-network"
print(f"\nRunning multi-step quantum simulation ({mode} U_phi, {n_total} qubits)...")
for j in range(n_steps):
    theta_cur, norm_cur, info = quantum_step(theta_cur, norm_cur, shots=SHOTS)
    u_quantum_hist.append(norm_cur * ansatz_state(theta_cur))
    infos.append(info)
    err = rel_l2_error(u_quantum_hist[-1], classical_solution[j + 1])
    print(f"step {j+1:3d}  t={dt*(j+1):.3f}  <Z>={info['ez']:.5f}  alpha(φ)={info['alpha']:.4f}  "
          f"c/max|psi|={info['c_ratio']:.3f}  tt_err={info['tt_error']:.1e}  rel.L2 err={err:.2e}")

u_quantum_hist = np.array(u_quantum_hist)

# Plotting
times_to_plot = T * np.array([0.25, 0.5, 0.75, 1.0])
fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
for ax, t in zip(axes.ravel(), times_to_plot):
    k = int(round(t / dt))
    ax.plot(x, classical_solution[k], "o-", color="#1f77b4", lw=2, label="Classical")
    ax.plot(x, u_quantum_hist[k], "s--", color="#d62728", lw=2, label="Quantum VQA")
    ax.set_title(f"t = {t:.2f}   (rel. L2 error {rel_l2_error(u_quantum_hist[k], classical_solution[k]):.1e})")
    ax.grid(alpha=0.5)
    ax.legend()
for ax in axes[1]:
    ax.set_xlabel("x")
for ax in axes[:, 0]:
    ax.set_ylabel("u(x,t)")
plt.suptitle(f"1D Burgers, N={N}: classical vs quantum VQA time stepping ({mode})")
plt.tight_layout()
figure_path = "results/figures/burgers_compare_times_tn.png"
plt.savefig(figure_path, dpi=130)
plt.show()
print(f"Simulation complete! Plot saved to {figure_path}")
