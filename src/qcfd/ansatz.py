import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize


def default_n_layers(N, n_data):
    return max(5, int(np.ceil(N / n_data)))


def build_ansatz(n_data, n_layers):
    theta = ParameterVector("θ", n_data * (n_layers + 1))
    circuit = QuantumCircuit(n_data)
    p = 0
    for layer in range(n_layers + 1):
        for q in range(n_data):
            circuit.ry(theta[p], q)
            p += 1
        if layer < n_layers:
            for q in range(n_data - 1):
                circuit.cx(q, q + 1)
    return circuit, theta


def ansatz_state(circuit, theta, params):
    return Statevector(circuit.assign_parameters({theta: params})).data.real


def ansatz_gate(circuit, theta, params, label, inverse=False):
    bound = circuit.assign_parameters(dict(zip(theta, params)))
    return (bound.inverse() if inverse else bound).to_gate(label=label)


def fit_initial_state(circuit, theta, u_target, restarts=10, tol=1e-9):
    def cost(params):
        return 1.0 - u_target @ ansatz_state(circuit, theta, params)

    best = None
    for seed in range(restarts):
        start = np.random.default_rng(seed).uniform(0, 2 * np.pi, circuit.num_parameters)
        attempt = minimize(cost, start, method="L-BFGS-B",
                           options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 2000})
        if best is None or attempt.fun < best.fun:
            best = attempt
        if best.fun < tol:
            break
    return best.x, float(u_target @ ansatz_state(circuit, theta, best.x))
