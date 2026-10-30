import numpy as np


def rel_l2_error(approx, reference):
    return np.linalg.norm(approx - reference) / np.linalg.norm(reference)


def alphas(phi):
    s2 = np.sum(phi ** 2)
    s4 = np.sum(phi ** 4)
    return s4 / s2 ** 2, s4 / (s2 * np.max(phi ** 2))


def make_field(kind, n_grid, generator):
    xs = np.linspace(0, 1, n_grid)
    if kind == "uniform":
        return np.ones(n_grid)
    if kind == "uniform random":
        return generator.uniform(0, 1, n_grid)
    if kind == "gaussian random":
        return generator.standard_normal(n_grid)
    if kind == "sin(pi x)":
        return np.sin(np.pi * xs)
    if kind == "shock":
        return -np.tanh((xs - 0.5) / 0.02)
    if kind == "narrow peak":
        index = np.arange(n_grid)
        return np.exp(-((index - n_grid // 2) / 2.0) ** 2)
    raise ValueError(kind)


def step_alpha(psi, norm_u, dt, nu, D1, D2, c_diag):
    N = len(psi)
    c_L = np.linalg.norm(np.eye(N) + dt * nu * D2, 2) * (1 + 1e-6)
    c_C = np.linalg.norm(dt * D1, 2) * (1 + 1e-6)
    beta_full = c_L + c_C * c_diag * norm_u
    out = psi + dt * nu * (D2 @ psi) - dt * norm_u * psi * (D1 @ psi)
    return np.linalg.norm(out) ** 2 / beta_full ** 2


def normalization_factor(alpha, phi):
    return (1 + alpha) / (2 * np.sin(phi) - (np.sqrt(alpha) - 1 / np.sqrt(alpha)) * np.cos(phi))


def fidelity_eq4(ez, alpha, phi):
    return (alpha * (ez + np.cos(phi)) + ez - np.cos(phi)) ** 2 / (4 * alpha * np.sin(phi) ** 2)


def circuit_resources(circuit, basis_gates=("cx", "u"), optimization_level=1):
    from qiskit import transpile
    compiled = transpile(circuit, basis_gates=list(basis_gates), optimization_level=optimization_level)
    counts = dict(compiled.count_ops())
    return dict(depth=compiled.depth(), two_qubit_gates=counts.get("cx", 0),
                total_gates=sum(counts.values()), ops=counts)
