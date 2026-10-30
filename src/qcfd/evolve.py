import argparse

import numpy as np
from qiskit.quantum_info import Statevector

from qcfd.ansatz import ansatz_state, build_ansatz, default_n_layers, fit_initial_state
from qcfd.block_encoding import block_encode
from qcfd.burgers import classical_solution
from qcfd.hadamard_vqa import HadamardTest, success_probability_estimate
from qcfd.lcu_step import Layout, build_prefix, build_suffix, check_prefix, lcu_weights
from qcfd.metrics import fidelity_eq4, normalization_factor, rel_l2_error
from qcfd.tensor_train import simulate_ansatz_mps, tt_to_vector
from qcfd.tn_compile import build_U_phi_gate


class BurgersVQA:
    def __init__(self, N=8, nu=0.1, T=1.0, shots=None, seed=0, encoding="dense", chi_max=None):
        if N < 4 or N & (N - 1):
            raise ValueError("N must be a power of two, at least 4")
        if encoding not in ("dense", "tn"):
            raise ValueError("encoding must be 'dense' or 'tn'")
        self.N, self.nu, self.T, self.shots, self.encoding = N, nu, T, shots, encoding
        self.n_data = int(np.log2(N))
        problem = classical_solution(N, nu, T)
        self.x, self.dt, self.n_steps = problem["x"], problem["dt"], problem["n_steps"]
        self.D1, self.D2, self.classical = problem["D1"], problem["D2"], problem["solution"]
        self.chi_max = chi_max or 2 ** (self.n_data // 2)
        self.n_bond = max(1, int(np.ceil(np.log2(self.chi_max))))
        n_a2 = 1 if encoding == "dense" else self.n_bond + self.n_data
        self.layout = Layout(self.n_data, n_a2)
        self.n_layers = default_n_layers(N, self.n_data)
        self.ansatz, self.theta = build_ansatz(self.n_data, self.n_layers)
        self.O_L = np.eye(N) + self.dt * nu * self.D2
        self.O_C = -self.dt * self.D1
        self.U_L, self.c_L = block_encode(self.O_L, "U_L")
        self.U_C, self.c_C = block_encode(self.O_C, "U_C")
        self.hadamard = HadamardTest(self.layout, self.ansatz, self.theta, shots=shots, seed=seed)

    def encode_field(self, th_j, psi_j):
        if self.encoding == "dense":
            U_phi, c_D = block_encode(np.diag(psi_j), "U_phi")
            return U_phi, c_D, psi_j
        cores = simulate_ansatz_mps(th_j, self.n_data, self.n_layers, self.chi_max)
        gate, c_D, _, _ = build_U_phi_gate(cores, n_bond=self.n_bond)
        return gate, c_D, tt_to_vector(cores)

    def quantum_step(self, th_j, norm_cur, check=True):
        psi_j = ansatz_state(self.ansatz, self.theta, th_j)
        U_phi_j, c_D_j, d_j = self.encode_field(th_j, psi_j)
        _, beta_full_j, omega_j = lcu_weights(self.c_L, self.c_C, c_D_j, norm_cur)

        O_full_j = self.O_L - self.dt * norm_cur * np.diag(d_j) @ self.D1
        sv_prefix = Statevector(build_prefix(self.layout, self.ansatz, self.theta, th_j,
                                             self.U_L, self.U_C, U_phi_j, omega_j))
        if check:
            check_prefix(self.layout, sv_prefix, O_full_j @ psi_j / (beta_full_j * np.sqrt(2)))

        alpha0_j = success_probability_estimate(self.layout, sv_prefix)
        x0 = np.append(th_j, 2 * np.arctan(np.sqrt(alpha0_j)))
        result = self.hadamard.train(x0, sv_prefix)
        th_next, phi_j = result.x[:-1], result.x[-1]
        ez_j = self.hadamard.sigma_z(result.x, sv_prefix)

        alpha_j = np.tan(phi_j / 2) ** 2
        norm_next = beta_full_j * normalization_factor(alpha_j, phi_j) * norm_cur

        psi_exact_j = O_full_j @ psi_j
        alpha_exact_j = np.linalg.norm(psi_exact_j) ** 2 / beta_full_j ** 2
        psi_exact_j /= np.linalg.norm(psi_exact_j)

        info = dict(ez=ez_j, alpha=alpha_j, alpha_exact=alpha_exact_j, phi=phi_j,
                    F_eq4=fidelity_eq4(ez_j, alpha_j, phi_j),
                    F_true=(ansatz_state(self.ansatz, self.theta, th_next) @ psi_exact_j) ** 2,
                    beta_full=beta_full_j, c_D=c_D_j)
        return th_next, norm_next, info

    def full_step_circuit(self, th_j, norm_cur, th_next, phi):
        psi_j = ansatz_state(self.ansatz, self.theta, th_j)
        U_phi_j, c_D_j, _ = self.encode_field(th_j, psi_j)
        _, _, omega_j = lcu_weights(self.c_L, self.c_C, c_D_j, norm_cur)
        prefix = build_prefix(self.layout, self.ansatz, self.theta, th_j, self.U_L, self.U_C, U_phi_j, omega_j)
        suffix = build_suffix(self.layout, self.ansatz, self.theta, th_next, phi)
        return prefix.compose(suffix)

    def run(self, verbose=True):
        u0 = self.classical[0]
        norm_cur = np.linalg.norm(u0)
        theta_cur, overlap = fit_initial_state(self.ansatz, self.theta, u0 / norm_cur)
        history = [norm_cur * ansatz_state(self.ansatz, self.theta, theta_cur)]
        infos, errors, step_inputs = [], [], []
        if verbose:
            print(f"N={self.N}  qubits={self.layout.n_total}  dt={self.dt:.5f}  steps={self.n_steps}  "
                  f"encoding={self.encoding}  initial overlap={overlap:.10f}")
        for j in range(self.n_steps):
            step_inputs.append((theta_cur.copy(), norm_cur))
            theta_cur, norm_cur, info = self.quantum_step(theta_cur, norm_cur)
            history.append(norm_cur * ansatz_state(self.ansatz, self.theta, theta_cur))
            infos.append(info)
            errors.append(rel_l2_error(history[-1], self.classical[j + 1]))
            if verbose:
                print(f"step {j + 1:3d}  t={self.dt * (j + 1):.3f}  <Z>={info['ez']:.5f}  "
                      f"alpha={info['alpha']:.4f} (exact {info['alpha_exact']:.4f})  "
                      f"F_eq4={info['F_eq4']:.5f}  F_true={info['F_true']:.5f}  err={errors[-1]:.2e}")
        return dict(N=self.N, nu=self.nu, T=self.T, dt=self.dt, n_steps=self.n_steps, x=self.x,
                    classical=self.classical, quantum=np.array(history), infos=infos,
                    rel_errors=np.array(errors), initial_overlap=overlap, theta_final=theta_cur,
                    step_inputs=step_inputs, encoding=self.encoding, shots=self.shots)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=8)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--encoding", choices=["dense", "tn"], default="dense")
    parser.add_argument("--chi-max", type=int, default=None)
    args = parser.parse_args()
    solver = BurgersVQA(args.N, T=args.T, shots=args.shots, encoding=args.encoding, chi_max=args.chi_max)
    result = solver.run()
    print(f"final relative L2 error: {result['rel_errors'][-1]:.3e}")


if __name__ == "__main__":
    main()
