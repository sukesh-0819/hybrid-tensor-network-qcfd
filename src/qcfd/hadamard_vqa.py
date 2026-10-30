import numpy as np
from qiskit.circuit import Parameter, ParameterVector
from scipy.optimize import minimize

from qcfd.lcu_step import build_suffix, z_from_probabilities


class HadamardTest:
    def __init__(self, layout, ansatz, theta, shots=None, seed=0):
        self.layout = layout
        self.shots = shots
        self.rng = np.random.default_rng(seed)
        self.theta_p = ParameterVector("θ'", ansatz.num_parameters)
        self.phi_p = Parameter("φ")
        self.suffix_template = build_suffix(layout, ansatz, theta, self.theta_p, self.phi_p)

    def sigma_z(self, params, sv_prefix):
        th, phi = params[:-1], params[-1]
        bound = self.suffix_template.assign_parameters({**dict(zip(self.theta_p, th)), self.phi_p: phi})
        pr = sv_prefix.evolve(bound).probabilities()
        lay = self.layout
        if self.shots is None:
            return z_from_probabilities(lay, pr)
        out = self.rng.choice(lay.dim, size=self.shots, p=pr)
        ok = lay.keep[out]
        n0 = np.sum(ok & (lay.cbit[out] == 0))
        n1 = np.sum(ok & (lay.cbit[out] == 1))
        return 0.0 if n0 + n1 == 0 else (n0 - n1) / (n0 + n1)

    def train(self, x0, sv_prefix):
        def cost_fn(params):
            return 1.0 - self.sigma_z(params, sv_prefix)

        if self.shots is None:
            return minimize(cost_fn, x0, method="L-BFGS-B",
                            options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-9})
        return minimize(cost_fn, x0, method="COBYLA",
                        options={"maxiter": 2000, "rhobeg": 0.1})


def success_probability_estimate(layout, sv_prefix):
    return 2 * sv_prefix.probabilities()[layout.success1].sum()
