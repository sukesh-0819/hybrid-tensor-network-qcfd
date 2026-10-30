import numpy as np
from qiskit import QuantumCircuit

from qcfd.ansatz import ansatz_gate


class Layout:
    def __init__(self, n_data, n_a2=1):
        self.n_data = n_data
        self.n_a2 = n_a2
        self.control = 0
        self.lcu = 1
        self.data = list(range(2, 2 + n_data))
        self.a1 = 2 + n_data
        self.a2 = list(range(3 + n_data, 3 + n_data + n_a2))
        self.n_total = 3 + n_data + n_a2
        self.dim = 2 ** self.n_total
        idx = np.arange(self.dim)
        keep = (((idx >> self.lcu) & 1) == 0) & (((idx >> self.a1) & 1) == 0)
        for q in self.a2:
            keep &= ((idx >> q) & 1) == 0
        self.keep = keep
        self.cbit = (idx >> self.control) & 1
        self.success1 = self.keep & (self.cbit == 1)
        self.branch1 = (1 << self.control) + (np.arange(2 ** n_data) << self.data[0])


def lcu_weights(c_L, c_C, c_D, norm_cur):
    beta2 = c_C * c_D * norm_cur
    beta_full = c_L + beta2
    omega = 2 * np.arccos(np.sqrt(beta2 / beta_full))
    return beta2, beta_full, omega


def build_prefix(layout, ansatz, theta, th_j, U_L, U_C, U_phi, omega):
    lay = layout
    c = QuantumCircuit(lay.n_total)
    c.h(lay.control)
    c.append(ansatz_gate(ansatz, theta, th_j, "U(θ_j)").control(1), [lay.control] + lay.data)
    c.cry(omega, lay.control, lay.lcu)
    c.append(U_L.control(2, ctrl_state=0b11), [lay.control, lay.lcu] + lay.data + [lay.a1])
    c.append(U_C.control(2, ctrl_state=0b01), [lay.control, lay.lcu] + lay.data + [lay.a1])
    c.append(U_phi.control(2, ctrl_state=0b01), [lay.control, lay.lcu] + lay.data + lay.a2)
    c.cry(-omega, lay.control, lay.lcu)
    return c


def build_suffix(layout, ansatz, theta, th_next, phi):
    lay = layout
    s = QuantumCircuit(lay.n_total)
    s.append(ansatz_gate(ansatz, theta, th_next, "U†(θ_j+1)", inverse=True).control(1),
             [lay.control] + lay.data)
    s.ry(-phi, lay.control)
    return s


def z_from_probabilities(layout, pr):
    n0 = pr[layout.keep & (layout.cbit == 0)].sum()
    n1 = pr[layout.keep & (layout.cbit == 1)].sum()
    return (n0 - n1) / (n0 + n1)


def check_prefix(layout, sv_prefix, target, atol=1e-7):
    if not np.allclose(sv_prefix.data[layout.branch1], target, atol=atol):
        raise AssertionError("circuit does not reproduce O_full psi / beta_full")
