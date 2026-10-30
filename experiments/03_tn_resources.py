from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qcfd.metrics import make_field
from qcfd.tensor_train import tt_bond_dims, vector_to_tt
from qcfd.tn_compile import best_canonical_center, required_bond_qubits
from qcfd.tensor_train import tt_to_diagonal_tto

ROOT = Path(__file__).resolve().parents[1]
FIG, DATA = ROOT / "results" / "figures", ROOT / "results" / "data"

kinds = ["uniform", "sin(pi x)", "shock", "uniform random", "gaussian random"]
n_list = np.arange(3, 12)
rng = np.random.default_rng(5)
ratio, chi = {k: [] for k in kinds}, {k: [] for k in kinds}

for kind in kinds:
    for n_q in n_list:
        psi = make_field(kind, 2 ** n_q, rng)
        psi = psi / np.linalg.norm(psi)
        cores, _, scale = best_canonical_center(vector_to_tt(psi, n_q, tol=1e-10))
        ratio[kind].append(scale / np.max(np.abs(psi)))
        chi[kind].append(max(tt_bond_dims(cores)))

columns, header = [2 ** n_list], ["N"]
for kind in kinds:
    columns += [np.array(ratio[kind]), np.array(chi[kind])]
    header += [f"{kind}:c_ratio", f"{kind}:chi"]
np.savetxt(DATA / "tn_scale_and_bond.csv", np.column_stack(columns), delimiter=",", header=",".join(header), comments="")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for kind in kinds:
    ax[0].plot(2 ** n_list, ratio[kind], "o-", label=kind)
    ax[1].plot(2 ** n_list, chi[kind], "o-", label=kind)
for a in ax:
    a.set_xscale("log", base=2)
    a.set_xlabel("N")
    a.grid(alpha=0.5)
ax[0].set_ylabel("c_TT / max|ψ|")
ax[1].set_ylabel("max bond dimension")
ax[0].legend()
fig.tight_layout()
fig.savefig(FIG / "tn_scale_and_bond.png", dpi=150)

rows = []
print(f"{'N':>6} {'chi':>4} {'aux':>4} {'qubits':>7} {'ctrl gates':>10} {'c/max|psi|':>11}")
for n_q in range(3, 13):
    x = np.linspace(0, 1, 2 ** n_q)
    psi = np.sin(np.pi * x)
    psi = psi / np.linalg.norm(psi)
    cores, _, scale = best_canonical_center(vector_to_tt(psi, n_q, tol=1e-10))
    n_bond = required_bond_qubits(tt_to_diagonal_tto(cores))
    rows.append((2 ** n_q, max(tt_bond_dims(cores)), n_bond + n_q, 2 * n_q + n_bond, 2 * n_q,
                 scale / np.max(np.abs(psi))))
    print(f"{rows[-1][0]:>6} {rows[-1][1]:>4} {rows[-1][2]:>4} {rows[-1][3]:>7} {rows[-1][4]:>10} {rows[-1][5]:>11.3f}")
np.savetxt(DATA / "tn_resources_sin.csv", np.array(rows), delimiter=",",
           header="N,chi,aux_qubits,total_qubits,controlled_gates,c_ratio", comments="")
