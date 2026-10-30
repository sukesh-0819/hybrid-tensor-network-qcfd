from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qcfd.metrics import step_alpha
from qcfd.tensor_train import vector_to_tt
from qcfd.tn_compile import best_canonical_center

ROOT = Path(__file__).resolve().parents[1]
FIG, DATA = ROOT / "results" / "figures", ROOT / "results" / "data"


def paper_regime_setup(n_q, nu=1e-3, cfl=0.5):
    N = 2 ** n_q
    x = np.linspace(0, 1, N)
    dx = x[1] - x[0]
    D1 = (np.diag(np.ones(N - 1), 1) - np.diag(np.ones(N - 1), -1)) / (2 * dx)
    D2 = (np.diag(np.ones(N - 1), 1) + np.diag(np.ones(N - 1), -1) - 2 * np.eye(N)) / dx ** 2
    for D in (D1, D2):
        D[0], D[-1] = 0.0, 0.0
    return x, cfl * dx, nu, D1, D2


n_list = np.arange(3, 11)
rows = []
for n_q in n_list:
    x, dt, nu, D1, D2 = paper_regime_setup(n_q)
    u0 = np.sin(np.pi * x)
    u0[0] = u0[-1] = 0.0
    norm_u = np.linalg.norm(u0)
    psi = u0 / norm_u
    _, _, c_tn = best_canonical_center(vector_to_tt(psi, n_q, tol=1e-10))
    rows.append((2 ** n_q,
                 step_alpha(psi, norm_u, dt, nu, D1, D2, 1.0),
                 step_alpha(psi, norm_u, dt, nu, D1, D2, np.max(np.abs(psi)) * (1 + 1e-6)),
                 step_alpha(psi, norm_u, dt, nu, D1, D2, c_tn)))
rows = np.array(rows)
np.savetxt(DATA / "step_alpha_vs_N.csv", rows, delimiter=",",
           header="N,state_based,block_diagonal_ideal,block_diagonal_tt", comments="")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.loglog(rows[:, 0], rows[:, 1], "p--", label="state-based")
ax.loglog(rows[:, 0], rows[:, 2], "*-", label="block-diagonal, ideal c")
ax.loglog(rows[:, 0], rows[:, 3], "o-", label="block-diagonal, TT-compiled c")
ax.set_xlabel("N")
ax.set_ylabel("α_succ of one Euler step")
ax.grid(alpha=0.4, which="both")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "step_alpha_vs_N.png", dpi=150)
for r in rows:
    print(f"N={int(r[0]):5d}  state={r[1]:.3e}  ideal={r[2]:.4f}  tt={r[3]:.4f}")
