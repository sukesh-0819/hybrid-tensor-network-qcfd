from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qcfd.metrics import alphas, make_field

ROOT = Path(__file__).resolve().parents[1]
FIG, DATA = ROOT / "results" / "figures", ROOT / "results" / "data"

kinds = ["uniform", "uniform random", "gaussian random", "sin(pi x)", "shock", "narrow peak"]
qubits = np.arange(3, 21)
sizes = 2 ** qubits
n_trials = 20
rng = np.random.default_rng(0)

results = {}
for kind in kinds:
    state_vals, block_vals = [], []
    for n_grid in sizes:
        trials = n_trials if "random" in kind else 1
        a = np.array([alphas(make_field(kind, n_grid, rng)) for _ in range(trials)])
        state_vals.append(a[:, 0].mean())
        block_vals.append(a[:, 1].mean())
    results[kind] = (np.array(state_vals), np.array(block_vals))

columns, header = [sizes], ["N"]
for kind in kinds:
    columns += [results[kind][0], results[kind][1]]
    header += [f"{kind}:state", f"{kind}:block"]
np.savetxt(DATA / "alpha_vs_N.csv", np.column_stack(columns), delimiter=",", header=",".join(header), comments="")

colors = dict(zip(kinds, plt.cm.tab10.colors))
fig, ax = plt.subplots(figsize=(8, 5.5))
for kind in kinds:
    ax.loglog(sizes, results[kind][1], "*-", color=colors[kind], label=f"{kind}: block-diagonal")
    ax.loglog(sizes, results[kind][0], "p--", color=colors[kind], label=f"{kind}: state-based")
ax.loglog(sizes, 9 / (5 * sizes), "k:", label="theory 9/(5N), uniform random, state-based")
ax.loglog(sizes, 3 / sizes, "k--", label="theory 3/N, gaussian, state-based")
ax.loglog(sizes, 3 / (2 * qubits * np.log(2) + np.log(qubits * np.log(2))), "k-.",
          label="theory gaussian, block-diagonal")
ax.set_xlabel("number of grid points N")
ax.set_ylabel("success probability α_succ")
ax.grid(alpha=0.4, which="both")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(FIG / "alpha_vs_N.png", dpi=150)
print("saved", FIG / "alpha_vs_N.png")
for kind in ("uniform", "uniform random", "sin(pi x)"):
    print(f"{kind:16s} N=8 state={results[kind][0][0]:.3e} block={results[kind][1][0]:.3f}   "
          f"N=2^20 state={results[kind][0][-1]:.3e} block={results[kind][1][-1]:.3f}")
