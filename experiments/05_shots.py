import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qcfd.evolve import BurgersVQA

ROOT = Path(__file__).resolve().parents[1]
FIG, DATA = ROOT / "results" / "figures", ROOT / "results" / "data"

parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, default=8)
parser.add_argument("--T", type=float, default=0.2)
parser.add_argument("--shots", type=int, nargs="+", default=[2000, 20000, 200000])
parser.add_argument("--seeds", type=int, default=3)
args = parser.parse_args()

rows = []
for shots in args.shots:
    errors = []
    for seed in range(args.seeds):
        result = BurgersVQA(args.N, T=args.T, shots=shots, seed=seed).run(verbose=False)
        errors.append(result["rel_errors"][-1])
    rows.append((shots, np.mean(errors), np.std(errors)))
    print(f"shots={shots:8d}  final error {rows[-1][1]:.3e} +- {rows[-1][2]:.1e}")

rows = np.array(rows)
np.savetxt(DATA / f"shots_N{args.N}.csv", rows, delimiter=",", header="shots,mean_rel_l2_error,std", comments="")
fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(rows[:, 0], rows[:, 1], yerr=rows[:, 2], fmt="o-")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("shots per expectation value")
ax.set_ylabel(f"relative L2 error at t={args.T}")
ax.grid(alpha=0.5, which="both")
fig.tight_layout()
fig.savefig(FIG / f"shots_N{args.N}.png", dpi=150)
