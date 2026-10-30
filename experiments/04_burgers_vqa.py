import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

from qcfd.evolve import BurgersVQA
from qcfd.plotting import plot_error_alpha, plot_time_comparison

ROOT = Path(__file__).resolve().parents[1]
FIG, DATA = ROOT / "results" / "figures", ROOT / "results" / "data"

parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, default=8)
parser.add_argument("--T", type=float, default=1.0)
parser.add_argument("--encoding", choices=["dense", "tn"], default="dense")
parser.add_argument("--chi-max", type=int, default=None)
parser.add_argument("--draw", action="store_true")
args = parser.parse_args()

solver = BurgersVQA(args.N, T=args.T, encoding=args.encoding, chi_max=args.chi_max)
result = solver.run()
tag = f"N{args.N}_{args.encoding}"

plot_time_comparison(result, FIG / f"burgers_time_comparison_{tag}.png")
plot_error_alpha(result, FIG / f"burgers_error_alpha_{tag}.png")

steps = np.arange(1, result["n_steps"] + 1)
table = np.column_stack([steps, steps * result["dt"], result["rel_errors"],
                         [i["alpha"] for i in result["infos"]], [i["alpha_exact"] for i in result["infos"]],
                         [i["ez"] for i in result["infos"]], [i["F_eq4"] for i in result["infos"]],
                         [i["F_true"] for i in result["infos"]]])
np.savetxt(DATA / f"burgers_vqa_{tag}.csv", table, delimiter=",",
           header="step,t,rel_l2_error,alpha_from_phi,alpha_exact,sigma_z,F_eq4,F_true", comments="")

if args.draw:
    th_j, norm_j = result["step_inputs"][-1]
    circuit = solver.full_step_circuit(th_j, norm_j, result["theta_final"], result["infos"][-1]["phi"])
    circuit.draw("mpl", style="clifford", scale=0.8, fold=-1).savefig(FIG / f"full_circuit_{tag}.png", dpi=200,
                                                                     bbox_inches="tight")
print(f"final relative L2 error at t={args.T}: {result['rel_errors'][-1]:.3e}")
