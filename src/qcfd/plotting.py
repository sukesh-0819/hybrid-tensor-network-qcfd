import matplotlib.pyplot as plt
import numpy as np

from qcfd.metrics import rel_l2_error


def plot_time_comparison(result, path, fractions=(0.25, 0.5, 0.75, 1.0)):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, fraction in zip(axes.ravel(), fractions):
        k = int(round(fraction * result["n_steps"]))
        t = k * result["dt"]
        error = rel_l2_error(result["quantum"][k], result["classical"][k])
        ax.plot(result["x"], result["classical"][k], "o-", color="#1f77b4", lw=2, label="Classical")
        ax.plot(result["x"], result["quantum"][k], "s--", color="#d62728", lw=2, label="Quantum VQA")
        ax.set_title(f"t = {t:.2f}   (rel. L2 error {error:.1e})")
        ax.grid(alpha=0.5)
        ax.legend()
    for ax in axes[1]:
        ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("u(x,t)")
    fig.suptitle(f"1D Burgers, N={result['N']}: classical vs quantum VQA time stepping")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_error_alpha(result, path):
    t = result["dt"] * np.arange(1, result["n_steps"] + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogy(t, result["rel_errors"], "o-")
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("relative L2 error")
    ax[0].grid(alpha=0.5)
    ax[1].plot(t, [i["alpha"] for i in result["infos"]], "o-", label="α_succ from trained φ")
    ax[1].plot(t, [i["alpha_exact"] for i in result["infos"]], "x--", label="exact α")
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("success probability")
    ax[1].grid(alpha=0.5)
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
