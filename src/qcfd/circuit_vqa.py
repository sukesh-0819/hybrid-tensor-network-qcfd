import numpy as np

def normalization_factor(alpha, phi):
    return (1 + alpha) / (2 * np.sin(phi) - (np.sqrt(alpha) - 1 / np.sqrt(alpha)) * np.cos(phi))

def fidelity_eq4(ez, alpha, phi):
    return (alpha * (ez + np.cos(phi)) + ez - np.cos(phi)) ** 2 / (4 * alpha * np.sin(phi) ** 2)

def z_from_probabilities(pr, keep, cbit, dim_total, shots, rng=None):
    if shots is None:
        n0 = pr[keep & (cbit == 0)].sum()
        n1 = pr[keep & (cbit == 1)].sum()
        return (n0 - n1) / (n0 + n1)
    
    out = rng.choice(dim_total, size=shots, p=pr)
    ok = keep[out]
    n0 = np.sum(ok & (cbit[out] == 0))
    n1 = np.sum(ok & (cbit[out] == 1))
    return 0.0 if n0 + n1 == 0 else (n0 - n1) / (n0 + n1)