import numpy as np


def ry_matrix(angle):
    c, s = np.cos(angle / 2), np.sin(angle / 2)
    return np.array([[c, -s], [s, c]])


def to_canonical(cores, center):
    cores = [c.copy() for c in cores]
    for k in range(center):
        a, _, b = cores[k].shape
        Q, R = np.linalg.qr(cores[k].reshape(a * 2, b))
        cores[k] = Q.reshape(a, 2, -1)
        cores[k + 1] = np.einsum("ab,bic->aic", R, cores[k + 1])
    for k in range(len(cores) - 1, center, -1):
        a, _, b = cores[k].shape
        Q, R = np.linalg.qr(cores[k].reshape(a, 2 * b).T)
        cores[k] = Q.T.reshape(-1, 2, b)
        cores[k - 1] = np.einsum("aib,bc->aic", cores[k - 1], R.T)
    return cores


def apply_cx(cores, q, chi_max):
    pair = np.einsum("aib,bjc->aijc", cores[q], cores[q + 1])
    new = pair.copy()
    new[:, 1, :, :] = pair[:, 1, ::-1, :]
    a, _, _, c = new.shape
    U, S, Vh = np.linalg.svd(new.reshape(a * 2, 2 * c), full_matrices=False)
    keep = max(1, min(chi_max, int(np.sum(S > 1e-14 * S[0]))))
    cores[q] = U[:, :keep].reshape(a, 2, keep)
    cores[q + 1] = (np.diag(S[:keep]) @ Vh[:keep]).reshape(keep, 2, c)


def circuit_to_tt(theta, n, n_layers, chi_max):
    cores = [np.array([1.0, 0.0]).reshape(1, 2, 1) for _ in range(n)]
    p = 0
    for layer in range(n_layers + 1):
        for q in range(n):
            cores[q] = np.einsum("ij,ajb->aib", ry_matrix(theta[p]), cores[q])
            p += 1
        if layer < n_layers:
            cores = to_canonical(cores, 0)
            for q in range(n - 1):
                apply_cx(cores, q, chi_max)
    return cores


def tt_to_vector(cores):
    T = cores[0]
    for core in cores[1:]:
        T = np.tensordot(T, core, axes=([-1], [0]))
    T = T.reshape(T.shape[1:-1])
    return T.transpose(list(range(len(cores)))[::-1]).reshape(-1)


def tt_to_diagonal_operator(cores):
    operator = []
    for core in cores:
        a, _, b = core.shape
        w = np.zeros((a, 2, 2, b))
        w[:, 0, 0, :] = core[:, 0, :]
        w[:, 1, 1, :] = core[:, 1, :]
        operator.append(w)
    return operator


def site_scale(w):
    return max(np.linalg.norm(w[:, i, i, :], 2) for i in range(2)) * (1 + 1e-6)


def choose_gauge(cores):
    best = None
    for center in range(len(cores)):
        trial = to_canonical(cores, center)
        scale = np.prod([site_scale(w) for w in tt_to_diagonal_operator(trial)])
        if best is None or scale < best[0]:
            best = (scale, trial)
    return best[1]


def sqrt_psd(H):
    values, vectors = np.linalg.eigh((H + H.conj().T) / 2)
    return (vectors * np.sqrt(np.clip(values, 0, None))) @ vectors.conj().T


def dilate(A):
    I = np.eye(A.shape[0])
    return np.block([[A, sqrt_psd(I - A @ A.conj().T)],
                     [sqrt_psd(I - A.conj().T @ A), -A.conj().T]])


def tn_compile(cores, n_bond):
    operator = tt_to_diagonal_operator(choose_gauge(cores))
    max_bond = max(max(w.shape[0], w.shape[3]) for w in operator)
    if 2 ** n_bond < max_bond:
        raise ValueError(f"bond register of {n_bond} qubits is too small for bond dimension {max_bond}")
    size = 2 ** n_bond
    ops, scale = [], 1.0
    for k, w in enumerate(operator):
        c_k = site_scale(w) or 1.0
        scale *= c_k
        for i in range(2):
            A = np.zeros((size, size))
            block = w[:, i, i, :].T / c_k
            A[:block.shape[0], :block.shape[1]] = block
            ops.append((k, i, dilate(A)))
    return ops, scale
