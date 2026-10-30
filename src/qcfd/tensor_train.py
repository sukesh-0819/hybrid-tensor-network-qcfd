import numpy as np


def ry_matrix(angle):
    c, s = np.cos(angle / 2), np.sin(angle / 2)
    return np.array([[c, -s], [s, c]])


def dense_ansatz_state(theta, n, n_layers):
    T = np.zeros((2,) * n)
    T[(0,) * n] = 1.0
    p = 0
    for layer in range(n_layers + 1):
        for q in range(n):
            T = np.moveaxis(np.tensordot(ry_matrix(theta[p]), T, axes=([1], [q])), 0, q)
            p += 1
        if layer < n_layers:
            for q in range(n - 1):
                index = [slice(None)] * n
                index[q] = 1
                T[tuple(index)] = np.flip(T[tuple(index)], axis=q)
    return T.transpose(list(range(n))[::-1]).reshape(-1)


def mps_zero(n):
    return [np.array([1.0, 0.0]).reshape(1, 2, 1) for _ in range(n)]


def mps_apply_1q(cores, q, gate):
    cores[q] = np.einsum("ij,ajb->aib", gate, cores[q])


def mps_move_center_left(cores):
    for k in range(len(cores) - 1, 0, -1):
        a, _, b = cores[k].shape
        Q, R = np.linalg.qr(cores[k].reshape(a, 2 * b).T)
        cores[k] = Q.T.reshape(Q.shape[1], 2, b)
        cores[k - 1] = np.einsum("aib,bc->aic", cores[k - 1], R.T)


def mps_apply_cx(cores, q, chi_max, tol=1e-14):
    pair = np.einsum("aib,bjc->aijc", cores[q], cores[q + 1])
    new = pair.copy()
    new[:, 1, :, :] = pair[:, 1, ::-1, :]
    a, _, _, c = new.shape
    U, S, Vh = np.linalg.svd(new.reshape(a * 2, 2 * c), full_matrices=False)
    keep = max(1, min(chi_max, int(np.sum(S > tol * S[0]))))
    U, S, Vh = U[:, :keep], S[:keep], Vh[:keep]
    cores[q] = U.reshape(a, 2, keep)
    cores[q + 1] = (np.diag(S) @ Vh).reshape(keep, 2, c)


def simulate_ansatz_mps(theta, n, n_layers, chi_max):
    cores = mps_zero(n)
    p = 0
    for layer in range(n_layers + 1):
        for q in range(n):
            mps_apply_1q(cores, q, ry_matrix(theta[p]))
            p += 1
        if layer < n_layers:
            mps_move_center_left(cores)
            for q in range(n - 1):
                mps_apply_cx(cores, q, chi_max)
    return cores


def tt_to_vector(cores):
    T = cores[0]
    for core in cores[1:]:
        T = np.tensordot(T, core, axes=([-1], [0]))
    T = T.reshape(T.shape[1:-1])
    return T.transpose(list(range(len(cores)))[::-1]).reshape(-1)


def vector_to_tt(vec, n, chi_max=None, tol=1e-14):
    rest = vec.reshape((2,) * n).transpose(list(range(n))[::-1]).reshape(1, -1)
    cores, r_prev = [], 1
    for k in range(n - 1):
        rest = rest.reshape(r_prev * 2, -1)
        U, S, Vh = np.linalg.svd(rest, full_matrices=False)
        keep = max(1, int(np.sum(S > tol * S[0])))
        if chi_max is not None:
            keep = min(keep, chi_max)
        U, S, Vh = U[:, :keep], S[:keep], Vh[:keep]
        cores.append(U.reshape(r_prev, 2, keep))
        rest = np.diag(S) @ Vh
        r_prev = keep
    cores.append(rest.reshape(r_prev, 2, 1))
    return cores


def tt_bond_dims(cores):
    return [core.shape[2] for core in cores[:-1]]


def tt_mixed_canonical(cores, center):
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


def tt_to_diagonal_tto(cores):
    W = []
    for core in cores:
        a, _, b = core.shape
        w = np.zeros((a, 2, 2, b))
        w[:, 0, 0, :] = core[:, 0, :]
        w[:, 1, 1, :] = core[:, 1, :]
        W.append(w)
    return W


def tto_to_dense(W):
    n = len(W)
    T = W[0]
    for w in W[1:]:
        T = np.tensordot(T, w, axes=([-1], [0]))
    T = T.reshape(T.shape[1:-1])
    out_axes = list(range(0, 2 * n, 2))[::-1]
    in_axes = list(range(1, 2 * n, 2))[::-1]
    return T.transpose(out_axes + in_axes).reshape(2 ** n, 2 ** n)
