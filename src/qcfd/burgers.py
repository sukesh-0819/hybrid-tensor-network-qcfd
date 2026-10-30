import numpy as np


def derivative_matrices(N, length=1.0):
    x = np.linspace(0, length, N)
    dx = x[1] - x[0]
    D1 = np.zeros((N, N))
    D2 = np.zeros((N, N))
    for i in range(1, N - 1):
        D1[i, i - 1], D1[i, i + 1] = -1 / (2 * dx), 1 / (2 * dx)
        D2[i, i - 1], D2[i, i], D2[i, i + 1] = 1 / dx ** 2, -2 / dx ** 2, 1 / dx ** 2
    return x, dx, D1, D2


def stable_time_stepping(dx, nu, T, u_max=1.0, safety=0.4):
    dt_max = safety * min(dx ** 2 / nu, dx / u_max)
    n_steps = 4 * int(np.ceil(T / (4 * dt_max)))
    return T / n_steps, n_steps


def initial_condition(x):
    u = np.sin(np.pi * x)
    u[0] = u[-1] = 0.0
    return u


def euler_step(u, dt, nu, D1, D2):
    u_new = u + dt * (nu * (D2 @ u) - u * (D1 @ u))
    u_new[0] = u_new[-1] = 0.0
    return u_new


def classical_solution(N, nu=0.1, T=1.0, length=1.0):
    x, dx, D1, D2 = derivative_matrices(N, length)
    dt, n_steps = stable_time_stepping(dx, nu, T)
    u = initial_condition(x)
    history = [u.copy()]
    for _ in range(n_steps):
        u = euler_step(u, dt, nu, D1, D2)
        history.append(u.copy())
    return dict(x=x, dx=dx, dt=dt, n_steps=n_steps, D1=D1, D2=D2, solution=np.array(history))
