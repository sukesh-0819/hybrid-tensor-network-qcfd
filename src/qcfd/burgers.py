import numpy as np

def setup_classical_burgers(N=16, L_dom=1.0, nu=0.1, T=1.0):
    x = np.linspace(0, L_dom, N)
    dx = x[1] - x[0]
    u_max = 1.0
    dt_max = 0.4 * min(dx ** 2 / nu, dx / u_max)
    n_steps = 4 * int(np.ceil(T / (4 * dt_max)))
    dt = T / n_steps
    
    u = np.sin(np.pi * x)
    u[0] = u[-1] = 0.0
    
    D1 = np.zeros((N, N))
    D2 = np.zeros((N, N))
    for i in range(1, N - 1):
        D1[i, i - 1], D1[i, i + 1] = -1 / (2 * dx), 1 / (2 * dx)
        D2[i, i - 1], D2[i, i], D2[i, i + 1] = 1 / dx ** 2, -2 / dx ** 2, 1 / dx ** 2
        
    classical_solution = [u.copy()]
    for _ in range(n_steps):
        u = u + dt * (nu * (D2 @ u) - u * (D1 @ u))
        u[0] = u[-1] = 0.0
        classical_solution.append(u.copy())
        
    return x, dx, dt, n_steps, D1, D2, np.array(classical_solution)