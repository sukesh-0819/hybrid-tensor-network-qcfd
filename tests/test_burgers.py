import numpy as np

from qcfd.burgers import classical_solution, derivative_matrices, stable_time_stepping


def test_stability_limits():
    for N in (4, 8, 16, 32, 64):
        _, dx, _, _ = derivative_matrices(N)
        dt, n_steps = stable_time_stepping(dx, 0.1, 1.0)
        assert 0.1 * dt / dx ** 2 <= 0.5
        assert dt / dx <= 0.5
        assert n_steps % 4 == 0
        assert np.isclose(dt * n_steps, 1.0)


def test_n8_settings():
    problem = classical_solution(8)
    assert problem["n_steps"] == 20
    assert np.isclose(problem["dt"], 0.05)


def test_boundaries_and_decay():
    solution = classical_solution(16)["solution"]
    assert np.all(solution[:, 0] == 0)
    assert np.all(solution[:, -1] == 0)
    assert np.max(np.abs(solution[-1])) < np.max(np.abs(solution[0]))
