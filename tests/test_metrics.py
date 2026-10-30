import numpy as np

from qcfd.metrics import alphas, fidelity_eq4, make_field, normalization_factor


def test_uniform_field():
    for N in (8, 64, 1024):
        state, block = alphas(np.ones(N))
        assert np.isclose(state, 1 / N)
        assert np.isclose(block, 1.0)


def test_uniform_random_matches_appendix_a():
    rng = np.random.default_rng(0)
    state, block = alphas(make_field("uniform random", 2 ** 18, rng))
    assert np.isclose(state * 2 ** 18, 9 / 5, rtol=0.02)
    assert np.isclose(block, 3 / 5, rtol=0.02)


def test_gaussian_state_based_matches_appendix_a():
    rng = np.random.default_rng(1)
    state, _ = alphas(make_field("gaussian random", 2 ** 18, rng))
    assert np.isclose(state * 2 ** 18, 3.0, rtol=0.05)


def test_localized_field_keeps_success_probability():
    rng = np.random.default_rng(2)
    values = [alphas(make_field("narrow peak", 2 ** n, rng))[0] for n in (6, 10, 14)]
    assert np.allclose(values, values[0], rtol=0.05)


def test_eq5_reduces_to_sqrt_alpha_at_optimal_phi():
    for alpha in (0.1, 0.4, 0.9):
        phi = 2 * np.arctan(np.sqrt(alpha))
        assert np.isclose(normalization_factor(alpha, phi), np.sqrt(alpha))


def test_eq4_is_one_at_optimum():
    for alpha in (0.1, 0.4, 0.9):
        phi = 2 * np.arctan(np.sqrt(alpha))
        assert np.isclose(fidelity_eq4(1.0, alpha, phi), 1.0)
