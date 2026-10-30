import numpy as np
import pytest

pytest.importorskip("qiskit")

from qcfd.evolve import BurgersVQA


def test_dense_encoding_follows_classical_solution():
    result = BurgersVQA(N=8, T=0.2).run(verbose=False)
    assert result["initial_overlap"] > 1 - 1e-8
    assert result["rel_errors"].max() < 1e-3
    for info in result["infos"]:
        assert info["ez"] > 1 - 1e-6
        assert abs(info["alpha"] - info["alpha_exact"]) < 1e-4
        assert abs(info["F_eq4"] - info["F_true"]) < 1e-4


def test_tensor_network_encoding_matches_dense():
    dense = BurgersVQA(N=8, T=0.2, encoding="dense").run(verbose=False)
    tn = BurgersVQA(N=8, T=0.2, encoding="tn").run(verbose=False)
    assert tn["rel_errors"].max() < 1e-3
    assert np.allclose(tn["quantum"], dense["quantum"], atol=1e-3)
