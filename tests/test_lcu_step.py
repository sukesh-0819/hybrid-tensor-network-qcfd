import numpy as np
import pytest

pytest.importorskip("qiskit")

from qcfd.lcu_step import Layout, lcu_weights


def test_layout_for_dense_encoding():
    for n_data in (2, 3, 4, 5):
        lay = Layout(n_data)
        assert lay.n_total == n_data + 4
        assert lay.data == list(range(2, 2 + n_data))
        assert lay.a1 == 2 + n_data
        assert lay.a2 == [3 + n_data]


def test_layout_masks_select_the_right_branch():
    lay = Layout(3)
    assert lay.keep.sum() == 2 * 2 ** 3
    assert lay.success1.sum() == 2 ** 3
    assert lay.keep[lay.branch1].all()
    assert (lay.cbit[lay.branch1] == 1).all()
    assert len(set(lay.branch1)) == 2 ** 3


def test_layout_with_tensor_network_register():
    lay = Layout(3, n_a2=4)
    assert lay.n_total == 10
    assert lay.a2 == [6, 7, 8, 9]
    assert lay.success1.sum() == 2 ** 3


def test_lcu_weights():
    beta2, beta_full, omega = lcu_weights(1.0, 0.5, 0.8, 2.0)
    assert np.isclose(beta2, 0.8)
    assert np.isclose(beta_full, 1.8)
    assert np.isclose(np.cos(omega / 2) ** 2, beta2 / beta_full)
