import numpy as np
import pytest

from fem import Beam, errors, solve
from fem.q4 import element_stiffness, plane_stress_C


def test_exact_solution_is_consistent():
    """Strains of the exact displacement field reproduce the exact stresses (Hooke's law)."""
    b = Beam()
    x, y, eps = 20.0, 3.0, 1e-4
    ux = lambda x, y: b.displacement(x, y)[0]  # noqa: E731
    uy = lambda x, y: b.displacement(x, y)[1]  # noqa: E731
    exx = (ux(x + eps, y) - ux(x - eps, y)) / (2 * eps)
    eyy = (uy(x, y + eps) - uy(x, y - eps)) / (2 * eps)
    gxy = (ux(x, y + eps) - ux(x, y - eps)) / (2 * eps) + (uy(x + eps, y) - uy(x - eps, y)) / (2 * eps)
    s = plane_stress_C(b.E, b.nu) @ [exx, eyy, gxy]
    assert np.allclose(s, b.stress(x, y), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("inc", [False, True])
def test_rigid_body_modes(inc):
    K = element_stiffness(1.0, 0.5, plane_stress_C(1.0, 0.3), incompatible=inc)
    w = np.linalg.eigvalsh(K)
    assert np.sum(np.abs(w) < 1e-10 * w.max()) == 3  # 2 translations + 1 rotation


@pytest.mark.parametrize("inc", [False, True])
def test_patch_test_constant_strain(inc):
    """A linear displacement field must be reproduced exactly (patch test)."""
    K = element_stiffness(1.0, 0.5, plane_stress_C(1.0, 0.3), incompatible=inc)
    xy = np.array([[-1, -0.5], [1, -0.5], [1, 0.5], [-1, 0.5]])
    u = np.column_stack([0.01 * xy[:, 0] + 0.02 * xy[:, 1], -0.03 * xy[:, 0] + 0.005 * xy[:, 1]]).ravel()
    f = K @ u
    # nodal forces of a constant stress state must sum to zero
    assert abs(f[0::2].sum()) < 1e-12 and abs(f[1::2].sum()) < 1e-12


def test_q4_converges_second_order_in_l2():
    b = Beam()
    e = [errors(b, *solve(b, nx, ny))[0] for nx, ny in [(16, 4), (32, 8)]]
    assert np.log2(e[0] / e[1]) == pytest.approx(2.0, abs=0.1)


def test_incompatible_modes_cure_shear_locking():
    b = Beam()
    tip = []
    for inc in (False, True):
        mesh, U = solve(b, 8, 2, incompatible=inc)
        k = np.where(np.isclose(mesh.nodes[:, 0], b.L) & np.isclose(mesh.nodes[:, 1], 0))[0][0]
        tip.append(U[k, 1] / b.tip_deflection())
    assert tip[0] < 0.92  # Q4 locks
    assert tip[1] > 0.995  # QM6 does not
