"""2D linear elasticity (plane stress) with bilinear Q4 elements.

* Structured rectangular mesh, nodes numbered row-major in x.
* Element stiffness by 2x2 Gauss quadrature (full integration). Because
  every element in a uniform mesh is the same, K_e is computed once and
  scattered with a sparse COO assembly.
* Optional incompatible modes (Wilson / Taylor "QM6"): two extra bubble
  modes per direction that remove shear locking in bending. They are
  condensed out statically at element level.
* Dirichlet data are imposed strongly by partitioning; tractions become
  consistent nodal loads through 2-point Gauss on each loaded edge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

GP = np.array([-1.0, 1.0]) / np.sqrt(3.0)
XI = np.array([-1.0, 1.0, 1.0, -1.0])
ETA = np.array([-1.0, -1.0, 1.0, 1.0])


def plane_stress_C(E, nu):
    return E / (1 - nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])


def _B(xi, eta, a, b, incompatible=False):
    """Strain-displacement matrix for a rectangle of half-sizes a (x) and b (y)."""
    dNdx = 0.25 * XI * (1 + ETA * eta) / a
    dNdy = 0.25 * ETA * (1 + XI * xi) / b
    B = np.zeros((3, 8))
    B[0, 0::2] = dNdx
    B[1, 1::2] = dNdy
    B[2, 0::2] = dNdy
    B[2, 1::2] = dNdx
    if not incompatible:
        return B
    # bubble modes P1 = 1 - xi^2, P2 = 1 - eta^2 for u and v
    dP = np.array([[-2 * xi / a, 0.0], [0.0, -2 * eta / b]])  # rows: d/dx, d/dy; cols: P1, P2
    Bi = np.zeros((3, 4))
    Bi[0, 0:2] = dP[0]  # du/dx
    Bi[1, 2:4] = dP[1]  # dv/dy
    Bi[2, 0:2] = dP[1]  # du/dy
    Bi[2, 2:4] = dP[0]  # dv/dx
    return B, Bi


def _incompatible_blocks(a, b, C):
    det = a * b
    Kcc = np.zeros((8, 8))
    Kci = np.zeros((8, 4))
    Kii = np.zeros((4, 4))
    for xi in GP:
        for eta in GP:
            B, Bi = _B(xi, eta, a, b, incompatible=True)
            Kcc += B.T @ C @ B * det
            Kci += B.T @ C @ Bi * det
            Kii += Bi.T @ C @ Bi * det
    return Kcc, Kci, Kii


def element_stiffness(a, b, C, incompatible=False):
    det = a * b
    if not incompatible:
        K = np.zeros((8, 8))
        for xi in GP:
            for eta in GP:
                B = _B(xi, eta, a, b)
                K += B.T @ C @ B * det
        return K
    Kcc, Kci, Kii = _incompatible_blocks(a, b, C)
    return Kcc - Kci @ np.linalg.solve(Kii, Kci.T)


@dataclass
class Mesh:
    L: float
    D: float
    nx: int
    ny: int

    @property
    def nodes(self):
        x = np.linspace(0, self.L, self.nx + 1)
        y = np.linspace(-self.D / 2, self.D / 2, self.ny + 1)
        X, Y = np.meshgrid(x, y)
        return np.column_stack([X.ravel(), Y.ravel()])

    @property
    def elements(self):
        n = self.nx + 1
        i, j = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        n0 = (j * n + i).ravel()
        return np.column_stack([n0, n0 + 1, n0 + n + 1, n0 + n])  # counter-clockwise

    @property
    def half_sizes(self):
        return self.L / self.nx / 2, self.D / self.ny / 2


def solve(beam, nx, ny, incompatible=False):
    """Solve the cantilever benchmark. Returns (mesh, nodal displacement array (n, 2))."""
    mesh = Mesh(beam.L, beam.D, nx, ny)
    xy = mesh.nodes
    conn = mesh.elements
    a, b = mesh.half_sizes
    C = plane_stress_C(beam.E, beam.nu)
    Ke = element_stiffness(a, b, C, incompatible)

    dofs = np.empty((len(conn), 8), dtype=int)
    dofs[:, 0::2] = 2 * conn
    dofs[:, 1::2] = 2 * conn + 1
    rows = np.repeat(dofs, 8, axis=1).ravel()
    cols = np.tile(dofs, (1, 8)).ravel()
    ndof = 2 * len(xy)
    K = sp.coo_matrix((np.tile(Ke.ravel(), len(conn)), (rows, cols)), shape=(ndof, ndof)).tocsr()

    # traction on the free end x = L: t = (0, sigma_xy(L, y))
    f = np.zeros(ndof)
    right = np.where(np.isclose(xy[:, 0], beam.L))[0]
    right = right[np.argsort(xy[right, 1])]
    for n1, n2 in zip(right[:-1], right[1:]):
        y1, y2 = xy[n1, 1], xy[n2, 1]
        half = (y2 - y1) / 2
        for g in GP:
            y = (y1 + y2) / 2 + g * half
            ty = beam.stress(beam.L, y)[2]
            f[2 * n1 + 1] += ty * (1 - g) / 2 * half
            f[2 * n2 + 1] += ty * (1 + g) / 2 * half

    # exact displacements prescribed on the clamped end x = 0
    left = np.where(np.isclose(xy[:, 0], 0.0))[0]
    fixed = np.concatenate([2 * left, 2 * left + 1])
    u = np.zeros(ndof)
    ux, uy = beam.displacement(xy[left, 0], xy[left, 1])
    u[2 * left], u[2 * left + 1] = ux, uy
    free = np.setdiff1d(np.arange(ndof), fixed)
    rhs = f[free] - K[free][:, fixed] @ u[fixed]
    u[free] = spsolve(K[free][:, free].tocsc(), rhs)
    return mesh, u.reshape(-1, 2)


def errors(beam, mesh, U, incompatible=False):
    """Relative L2 displacement error and relative energy-norm error (3x3 Gauss).

    With incompatible modes, the condensed internal amplitudes are recovered
    element by element and included in the strain (not in the L2 displacement,
    which is measured on the conforming part).
    """
    g3 = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)])
    w3 = np.array([5, 8, 5]) / 9
    a, b = mesh.half_sizes
    C = plane_stress_C(beam.E, beam.nu)
    Cinv = np.linalg.inv(C)
    xy = mesh.nodes
    conn = mesh.elements
    ue = U[conn].reshape(len(conn), 8)  # (ne, 8): u1 v1 u2 v2 ...
    xc = xy[conn].mean(axis=1)
    alpha = None
    if incompatible:
        _, Kci, Kii = _incompatible_blocks(a, b, C)
        alpha = -np.linalg.solve(Kii, Kci.T @ ue.T).T  # (ne, 4)
    e_l2 = n_l2 = e_en = n_en = 0.0
    for gi, wi in zip(g3, w3):
        for gj, wj in zip(g3, w3):
            N = 0.25 * (1 + XI * gi) * (1 + ETA * gj)
            x = xc[:, 0] + a * gi
            y = xc[:, 1] + b * gj
            uh = ue[:, 0::2] @ N
            vh = ue[:, 1::2] @ N
            uex, vex = beam.displacement(x, y)
            w = wi * wj * a * b
            e_l2 += w * np.sum((uh - uex) ** 2 + (vh - vex) ** 2)
            n_l2 += w * np.sum(uex**2 + vex**2)
            if incompatible:
                B, Bi = _B(gi, gj, a, b, incompatible=True)
                sh = (C @ (B @ ue.T + Bi @ alpha.T)).T
            else:
                B = _B(gi, gj, a, b)
                sh = (C @ B @ ue.T).T
            se = np.column_stack(beam.stress(x, y))
            d = sh - se
            e_en += w * np.einsum("ij,jk,ik->", d, Cinv, d)
            n_en += w * np.einsum("ij,jk,ik->", se, Cinv, se)
    return np.sqrt(e_l2 / n_l2), np.sqrt(e_en / n_en)


def von_mises_elements(beam, mesh, U):
    """Plane-stress von Mises stress at element centres."""
    a, b = mesh.half_sizes
    C = plane_stress_C(beam.E, beam.nu)
    ue = U[mesh.elements].reshape(-1, 8)
    s = (C @ _B(0.0, 0.0, a, b) @ ue.T).T
    sxx, syy, sxy = s.T
    return np.sqrt(sxx**2 - sxx * syy + syy**2 + 3 * sxy**2)
