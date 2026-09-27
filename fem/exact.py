"""Timoshenko & Goodier exact plane-stress solution for a cantilever
with a parabolic shear traction of resultant P at its free end x = L.

Beam: 0 <= x <= L, -D/2 <= y <= D/2, unit thickness, I = D^3 / 12.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Beam:
    L: float = 48.0
    D: float = 12.0
    E: float = 3.0e7
    nu: float = 0.3
    P: float = 1000.0

    @property
    def I(self):  # noqa: E743 - standard notation
        return self.D**3 / 12.0

    def displacement(self, x, y):
        P, E, I, L, D, nu = self.P, self.E, self.I, self.L, self.D, self.nu
        ux = -P * y / (6 * E * I) * ((6 * L - 3 * x) * x + (2 + nu) * (y**2 - D**2 / 4))
        uy = P / (6 * E * I) * (3 * nu * y**2 * (L - x) + (4 + 5 * nu) * D**2 * x / 4 + (3 * L - x) * x**2)
        return ux, uy

    def stress(self, x, y):
        sxx = -self.P * (self.L - x) * y / self.I
        syy = np.zeros_like(sxx)
        sxy = self.P / (2 * self.I) * (self.D**2 / 4 - y**2)
        return sxx, syy, sxy

    def tip_deflection(self):
        """u_y at (L, 0): Euler-Bernoulli bending term + shear/elasticity correction."""
        P, E, I, L, D, nu = self.P, self.E, self.I, self.L, self.D, self.nu
        return P * L**3 / (3 * E * I) + (4 + 5 * nu) * P * D**2 * L / (24 * E * I)
