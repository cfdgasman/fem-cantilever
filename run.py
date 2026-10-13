"""Mesh-convergence study for the Timoshenko-Goodier cantilever; figures go to docs/."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from fem import Beam, errors, solve, von_mises_elements

MESHES = [(8, 2), (16, 4), (32, 8), (64, 16), (128, 32)]


def tip(beam, mesh, U):
    xy = mesh.nodes
    k = np.where(np.isclose(xy[:, 0], beam.L) & np.isclose(xy[:, 1], 0.0))[0][0]
    return U[k, 1]


def main():
    beam = Beam()
    exact_tip = beam.tip_deflection()
    print(f"exact tip deflection: {exact_tip:.6e}")
    res = {}
    for inc, name in ((False, "Q4"), (True, "QM6")):
        rows = []
        for nx, ny in MESHES:
            mesh, U = solve(beam, nx, ny, incompatible=inc)
            l2, en = errors(beam, mesh, U, incompatible=inc)
            rows.append((nx, ny, tip(beam, mesh, U) / exact_tip, l2, en))
        res[name] = rows

    h = np.array([beam.L / nx for nx, _ in MESHES])
    print("\n| Mesh | Q4 tip / exact | QM6 tip / exact | Q4 L2 err | QM6 L2 err | Q4 energy err | QM6 energy err |")
    print("|---|---|---|---|---|---|---|")
    for k, (nx, ny) in enumerate(MESHES):
        q, m = res["Q4"][k], res["QM6"][k]
        print(f"| {nx}×{ny} | {q[2]:.4f} | {m[2]:.5f} | {q[3]:.2e} | {m[3]:.2e} | {q[4]:.2e} | {m[4]:.2e} |")
    for name in res:
        l2 = np.array([r[3] for r in res[name]])
        en = np.array([r[4] for r in res[name]])
        o_l2 = np.polyfit(np.log(h[-3:]), np.log(l2[-3:]), 1)[0]
        o_en = np.polyfit(np.log(h[-3:]), np.log(en[-3:]), 1)[0]
        print(f"{name}: observed order L2 = {o_l2:.2f}, energy = {o_en:.2f}")

    # --- convergence plot
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for name, style in (("Q4", "o-"), ("QM6", "s-")):
        a1.loglog(h, [r[3] for r in res[name]], style, label=f"{name}  L2")
        a1.loglog(h, [r[4] for r in res[name]], style, mfc="none", label=f"{name}  energy")
        a2.semilogx(h, [r[2] for r in res[name]], style, label=name)
    a1.loglog(h, 0.4 * res["Q4"][0][4] * (h / h[0]), "k--", lw=1, label="O(h)")
    a1.loglog(h, 0.4 * res["Q4"][0][3] * (h / h[0]) ** 2, "k:", lw=1, label="O(h²)")
    a1.set(xlabel="element size h", ylabel="relative error", title="Error norms")
    a1.invert_xaxis()
    a2.axhline(1.0, color="k", lw=1)
    a2.set(xlabel="element size h", ylabel="tip deflection / exact", title="Tip deflection (shear locking)")
    a2.invert_xaxis()
    for a in (a1, a2):
        a.set_xticks(h, [f"{v:g}" for v in h])
        a.minorticks_off()
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("docs/convergence.png", dpi=120)

    # --- deformed shape with von Mises stress
    mesh, U = solve(beam, 48, 12, incompatible=True)
    vm = von_mises_elements(beam, mesh, U)
    scale = 0.06 * beam.L / np.abs(U).max()
    xy = mesh.nodes + scale * U
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.add_collection(PolyCollection(mesh.nodes[mesh.elements], facecolor="none", edgecolor="#bbbbbb", lw=0.3))
    pc = PolyCollection(xy[mesh.elements], array=vm, cmap="turbo", edgecolor="k", lw=0.15)
    ax.add_collection(pc)
    fig.colorbar(pc, ax=ax, label="von Mises stress")
    ax.autoscale()
    ax.set(aspect="equal", xlabel="x", ylabel="y",
           title=f"Deformed shape (×{scale:.0f}), QM6 48×12 mesh; grey = undeformed")
    fig.tight_layout()
    fig.savefig("docs/deformed.png", dpi=120)


if __name__ == "__main__":
    main()
