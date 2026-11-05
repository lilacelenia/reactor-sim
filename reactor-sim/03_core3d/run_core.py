"""
run_core.py -- static 3D analysis of a core described by a YAML configuration.

    python3 run_core.py mitr2_like.yaml

1. shim-bank sweep: k-eff vs blade withdrawal (the integral worth curve)
2. finds the critical bank height (secant iteration on k(h) = 1)
3. at critical: per-element power map, peaking, axial power shape, flux slices
Writes <name>_results.png and <name>_results.json.
"""
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle

from core_builder import build, load_config
from diffusion3d import solve_k


def blade_sweep(cfg, heights):
    results, guess = [], None
    for h in heights:
        t = time.time()
        model = build(cfg, blade_withdrawn=h)
        r = solve_k(model.problem, guess=guess)
        guess = r
        results.append((h, r.k))
        print(f"  blades at {h:5.1f} cm: k = {r.k:.5f}   ({r.outer_iterations} its, {time.time() - t:.1f} s)")
    return results


def find_critical(cfg, sweep, tol=2e-5):
    hs, ks = np.array([s[0] for s in sweep]), np.array([s[1] for s in sweep])
    if not (ks.min() < 1 < ks.max()):
        raise RuntimeError("bank cannot reach criticality in its travel range")
    i = np.searchsorted(ks, 1.0)            # k increases with withdrawal
    (h0, k0), (h1, k1) = (hs[i - 1], ks[i - 1]), (hs[i], ks[i])
    guess = None
    for _ in range(8):                      # secant / regula falsi on k(h) - 1
        h = h0 + (1 - k0) * (h1 - h0) / (k1 - k0)
        model = build(cfg, blade_withdrawn=h)
        r = solve_k(model.problem, tol_k=1e-7, tol_src=1e-6, guess=guess)
        guess = r
        print(f"  trial {h:6.2f} cm -> k = {r.k:.6f}")
        if abs(r.k - 1) < tol:
            return h, model, r
        if r.k < 1:
            h0, k0 = h, r.k
        else:
            h1, k1 = h, r.k
    return h, model, r


def plot(cfg, sweep, h_crit, model, r, out_png):
    lat, mesh = model.lattice, model.problem.mesh
    P = model.element_powers(r)
    fig = plt.figure(figsize=(15, 9.5))
    fig.suptitle(f"{cfg['name']} -- critical shim bank height {h_crit:.1f} cm "
                 f"(of {cfg['control']['shim_bank']['travel']:.0f} cm travel)", fontsize=13)

    # (a) element power map
    ax = fig.add_subplot(2, 3, 1)
    polys = [e["verts"] for e in lat.elements]
    fueled = P > 0
    pc = PolyCollection(polys, array=np.where(fueled, P, np.nan), cmap="inferno",
                        edgecolors="k", linewidths=0.8)
    pc.set_clim(P[fueled].min() * 0.95, P[fueled].max())
    pc.cmap.set_bad("#b0b8c0")
    ax.add_collection(pc)
    for e, p in zip(lat.elements, P):
        cx, cy = e["center"]
        txt = f"{e['label']}\n{p:.2f}" if p > 0 else f"{e['label']}\nAl"
        ax.text(cx, cy, txt, ha="center", va="center", fontsize=6.5,
                color="white" if 0 < p < np.median(P[fueled]) else "black")
    bank = cfg["control"]["shim_bank"]
    for side in bank["sides"]:
        ang = np.radians(60 * side)
        n, t = np.array([np.cos(ang), np.sin(ang)]), np.array([-np.sin(ang), np.cos(ang)])
        c = n * (lat.flat_distance + bank["gap"] + bank["thickness"] / 2)
        a, b = c - t * lat.hex_side / 2, c + t * lat.hex_side / 2
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#1f77b4", lw=3)
    ax.add_patch(Circle((0, 0), cfg["surroundings"]["tank_radius"], fill=False, ls="--", lw=0.8))
    ax.set(xlim=(-26, 26), ylim=(-26, 26), aspect="equal", title="Relative element power (blue = shim blades)")
    fig.colorbar(pc, ax=ax, shrink=0.8)

    # (b) integral blade worth curve
    ax = fig.add_subplot(2, 3, 2)
    hs = np.array([s[0] for s in sweep])
    rho = np.array([(s[1] - 1) / s[1] for s in sweep]) * 100
    ax.plot(hs, rho, "o-")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(h_crit, color="r", ls="--", lw=1, label=f"critical {h_crit:.1f} cm")
    ax.set(xlabel="shim bank withdrawal [cm]", ylabel="reactivity [% dk/k]",
           title="Integral shim bank worth (S-curve)")
    ax.legend()
    ax.grid(alpha=0.3)

    # (c) axial power profile of the hottest element
    ax = fig.add_subplot(2, 3, 3)
    e_hot = int(np.argmax(P))
    z = mesh.centers("z")
    prof = model.element_axial_profile(r, e_hot)
    m = prof > 0
    ax.plot(prof[m] / prof[m].mean(), z[m], "o-", ms=3)
    ax.axhline(h_crit, color="#1f77b4", ls="--", lw=1, label="blade tips")
    ax.set(xlabel="relative power", ylabel="height above fuel bottom [cm]",
           title=f"Axial power shape, hottest element {lat.elements[e_hot]['label']}")
    ax.legend()
    ax.grid(alpha=0.3)

    # (d,e) thermal flux slices
    ext_xy = [mesh.x0, mesh.x0 + mesh.nx * mesh.dx, mesh.y0, mesh.y0 + mesh.ny * mesh.dy]
    k_mid = int(np.argmin(abs(z - cfg["core"]["fuel_height"] / 4)))
    ax = fig.add_subplot(2, 3, 4)
    im = ax.imshow(r.phi2[:, :, k_mid].T, origin="lower", extent=ext_xy, cmap="viridis")
    ax.set(title=f"Thermal flux, horizontal slice z = {z[k_mid]:.0f} cm", xlabel="x [cm]", ylabel="y [cm]")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = fig.add_subplot(2, 3, 5)
    j_mid = mesh.ny // 2
    ext_xz = [mesh.x0, mesh.x0 + mesh.nx * mesh.dx, mesh.z0, mesh.z0 + mesh.nz * mesh.dz]
    im = ax.imshow(r.phi2[:, j_mid, :].T, origin="lower", extent=ext_xz, cmap="viridis", aspect="auto")
    ax.set(title="Thermal flux, vertical slice y = 0", xlabel="x [cm]", ylabel="z [cm]")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = fig.add_subplot(2, 3, 6)
    x = mesh.centers("x")
    ax.plot(x, r.phi1[:, j_mid, k_mid] / r.phi1[:, j_mid, k_mid].max(), label="fast")
    ax.plot(x, r.phi2[:, j_mid, k_mid] / r.phi1[:, j_mid, k_mid].max(), label="thermal")
    for edge in (cfg["surroundings"]["tank_radius"], cfg["surroundings"]["reflector"]["outer_radius"]):
        ax.axvline(edge, color="gray", ls=":", lw=1)
        ax.axvline(-edge, color="gray", ls=":", lw=1)
    ax.set(xlabel="x [cm]", ylabel="flux (fast peak = 1)", title="Radial flux traverse (dotted: tank, D2O edge)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=110)


def main(path):
    cfg = load_config(path)
    stem = Path(path).stem
    print(f"{cfg['name']}")
    travel = cfg["control"]["shim_bank"]["travel"]
    print("Shim bank sweep:")
    sweep = blade_sweep(cfg, np.linspace(0, travel, 7))
    print("Searching for critical bank height:")
    h_crit, model, r = find_critical(cfg, sweep)

    P = model.element_powers(r)
    fueled = P > 0
    order = np.argsort(-P)
    print(f"\nCritical shim bank height: {h_crit:.2f} cm  (k = {r.k:.6f})")
    print(f"Shim bank worth (out -> in): {((1/sweep[0][1]) - (1/sweep[-1][1]))*100:.2f} % dk/k")
    print(f"Excess reactivity, bank fully out: {(sweep[-1][1]-1)/sweep[-1][1]*100:.2f} % dk/k")
    print(f"Element peaking factor: {P.max():.3f} ({model.labels[order[0]]});  "
          f"lowest fueled: {P[fueled].min():.3f}")
    ring_avg = {ring: P[[i for i, e in enumerate(model.lattice.elements)
                         if e['ring'] == ring and P[i] > 0]].mean() for ring in range(3)}
    print("Ring-average relative power: " + ", ".join(f"{'ABC'[k]}: {v:.3f}" for k, v in ring_avg.items()))

    plot(cfg, sweep, h_crit, model, r, f"{stem}_results.png")
    json.dump({
        "name": cfg["name"],
        "critical_bank_height_cm": round(h_crit, 3),
        "k_at_critical": r.k,
        "sweep": [{"withdrawn_cm": float(h), "k": float(k)} for h, k in sweep],
        "element_power": {lab: round(float(p), 4) for lab, p in zip(model.labels, P)},
    }, open(f"{stem}_results.json", "w"), indent=2)
    print(f"Saved {stem}_results.png and {stem}_results.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mitr2_like.yaml")
