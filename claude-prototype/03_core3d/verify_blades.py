"""
verify_blades.py -- shim blades: geometry, fit to published HEU worths, independent checks.

The blade constants are "equivalent" diffusion values (a thin near-black absorber is outside
diffusion theory). They are fitted together with one fuel constant:
  blade Sa2 (Sa1 = Sa1_over_Sa2 * Sa2)  -> published HEU shim bank integral worth, 9927 pcm
  fuel nu-Sigma_f scale (nSf1 and nSf2)  -> published HEU initial critical bank height, 7.36 in
Both targets are HEU SAR / MCNP values quoted in ANL/RERTR/TM-10-40 (Tables 1 and 4).

[1] Geometry (no eigenvalues): each blade's cross-section area equals thickness x width
    exactly on the 2 and 4 cm meshes; moving all six blades individually to the same height
    reproduces the whole-bank cross sections; moving one blade changes only its own cells.
[2] Fit targets reproduced (2 cm mesh): bank worth and critical height.
[3] Independent comparisons with published values that were NOT fitted (2 cm mesh):
    excess reactivity, single-blade worth, bank and single-blade worth from 5 in to fully
    out, shutdown margin with the most reactive blade stuck out, and the same on the 4 cm
    live mesh.

    python3 verify_blades.py calibrate [cell [Sa2 fuel_scale]]   # fit (2 cm default, ~30 min);
                                                # optional starting guess
    python3 verify_blades.py                    # [1]-[3] (~25 min)
"""
import copy
import sys
import time

import numpy as np

from core_builder import build, load_config
from diffusion3d import solve_k

CFG = "mitr2_like.yaml"
INCH = 2.54
TOL = 1e-9


def load(cell=2.0):
    cfg = load_config(CFG)
    cfg["mesh"]["cell_size"] = float(cell)
    return cfg


def k_at(model, h, r=None, guess=None):
    return solve_k(model.problem_at(h, r), tol_k=TOL, tol_src=TOL * 10, guess=guess)


def critical_bank(model, r=None, guess=None, h_lo=4.0, h_hi=40.0):
    """Bank height with k = 1 (secant, falling back to bisection if it leaves the bracket)."""
    ra = k_at(model, h_lo, r, guess)
    rb = k_at(model, h_hi, r, ra)
    a, ka, b, kb, g = h_lo, ra.k, h_hi, rb.k, rb
    if not ka < 1 < kb:
        raise RuntimeError(f"critical height not in [{h_lo}, {h_hi}] cm (k {ka:.4f} .. {kb:.4f})")
    lo, hi = (a, b)
    for _ in range(40):
        c = b + (1 - kb) * (b - a) / (kb - ka)
        if not lo < c < hi:
            c = 0.5 * (lo + hi)
        g = k_at(model, c, r, g)
        if g.k < 1:
            lo = c
        else:
            hi = c
        a, ka, b, kb = b, kb, c, g.k
        if abs(g.k - 1) < 1e-8:
            break
    return c, g


def worth(k_lo, k_hi):
    """Reactivity difference 1/k_lo - 1/k_hi."""
    return 1.0 / k_lo - 1.0 / k_hi


def published(cfg):
    return cfg["control"]["shim_bank"]["published"]


def apply_fit(cfg, sa2, fuel_scale):
    """Configuration with blade Sa2 (and Sa1 = ratio * Sa2) and the fuel nu-Sigma_f scaled."""
    c = copy.deepcopy(cfg)
    b = c["materials"]["blade"]
    b["Sa2"] = float(sa2)
    b["Sa1"] = float(c["control"]["shim_bank"]["Sa1_over_Sa2"]) * float(sa2)
    for g in ("nSf1", "nSf2"):
        c["materials"]["fuel"][g] *= float(fuel_scale)
    return c


# ----------------------------------------------------------------------------- calibrate
def bank_state(model, guess=None):
    """k with the bank fully in and fully out, and the critical bank height (regulating rod
    at its configured position)."""
    travel = model.mix["travel"]
    r_in = k_at(model, 0.0, None, guess)
    r_out = k_at(model, travel, None, r_in)
    h_c, g = critical_bank(model, guess=r_out)
    return r_in.k, r_out.k, h_c, g


def calibrate(cell=2.0, start=None):
    cfg = load(cell)
    pub = published(cfg)
    W_t = float(pub["bank_worth_pcm"]) * 1e-5
    H_t = float(pub["critical_height_in"]) * INCH
    print(f"Fitting blade Sa2 and the fuel nu-Sigma_f scale on the {cell:g} cm mesh to the bank "
          f"worth {W_t * 1e5:.0f} pcm and critical height {H_t:.2f} cm "
          f"(regulating rod at {cfg['control']['regulating_rod']['withdrawn']} cm)", flush=True)

    def F(p):
        sa2, f = np.exp(p)
        m = build(apply_fit(cfg, sa2, f))
        k_in, k_out, h_c, _ = bank_state(m)
        r = np.array([worth(k_in, k_out) / W_t - 1.0, h_c / H_t - 1.0])
        print(f"  Sa2 = {sa2:.5f}, fuel x {f:.5f}: bank {worth(k_in, k_out) * 1e5:8.1f} pcm, "
              f"critical {h_c:.3f} cm, excess {(1 - 1 / k_out) * 1e5:.0f} pcm", flush=True)
        return r

    # Newton with a finite-difference Jacobian at the start, Broyden updates afterwards
    p = np.log(start if start else [float(cfg["materials"]["blade"]["Sa2"]), 1.0])
    r = F(p)
    J = np.zeros((2, 2))
    for j, d in enumerate((0.05, 0.01)):
        q = p.copy()
        q[j] += d
        J[:, j] = (F(q) - r) / d
    for _ in range(12):
        dp = -np.linalg.solve(J, r)
        dp *= min(1.0, 0.5 / np.abs(dp).max())         # limit steps in log space
        p_new = p + dp
        r_new = F(p_new)
        J += np.outer(r_new - r - J @ dp, dp) / (dp @ dp)
        p, r = p_new, r_new
        if np.abs(r).max() < 2e-5:
            break
    sa2, f = np.exp(p)
    fuel = cfg["materials"]["fuel"]
    print(f"Result: blade Sa2 = {sa2:.5f} /cm, Sa1 = {sa2 * cfg['control']['shim_bank']['Sa1_over_Sa2']:.6f} /cm; "
          f"fuel nSf1 = {fuel['nSf1'] * f:.6f}, nSf2 = {fuel['nSf2'] * f:.5f} /cm (scale {f:.5f})")


# ----------------------------------------------------------------------------- [1]
def test_geometry():
    print("[1] Blade geometry and individual blade positions")
    for cell in (2.0, 4.0):
        cfg = load(cell)
        bank = cfg["control"]["shim_bank"]
        m = build(cfg)
        lat = m.lattice
        exact = bank["thickness"] * lat.hex_side
        areas = np.array([b.sum() for b in m.mix["blade_xy_sides"]]) * cell**2
        err = np.abs(areas / exact - 1).max()
        print(f"    {cell:g} cm mesh: blade areas {areas.min():.4f} .. {areas.max():.4f} cm2 vs "
              f"thickness x width {exact:.4f} cm2: max error {err:.1e}")
        assert err < 2e-3
        h = 18.3
        xs_bank = m.xs_at(h)
        xs_each = m.xs_at([h] * 6)
        d = max(np.abs(xs_bank[n] - xs_each[n]).max() for n in xs_bank)
        print(f"      six blades moved individually to {h} cm vs the bank at {h} cm: max xs difference {d:.1e}")
        assert d < 1e-12
        v = np.full(6, h)
        v[2] = 0.0
        diff = np.abs(m.xs_at(v)["Sa2"] - xs_bank["Sa2"]) > 0
        own = (m.mix["blade_xy_sides"][2] > 0)[:, :, None] & np.ones(m.problem.mesh.nz, bool)
        assert diff.any() and not np.any(diff & ~own)
        print("      inserting blade 2 alone changes only cells that blade 2 covers")


# ----------------------------------------------------------------------------- [2], [3]
def blade_table(model, h_c, guess, travel, reg_travel):
    """Per blade: integral worth (others at critical), 5 in -> out worth, and the shutdown
    margin with that blade fully out, the others in and the regulating rod out."""
    rows, g_in = [], k_at(model, 0.0, reg_travel, guess)
    for i in range(6):
        v = np.full(6, h_c)
        v[i] = 0.0
        k0 = k_at(model, v, None, guess).k
        v[i] = 5 * INCH
        k5 = k_at(model, v, None, guess).k
        v[i] = travel
        k1 = k_at(model, v, None, guess).k
        s = np.zeros(6)
        s[i] = travel
        ks = k_at(model, s, reg_travel, g_in).k
        rows.append((worth(k0, k1), worth(k5, k1), (1 - ks) / ks))
    return np.array(rows)


def test_worths():
    cfg = load(2.0)
    pub = published(cfg)
    bank = cfg["control"]["shim_bank"]
    travel = float(bank["travel"])
    reg_travel = float(cfg["control"]["regulating_rod"]["travel"])
    t0 = time.time()
    m = build(cfg)
    k_in, k_out, h_c, g = bank_state(m)
    W = worth(k_in, k_out)
    H_t = float(pub["critical_height_in"]) * INCH
    print("[2] Fit targets (2 cm mesh, regulating rod at "
          f"{cfg['control']['regulating_rod']['withdrawn']} cm)")
    print(f"    bank integral worth {W * 1e5:.1f} pcm vs published {pub['bank_worth_pcm']} pcm: "
          f"{abs(W * 1e5 / pub['bank_worth_pcm'] - 1):.1e}")
    print(f"    critical bank height {h_c:.3f} cm vs published {pub['critical_height_in']} in = "
          f"{H_t:.3f} cm: {abs(h_c - H_t):.3f} cm")
    assert abs(W * 1e5 / pub["bank_worth_pcm"] - 1) < 2e-3 and abs(h_c - H_t) < 0.05

    print("[3] Not fitted: model vs published HEU values")
    beta = float(pub["pcm_per_beta"])
    ex = 1 - 1 / k_out
    lo, hi = (x * beta for x in pub["excess_reactivity_beta"])
    print(f"    excess reactivity (bank out) {ex * 1e5:.0f} pcm = {ex * 1e5 / beta:.2f} beta vs published "
          f"{pub['excess_reactivity_beta'][0]}-{pub['excess_reactivity_beta'][1]} beta ({lo:.0f}-{hi:.0f} pcm)")
    k5 = k_at(m, 5 * INCH, None, g).k
    W5 = worth(k5, k_out)
    print(f"    bank 5 in -> out {W5 * 1e5:.0f} pcm = {W5 / W:.3f} of the bank vs published "
          f"{pub['bank_5in_to_out_pcm']} pcm of {pub['bank_worth_typical_pcm']} = "
          f"{pub['bank_5in_to_out_pcm'] / pub['bank_worth_typical_pcm']:.3f}")
    T = blade_table(m, h_c, g, travel, reg_travel)
    print("    per blade (this program's sides; MITR numbering not mapped):")
    print("      side   worth   5in->out   SDM, this blade out")
    for i, (w, w5, sdm) in enumerate(T):
        print(f"      {i}    {w * 1e5:7.0f}   {w5 * 1e5:7.0f}    {sdm * 100:6.2f} %")
    print(f"    single blade worth {T[:, 0].min() * 1e5:.0f}-{T[:, 0].max() * 1e5:.0f} pcm "
          f"(mean {T[:, 0].mean() * 1e5:.0f}) vs published blade #5 {pub['single_blade_worth_pcm']} pcm")
    print(f"    single blade 5 in -> out {T[:, 1].mean() * 1e5:.0f} pcm (mean) vs published "
          f"{pub['single_blade_5in_to_out_pcm']} pcm")
    s_lo, s_hi = pub["shutdown_margin_pct"]
    print(f"    shutdown margin {T[:, 2].min() * 100:.2f}-{T[:, 2].max() * 100:.2f} % dk/k vs published "
          f"{s_lo}-{s_hi} % (HEU core 2); tech-spec minimum {pub['shutdown_margin_min_pct']} %")
    if T[:, 2].min() * 100 < pub["shutdown_margin_min_pct"]:
        print("      NOTE: below the tech-spec minimum for some blades. Not a code error: the model's"
              " excess reactivity is too high (see above); the blade constants cannot change that"
              " (refits with Sa1/Sa2 = 0.3 or without the cusping correction moved it < 1 %)")
    # live mesh
    m4 = build(load(4.0))
    k_in4, k_out4, h4, _ = bank_state(m4)
    print(f"    4 cm live mesh: bank worth {worth(k_in4, k_out4) * 1e5:.0f} pcm, critical bank "
          f"{h4:.3f} cm, excess {(1 - 1 / k_out4) * 1e5:.0f} pcm  ({time.time() - t0:.0f} s)")
    np.savez("vb_worths.npz", table=T, W=W, W5=W5, ex=ex, h_c=h_c, h4=h4,
             W4=worth(k_in4, k_out4))


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "calibrate":
        calibrate(float(args[1]) if len(args) > 1 else 2.0,
                  [float(a) for a in args[2:4]] if len(args) > 3 else None)
    else:
        test_geometry()
        test_worths()
        print("All blade checks passed.")
