"""
verify_blades.py -- shim blades: geometry, fit to published HEU worths, independent checks.

The blade constants are "equivalent" diffusion values (a thin near-black absorber is outside
diffusion theory). They are fitted together with one fuel constant, to MITR-II HEU core 2
(ANL/RERTR/TM-10-41; the YAML holds the loading and the published values):
  blade Sa2 (Sa1 = Sa1_over_Sa2 * Sa2)  -> mean measured single-blade worth, 2253 pcm (Table 4-6)
  fuel nu-Sigma_f scale (nSf1 and nSf2)  -> measured critical bank height, 8.3 in (Table 4-2)
A single-blade worth is the blade's integral worth from fully in to fully out with the other
five blades at the critical bank height.

[1] Geometry (no eigenvalues): each blade's cross-section area equals thickness x width
    exactly on the 2 and 4 cm meshes; moving all six blades individually to the same height
    reproduces the whole-bank cross sections; moving one blade changes only its own cells;
    the MITR position labels match Fig. 2-7 and every element has the exact rhombus area.
[2] Fit targets reproduced (2 cm mesh): mean single-blade worth and critical height.
[3] Independent comparisons that were NOT fitted (2 cm mesh): each blade's worth vs its
    measured value (by MITR blade number), shutdown margin per stuck blade vs MCNP, bank
    worth vs the three published values, and the 4 cm live mesh.

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


def single_blade_worths(model, h_c, guess):
    """Per program side: integral worth of that blade, fully in -> fully out, the other five
    at h_c (regulating rod at its configured position)."""
    travel = model.mix["travel"]
    out = []
    for i in range(6):
        v = np.full(6, h_c)
        v[i] = 0.0
        k0 = k_at(model, v, None, guess).k
        v[i] = travel
        out.append(worth(k0, k_at(model, v, None, guess).k))
    return np.array(out)


def mitr_blade_numbers(cfg):
    """MITR blade number for each program side: blade 1 is the first clockwise (seen from
    the top) from the regulating-rod corner (TM-10-41, 2.3 and Fig. 2-7)."""
    a0 = float(cfg["control"]["regulating_rod"]["corner_angle"])
    sides = cfg["control"]["shim_bank"]["sides"]
    cw = [(a0 - 60.0 * s) % 360.0 for s in sides]
    order = np.argsort(cw)
    num = np.empty(len(sides), int)
    num[order] = np.arange(1, len(sides) + 1)
    return num


def calibrate(cell=2.0, start=None):
    cfg = load(cell)
    pub = published(cfg)
    W_t = float(np.mean(pub["single_blade_worths_pcm"])) * 1e-5
    H_t = float(pub["critical_height_in"]) * INCH
    print(f"Fitting blade Sa2 and the fuel nu-Sigma_f scale on the {cell:g} cm mesh to the mean "
          f"single-blade worth {W_t * 1e5:.0f} pcm and critical height {H_t:.2f} cm "
          f"(regulating rod at {cfg['control']['regulating_rod']['withdrawn']} cm)", flush=True)

    def F(p):
        sa2, f = np.exp(p)
        m = build(apply_fit(cfg, sa2, f))
        k_in, k_out, h_c, g = bank_state(m)
        w = single_blade_worths(m, h_c, g).mean()
        r = np.array([w / W_t - 1.0, h_c / H_t - 1.0])
        print(f"  Sa2 = {sa2:.5f}, fuel x {f:.5f}: single blade {w * 1e5:7.1f} pcm, "
              f"critical {h_c:.3f} cm, bank {worth(k_in, k_out) * 1e5:.0f} pcm", flush=True)
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
    print(f"Result: blade Sa2 = {sa2:.6g} /cm, Sa1 = {sa2 * cfg['control']['shim_bank']['Sa1_over_Sa2']:.6g} /cm; "
          f"fuel nSf1 = {fuel['nSf1'] * f:.7g}, nSf2 = {fuel['nSf2'] * f:.6g} /cm (scale {f:.6f})")


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
    # MITR position labels vs ANL/RERTR/TM-10-41 Fig. 2-7. Angles of the label positions in
    # the figure, seen from the top with the regulating rod at 0 deg (read off the figure;
    # labels sit near, not at, the element centres), counter-clockwise.
    fig = {"A-1": 0, "A-2": 240, "A-3": 120, "B-1": 343.3, "B-2": 306.5, "B-3": 270,
           "B-4": 223.4, "B-5": 179, "B-6": 135.8, "B-7": 90, "B-8": 54.5, "B-9": 18.4,
           "C-1": 349.7, "C-2": 325.6, "C-3": 304.3, "C-4": 282.2, "C-5": 257.8, "C-6": 231.5,
           "C-7": 204.4, "C-8": 179.4, "C-9": 155, "C-10": 127.6, "C-11": 102.1, "C-12": 77.9,
           "C-13": 56, "C-14": 35, "C-15": 11.4}
    cfg = load(2.0)
    lat = build(cfg).lattice
    a_rod = float(cfg["control"]["regulating_rod"]["corner_angle"])
    worst = 0.0
    for e in lat.elements:
        a = np.degrees(np.arctan2(e["center"][1], e["center"][0])) - a_rod
        best = min(fig, key=lambda L: abs((a - fig[L] + 180) % 360 - 180)
                   if L[0] == e["label"][0] else 1e9)
        assert best == e["label"], (e["label"], best)
        worst = max(worst, abs((a - fig[best] + 180) % 360 - 180))
    print(f"    MITR labels: every element's nearest Fig. 2-7 label in its ring is its own "
          f"(largest centre-to-label angle {worst:.1f} deg)")
    n = 3000
    g = (np.arange(n) + 0.5) / n * 2 * lat.hex_side - lat.hex_side
    X, Y = np.meshgrid(g, g, indexing="ij")
    idx = lat.locate(X, Y)
    A = np.bincount(idx[idx >= 0], minlength=len(lat.elements)) * (2 * lat.hex_side / n) ** 2
    exact = lat.s ** 2 * np.sin(np.pi / 3)
    err = np.abs(A / exact - 1).max()
    print(f"    element areas on a {n}^2 sample grid: max error {err:.1e} vs the rhombus area "
          f"{exact:.3f} cm2 (inner ring rotated: {lat.inner_rotated})")
    assert err < 2e-3
    print(f"    program sides 0-5 are MITR blades {[int(b) for b in mitr_blade_numbers(cfg)]}")


# ----------------------------------------------------------------------------- [2], [3]
def blade_table(model, h_c, guess, travel, reg_travel):
    """Per program side: integral worth (others at critical), and the shutdown margin with
    that blade fully out, the others in and the regulating rod out."""
    w = single_blade_worths(model, h_c, guess)
    g_in = k_at(model, 0.0, reg_travel, guess)
    sdm = []
    for i in range(6):
        s = np.zeros(6)
        s[i] = travel
        ks = k_at(model, s, reg_travel, g_in).k
        sdm.append((1 - ks) / ks)
    return np.column_stack([w, sdm])


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
    T = blade_table(m, h_c, g, travel, reg_travel)
    num = mitr_blade_numbers(cfg)
    meas = np.array(pub["single_blade_worths_pcm"])[num - 1] * 1e-5   # per program side
    sdm_ref = np.array(pub["shutdown_margin_pct"])[num - 1]
    print("[2] Fit targets (2 cm mesh, regulating rod at "
          f"{cfg['control']['regulating_rod']['withdrawn']} cm)")
    print(f"    mean single-blade worth {T[:, 0].mean() * 1e5:.1f} pcm vs measured mean "
          f"{meas.mean() * 1e5:.1f} pcm: {abs(T[:, 0].mean() / meas.mean() - 1):.1e}")
    print(f"    critical bank height {h_c:.3f} cm vs measured {pub['critical_height_in']} in = "
          f"{H_t:.3f} cm: {abs(h_c - H_t):.3f} cm")
    assert abs(T[:, 0].mean() / meas.mean() - 1) < 2e-3 and abs(h_c - H_t) < 0.05

    print("[3] Not fitted: model vs published HEU core 2 values")
    print("      MITR blade   worth (model / measured)    SDM, this blade out (model / MCNP)")
    for b in range(1, 7):
        i = int(np.where(num == b)[0][0])
        print(f"      {b}  (side {i})   {T[i, 0] * 1e5:6.0f} / {meas[i] * 1e5:4.0f} pcm  "
              f"{(T[i, 0] / meas[i] - 1) * 100:+5.1f} %     {T[i, 1] * 100:5.2f} / {sdm_ref[i]:.2f} %")
    dev = T[:, 0] / meas - 1
    print(f"    single-blade worths: RMS deviation {np.sqrt(np.mean(dev ** 2)) * 100:.1f} %, "
          f"largest {np.abs(dev).max() * 100:.1f} % (MCNP in TM-10-41 Table 4-6: 2-17 % low)")
    print(f"    shutdown margin {T[:, 1].min() * 100:.2f}-{T[:, 1].max() * 100:.2f} % dk/k vs MCNP "
          f"{min(pub['shutdown_margin_pct'])}-{max(pub['shutdown_margin_pct'])} %; tech-spec "
          f"minimum {pub['shutdown_margin_min_pct']} %")
    bw = pub["bank_worth_pcm"]
    print(f"    bank integral worth {W * 1e5:.0f} pcm vs MCNP core 2 {bw['mcnp_core2']:.0f}, "
          f"startup report {bw['startup_report'][0]:.0f}-{bw['startup_report'][1]:.0f}, "
          f"HEU SAR {bw['heu_sar']:.0f} pcm")
    beta = float(pub["pcm_per_beta"])
    ex = 1 - 1 / k_out
    print(f"    excess reactivity (bank out) {ex * 1e5:.0f} pcm = {ex * 1e5 / beta:.2f} beta "
          f"(fresh core 2; modern burned cores: {pub['excess_reactivity_beta_modern'][0]:g}-"
          f"{pub['excess_reactivity_beta_modern'][1]:g} beta, context only)")
    assert T[:, 1].min() * 100 > pub["shutdown_margin_min_pct"]
    # live mesh
    m4 = build(load(4.0))
    k_in4, k_out4, h4, _ = bank_state(m4)
    print(f"    4 cm live mesh: bank worth {worth(k_in4, k_out4) * 1e5:.0f} pcm, critical bank "
          f"{h4:.3f} cm, excess {(1 - 1 / k_out4) * 1e5:.0f} pcm  ({time.time() - t0:.0f} s)")
    np.savez("vb_worths.npz", table=T, W=W, ex=ex, h_c=h_c, h4=h4,
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
