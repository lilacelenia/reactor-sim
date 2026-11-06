"""
verify_startup.py -- checks for source-driven (startup) kinetics and the instrumentation.

[1] Exact separable reference with a source: bare cube, source shaped like the fundamental
    mesh mode, reactivity ramped from deeply subcritical through critical. The 3D problem
    reduces exactly to 8 ODEs (2 groups + 6 precursors) with a source term.
[2] Source-driven steady state of the MITR-II-like core: nothing moves -> nothing changes;
    absolute power and multiplication are consistent with an independent fixed-source solve.
[3] Instruments: dead time, counting statistics, period meter, trips.
[4] Approach to criticality: blades withdrawn in steps with counting; the 1/M extrapolation
    must converge on the critical height found by a static eigenvalue search.

    python3 verify_startup.py            (all, ~10 min)
    python3 verify_startup.py cube       (single part)
"""
import sys
import time

import numpy as np

from core_builder import load_config as _load_config
from scipy.integrate import solve_ivp

from kinetics3d import BladeProgram, KineticsData, TransientSolver
from verify_kinetics import CubeModel

H0 = float(_load_config("mitr2_like.yaml")["transient"]["start_height"])   # critical bank height, cm

KIN = KineticsData()


# ----------------------------------------------------------------------------- [1]
def cube_mode(model):
    """Fundamental mode of the cube's discrete Laplacian (separable product of 1D modes)."""
    n = model.mesh.nx
    T = (np.diag(np.full(n, 2.0)) - np.diag(np.ones(n - 1), 1) - np.diag(np.ones(n - 1), -1))
    T[0, 0] = T[-1, -1] = 3.0
    _, V = np.linalg.eigh(T)
    u1 = np.abs(V[:, 0])
    return np.einsum("i,j,k->ijk", u1, u1, u1)


def test_cube_source():
    print("[1] Cube with source vs exact separable solution (physical units)")
    model = CubeModel(a=71.114, n=12, kappa=0.0617)     # k = 0.95 at h = 1, 1.004 at h = 2
    k = KIN
    m, B2 = model.MAT, model.B2
    u = cube_mode(model)
    q0 = 1.0e4                                   # n/(cm^3 s) at the mode's peak (arbitrary)
    Q = q0 * u / u.max()
    k_lo, k_hi = model.k_exact(1.0), model.k_exact(2.0)
    # control: h = 1 deeply subcritical -> ramp to 2 (supercritical) over 10 s, hold
    prog = BladeProgram(h0=1.0, drive_speed=0.1, scram_speed=50.0, travel=2.0)
    prog.command(2.0, 2.0)
    print(f"    k from {k_lo:.4f} (rho = ${(k_lo - 1) / k_lo / k.beta:+.2f}) to "
          f"{k_hi:.4f} (${(k_hi - 1) / k_hi / k.beta:+.2f}), ramp 2-12 s, run to 25 s")
    r1 = m["D1"] * B2 + m["Sa1"] + m["Ss12"]
    uhat = (u / u.max()).ravel()       # exact solution: phi_g(r, t) = T_g(t) * uhat(r)
    qm = q0

    def rhs(t, y):
        T1, T2, c = y[0], y[1], y[2:]
        S = m["nSf1"] * T1 + m["nSf2"] * T2
        r2 = m["D2"] * B2 + model.sa2(prog.position(t))
        dT1 = k.v1 * (-r1 * T1 + (1 - k.beta) * S + k.lam_i @ c + qm)
        dT2 = k.v2 * (m["Ss12"] * T1 - r2 * T2)
        return np.concatenate(([dT1, dT2], k.beta_i * S - k.lam_i * c))

    r2_0 = m["D2"] * B2 + model.sa2(1.0)
    A = np.array([[r1 - m["nSf1"], -m["nSf2"]], [-m["Ss12"], r2_0]])
    T1, T2 = np.linalg.solve(A, [qm, 0.0])
    S0 = m["nSf1"] * T1 + m["nSf2"] * T2
    y = np.concatenate(([T1, T2], k.beta_i * S0 / k.lam_i))
    sols = []
    knots = [0.0, 2.0, 12.0, 25.0]
    for a, b in zip(knots[:-1], knots[1:]):
        sol = solve_ivp(rhs, (a, b), y, method="Radau", rtol=1e-11, atol=1e-14, dense_output=True)
        y = sol.y[:, -1]
        sols.append(sol)

    def ref(t):
        for sol in sols:
            if sol.t[0] - 1e-12 <= t <= sol.t[-1] + 1e-12:
                T1, T2 = sol.sol(t)[:2]
                return (m["nSf1"] * T1 + m["nSf2"] * T2) / S0
    ref = np.vectorize(ref)
    errs = {}
    for method, dt in (("direct", 0.01), ("direct", 0.005), ("pcqs", 1.0), ("pcqs", 0.25)):
        t0 = time.time()
        p = BladeProgram(1.0, 0.1, 50.0, 2.0)
        p.command(2.0, 2.0)
        s = TransientSolver(model, p, method=method, dt_max=dt, dh_max=0.1 * dt if method == "direct" else 0.05,
                            tol=1e-9, source=Q)
        if method == "direct" and dt == 0.01:
            S_ex = (m["nSf1"] * T1 + m["nSf2"] * T2) * uhat.sum()
            print(f"    source-driven steady state: total fission rate {s.S0:.8e} vs exact {S_ex:.8e} "
                  f"(rel. diff {abs(s.S0 / S_ex - 1):.1e})")
            assert abs(s.S0 / S_ex - 1) < 1e-6
        s.run(25.0)
        t, P = np.array(s.hist.t), np.array(s.hist.power)
        e = np.abs(P / ref(t) - 1)
        errs[(method, dt)] = e.max()
        print(f"    {method:6s} dt_max={dt:<6} {len(t) - 1:5d} steps: P(25 s)/P(0) = {P[-1]:.5g} "
              f"(exact {float(ref(25.0)):.5g}), max rel. error {e.max():.1e}  ({time.time() - t0:.0f} s)")
    order = np.log2(errs[("direct", 0.01)] / errs[("direct", 0.005)])
    print(f"    direct convergence order {order:.2f} (expect ~1)")
    assert order > 0.8
    assert errs[("pcqs", 0.25)] < errs[("direct", 0.005)] and errs[("pcqs", 1.0)] < 1e-3
    return errs


# ----------------------------------------------------------------------------- MITR helpers
def mitr(cell=4.0):
    from core_builder import build, load_config
    from kinetics3d import kinetics_from_config
    cfg = load_config("mitr2_like.yaml")
    cfg["mesh"]["cell_size"] = cell
    model = build(cfg)
    return cfg, model, kinetics_from_config(cfg)


def static_critical_height(model, h_a=15.0, h_b=30.0):
    """Blade height where k = 1 (reference temperatures), by secant iteration."""
    from diffusion3d import solve_k
    ka = solve_k(model.problem_at(h_a), tol_k=1e-9, tol_src=1e-8).k
    kb = solve_k(model.problem_at(h_b), tol_k=1e-9, tol_src=1e-8).k
    for _ in range(30):
        h = h_b + (1 - kb) * (h_b - h_a) / (kb - ka)
        k = solve_k(model.problem_at(h), tol_k=1e-10, tol_src=1e-9).k
        h_a, ka, h_b, kb = h_b, kb, h, k
        if abs(k - 1) < 2e-7:
            return h
    return h


def source_solver(cfg, model, kin, h0=0.0, method="pcqs", **kw):
    from instruments import source_from_config
    from thermal import ChannelThermal
    bank = cfg["control"]["shim_bank"]
    prog = BladeProgram(h0, bank["drive_speed"], bank["scram_speed"], bank["travel"])
    return TransientSolver(model, prog, kin=kin, method=method, thermal=ChannelThermal(model, cfg),
                           source=source_from_config(cfg, model), **kw)


# ----------------------------------------------------------------------------- [2]
def test_source_steady():
    print("[2] Source-driven steady state (MITR-II-like, 4 cm mesh, blades fully in)")
    from diffusion3d import solve_fixed_source
    cfg, model, kin = mitr()
    s = source_solver(cfg, model, kin, dt_max=5.0)
    fs = solve_fixed_source(s._problem(0.0), Q1=s.Q.reshape(model.problem.mesh.shape), tol=1e-10)
    S_fs = (model.problem.xs["nSf1"] * fs.phi1 + model.problem.xs["nSf2"] * fs.phi2).sum()
    print(f"    k_eff = {s.k_initial:.5f}; total fission source {s.S0:.8e} vs independent source "
          f"iteration {S_fs:.8e} (rel. diff {abs(s.S0 / S_fs - 1):.1e}); power {s.P0 * 1e3:.3f} mW")
    print(f"    multiplication: fission neutrons / source neutrons = "
          f"{s.S0 * model.problem.mesh.cell_volume / (s.Q.sum() * model.problem.mesh.cell_volume):.3f} "
          f"(point estimate k/(1-k) = {s.k_initial / (1 - s.k_initial):.3f})")
    assert abs(s.S0 / S_fs - 1) < 1e-6
    for method in ("pcqs", "direct"):
        s = source_solver(cfg, model, kin, method=method, dt_max=5.0)
        s.run(30.0)
        dev = np.max(np.abs(np.array(s.hist.power) - 1))
        print(f"    {method:6s}: nothing moves for 30 s -> max |P/P0 - 1| = {dev:.1e}")
        assert dev < 1e-6


# ----------------------------------------------------------------------------- [3]
def test_instruments():
    print("[3] Instruments")
    from instruments import InstrumentSystem, run_with_trips
    from thermal import ChannelThermal
    from kinetics3d import PlantProgram
    cfg, model, kin = mitr()
    inst = InstrumentSystem(cfg, model, kin, seed=3)
    s = source_solver(cfg, model, kin, dt_max=5.0)
    inst.update(s)
    ch1 = next(r for r in inst.readings if r.name == "Ch 1")
    vals = []
    for _ in range(4000):                         # same state, repeated meter readings
        inst.update(s)
        vals.append(next(r for r in inst.readings if r.name == "Ch 1").value)
    sd_expected = np.sqrt(ch1.expected / (2 * inst.tau_meter))
    print(f"    shutdown: " + ", ".join(f"{r.name} {r.text()}" for r in inst.readings[:3]))
    print(f"    count-rate meter noise: mean {np.mean(vals):.3f} cps (expected {ch1.expected:.3f}), "
          f"std {np.std(vals):.3f} (RC meter theory {sd_expected:.3f})")
    assert abs(np.mean(vals) / ch1.expected - 1) < 0.05 and abs(np.std(vals) / sd_expected - 1) < 0.1
    # counting: 1000 s count at the steady shutdown rate -> Poisson
    inst.start_count(s, seconds=1000.0)
    s.run(1000.0, callback=lambda x: inst.update(x))
    cr, sig = inst.one_over_m.points[-1][1:3]
    print(f"    1000 s count: {cr:.3f} +- {sig:.3f} cps (true {ch1.expected:.3f})")
    assert abs(cr - ch1.expected) < 4 * sig
    # dead time
    ch = inst.channels[0]
    for P in (1e3, 5e3):
        n = ch["_cps_per_w"] * P
        print(f"    fission chamber at {P / 1e3:.0f} kW (reference shape): true {n:.3g} cps, "
              f"counted {inst._fc_rate(ch, P):.3g} cps (loss {1 - inst._fc_rate(ch, P) / n:.0%})")

    # period meter vs actual stable period, and the short-period scram
    from kinetics3d import TransientSolver
    prog = BladeProgram(H0, 0.2, 60.0, 53.0)
    prog.command(5.0, H0 + 0.1)
    s = TransientSolver(model, prog, kin=kin, thermal=ChannelThermal(model, cfg), power_w=1e3, dt_max=5.0)
    inst.reset()
    run_with_trips(s, inst, 200.0)
    h = s.hist
    T_true = (h.t[-1] - h.t[-4]) / np.log(h.power[-1] / h.power[-4])
    per = {r.name: r.period for r in inst.readings if np.isfinite(r.period)}
    print(f"    1 kW, +0.1 cm: true period {T_true:.1f} s; meters " +
          ", ".join(f"{k} {v:.1f} s" for k, v in per.items()) + f"; trips: {h.events or 'none'}")
    assert all(abs(v / T_true - 1) < 0.05 for v in per.values()) and not h.events
    prog = BladeProgram(H0, 0.2, 60.0, 53.0)
    prog.command(5.0, 30.0)                      # continuous withdrawal
    s = TransientSolver(model, prog, kin=kin, thermal=ChannelThermal(model, cfg), power_w=1e3, dt_max=1.0)
    inst.reset()
    run_with_trips(s, inst, 40.0)
    print(f"    continuous withdrawal from 1 kW: {s.hist.events[0][1]} at {s.hist.events[0][0]:.1f} s; "
          f"blades {s.hist.blade[-1]:.1f} cm, power {s.hist.power[-1] * 1e3:.1f} W at 40 s")
    assert "short-period" in s.hist.events[0][1] and s.hist.blade[-1] == 0.0
    # high flux trip at full power, and the no-flow limit
    prog = BladeProgram(H0, 0.2, 60.0, 53.0)
    prog.command(5.0, H0 + 0.1)
    s = TransientSolver(model, prog, kin=kin, thermal=ChannelThermal(model, cfg), power_w=6.0e6, dt_max=1.0)
    inst.reset()
    run_with_trips(s, inst, 120.0)
    print(f"    6 MW, +0.1 cm: {s.hist.events[0][1]} at {s.hist.events[0][0]:.1f} s")
    assert "high-flux" in s.hist.events[0][1]
    plant = PlantProgram()
    plant.set(5.0, "pumps", 0)
    s = TransientSolver(model, BladeProgram(H0, 0.2, 60.0, 53.0), kin=kin,
                        thermal=ChannelThermal(model, cfg), power_w=1.0e5, dt_max=1.0, plant=plant)
    inst.reset()
    run_with_trips(s, inst, 10.0)
    print(f"    100 kW, both pumps trip at 5 s: {s.hist.events[0][1]} at {s.hist.events[0][0]:.2f} s")
    assert "low primary flow" in s.hist.events[0][1]
    # below the no-flow limit the reactor may stay up in natural circulation; raising power
    # past 80 kW without flow then trips the high-flux scram
    plant = PlantProgram()
    plant.set(5.0, "pumps", 0)
    prog = BladeProgram(H0, 0.2, 60.0, 53.0)
    prog.command(400.0, H0 + 1.0)                    # ~190 pcm, after natural circulation has settled
    s = TransientSolver(model, prog, kin=kin, thermal=ChannelThermal(model, cfg), power_w=5.0e4,
                        dt_max=5.0, plant=plant)
    inst.reset()
    run_with_trips(s, inst, 800.0)
    ev = s.hist.events
    i400 = int(np.searchsorted(s.hist.t, 400.0))
    print(f"    50 kW, pumps trip at 5 s (no scram: below 80 kW; natural-circulation flow "
          f"{s.hist.core_flow[i400]:.2f} kg/s at 400 s), +1.0 cm at 400 s: "
          f"{ev[0][1] if ev else 'no trip'} at {ev[0][0] if ev else float('nan'):.1f} s")
    assert ev and "without primary flow" in ev[0][1] and ev[0][0] > 400.0


# ----------------------------------------------------------------------------- [4]
def test_approach(plot=True):
    print("[4] Approach to criticality with 1/M (4 cm mesh, source in, blades from 0 cm)")
    from instruments import InstrumentSystem, run_with_trips
    cfg, model, kin = mitr()
    h_crit = static_critical_height(model)
    print(f"    static eigenvalue search: critical height on this mesh {h_crit:.3f} cm")
    inst = InstrumentSystem(cfg, model, kin, seed=7)
    s = source_solver(cfg, model, kin, dt_max=10.0)
    speed = cfg["control"]["shim_bank"]["drive_speed"]
    MITR_STEP = 0.254                                      # 0.1 inch
    log = []

    def advance(t_end):
        run_with_trips(s, inst, t_end)
        if s.hist.events:
            raise RuntimeError(f"unexpected trip: {s.hist.events}")

    def count(seconds=30.0):
        inst.start_count(s, seconds)
        advance(s.t + seconds + 1e-9)
        p = inst.one_over_m.points[-1]
        log.append((s.t, p[0], p[1], p[3], inst.one_over_m.ecp()))
        return p

    h = 0.0
    count(60.0)
    last_e = None
    while True:
        p = inst.one_over_m.points[-1]
        inv_m, cr = p[3], p[1]
        e = inst.one_over_m.ecp()
        last_e = e if e is not None else last_e
        if last_e is not None and inv_m < 0.5 and last_e - h < MITR_STEP:
            break
        # far from critical: big steps; closer: half the distance to the ECP; close (1/M < 0.2):
        # single 0.1 inch increments as in the published MITR practice (bigger steps near
        # critical trip the 10 s period scram -- tried, and it did)
        if last_e is None or inv_m > 0.5:
            step = 4.0
        elif inv_m > 0.2:
            step = float(np.clip(0.5 * (last_e - h), MITR_STEP, 2.0))
        else:
            step = MITR_STEP
        h_new = h + step
        if last_e is not None and inv_m <= 0.5:
            h_new = min(h_new, last_e - 0.5 * MITR_STEP)
            if h_new <= h + 1e-6:
                break
        s.prog.command(s.t, h_new)
        wait = abs(h_new - h) / speed + min(600.0, 40.0 / max(inv_m, 0.067))
        advance(s.t + wait)
        h = h_new
        count(float(np.clip(2500.0 / max(cr, 1e-3), 30.0, 600.0)))   # ~2 % counting precision
        pt = inst.one_over_m.points[-1]
        e = inst.one_over_m.ecp()
        print(f"      blades {h:6.2f} cm: {pt[1]:9.1f} cps, 1/M = {pt[3]:.4f} +- {pt[4]:.4f}, "
              f"ECP = {e if e is not None else float('nan'):.3f} cm")
    ecp = last_e
    print(f"    final ECP {ecp:.3f} cm vs static critical height {h_crit:.3f} cm "
          f"(difference {ecp - h_crit:+.3f} cm; one MITR step = {MITR_STEP} cm); "
          f"{len(inst.one_over_m.points)} counts, {s.t / 60:.0f} min of reactor time")
    assert abs(ecp - h_crit) < MITR_STEP
    # go just supercritical like an operator: single 0.1 inch steps to ECP + 0.1 inch
    # (moving 0.5 cm in one go from here tripped the 10 s period scram)
    t_super = s.t
    target = ecp + MITR_STEP
    if inst.one_over_m.points[-1][3] < 1e-3:   # last count already diverging: supercritical,
        target = h                             # so an operator would not withdraw further
    while h < target - 1e-6:
        h_new = min(h + MITR_STEP, target)
        s.prog.command(s.t, h_new)
        advance(s.t + (h_new - h) / speed + 120.0)
        h = h_new
    advance(s.t + 300.0)
    per = [r.period for r in inst.readings if r.name == "Ch 1"][0]
    T_true = (s.hist.t[-1] - s.hist.t[-4]) / np.log(s.hist.power[-1] / s.hist.power[-4])
    print(f"    stepped to {target:.2f} cm (ECP + 0.1 inch, or stayed if already supercritical): power "
          f"{s.hist.power[-1] * s.P0:.3g} W, period meter {per:.0f} s (true {T_true:.0f} s), no trip")
    assert per > 10.0 and T_true > 0
    if plot:
        _plot_approach(s, inst, log, h_crit, t_super)
    return ecp, h_crit


def _plot_approach(s, inst, log, h_crit, t_super, out="startup_approach.png"):
    import matplotlib.pyplot as plt
    pts = np.array([(p[0], p[3], p[4]) for p in inst.one_over_m.points])
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    a = axs[0]
    a.errorbar(pts[:, 0], pts[:, 1], yerr=pts[:, 2], fmt="o-", ms=4, capsize=2, label="1/M (Ch 1 counts)")
    for i in range(1, len(pts)):
        h1, m1 = pts[i - 1, :2]
        h2, m2 = pts[i, :2]
        if m2 < m1:
            e = h2 + m2 * (h2 - h1) / (m1 - m2)
            a.plot([h2, e], [m2, 0], ":", color="C1", lw=0.8)
    a.axvline(h_crit, color="k", ls="--", lw=1, label=f"static critical height {h_crit:.2f} cm")
    a.set(xlabel="shim bank withdrawal [cm]", ylabel="1/M = CR(0) / CR", ylim=(0, 1.1),
          title="Inverse multiplication (dotted: ECP extrapolations)")
    a.legend(fontsize=8)
    a = axs[1]
    ecps = [(t, e) for t, _, _, _, e in log if e is not None]
    a.plot([e for _, e in ecps], "o-")
    a.axhline(h_crit, color="k", ls="--", lw=1)
    a.set(xlabel="count number", ylabel="estimated critical position [cm]",
          title="ECP estimate after each count")
    a = axs[2]
    h = s.hist
    a.semilogy(np.array(h.t) / 60, np.array(h.power) * s.P0, "C3", label="fission power")
    a.set(xlabel="time [min]", ylabel="power [W]", title="Approach to criticality and first period")
    a2 = a.twinx()
    a2.plot(np.array(h.t) / 60, h.blade, "C0", lw=1)
    a2.set_ylabel("blades [cm]", color="C0")
    a.axvline(t_super / 60, color="k", ls=":", lw=1)
    for a in axs:
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"    saved {out}")


if __name__ == "__main__":
    args = sys.argv[1:] or ["cube", "steady", "instruments", "approach"]
    if "cube" in args:
        test_cube_source()
    if "steady" in args:
        test_source_steady()
    if "instruments" in args:
        test_instruments()
    if "approach" in args:
        test_approach()
    print("\nStartup verification finished.")
