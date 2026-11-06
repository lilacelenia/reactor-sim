"""
verify_control.py -- regulating rod and automatic power controller.

[1] Worth (2 cm mesh, static eigenvalues): integral worth at the critical bank height vs the
    published HEU value it is calibrated to (124 pcm); worth curve; shadowing (worth vs bank
    height) vs the published trend; tech-spec limits for a device on an automatic
    controller (available positive reactivity < 0.5 % dk/k; insertion rate < 5e-4 dk/k/s).
[2] Kinetics (4 cm mesh, zero power): the rod is withdrawn 10 cm by hand (nominal-7 -> nominal+3). The
    transient's reactivity must match the static value 1 - k0/k1 from independent eigenvalue
    solves, and the stable period must match the inhour equation with that reactivity.
[3] Closed loop at zero power (4 cm, 1 W, no feedback): AUTO holds the power, then the
    setpoint is raised 10 %. The power must settle exactly on the setpoint, and the rod on the
    position where an independent static search gives k = k0 (critical again).
    At 400 s the shim bank is moved +0.05 cm (a disturbance the controller must take out), so
    the final rod position differs from the start. A scram then switches the controller to
    manual and leaves the rod where it is.
[4] Closed loop at power (4 cm, 2 MW, feedback, decay heat off): setpoint +10 %. Final
    power = setpoint; final rod position vs an independent static search for the position at
    which the eigenvalue with steady-state temperatures at the new power returns to k0.
[5] Time accuracy (4 cm, 1 MW, feedback): setpoint +5 % at 2 s; PCQS (1 s and 0.5 s steps)
    vs a Richardson-extrapolated direct reference (0.1, 0.05, 0.025 s) at the controller's
    sample times, which every run shares exactly.

    python3 verify_control.py calibrate        # fit reg_rod Sa2 to 124 pcm (2 cm mesh, ~10 min)
    python3 verify_control.py                  # [1]-[4] (~20 min)
    python3 verify_control.py run [NAME ...] [--jobs N]   # [5] reference runs (all if no names),
                                                         # several processes at once
    python3 verify_control.py compare          # [5] compare and plot
"""
import copy
import sys
import time

import numpy as np

from controller import controller_from_config, regulating_rod_program
from core_builder import build, load_config
from parallel import pop_jobs, run_parallel
from diffusion3d import Problem, solve_k
from kinetics3d import BladeProgram, TransientSolver, kinetics_from_config
from run_transient import inhour_period
from thermal import ChannelThermal

CFG = "mitr2_like.yaml"


def load(cell):
    cfg = load_config(CFG)
    cfg["mesh"]["cell_size"] = float(cell)
    return cfg


def k_at(model, h, r, guess=None, tol=1e-11):
    return solve_k(model.problem_at(h, r), tol_k=tol, tol_src=tol * 10, guess=guess)


def critical_bank(model, r, h_lo=12.0, h_hi=26.0, tol=1e-9):
    """Bank height with k = 1 for the regulating rod at r (secant / regula falsi)."""
    ra, rb = k_at(model, h_lo, r), k_at(model, h_hi, r)
    a, ka, b, kb, g = h_lo, ra.k, h_hi, rb.k, rb
    for _ in range(30):
        c = b + (1 - kb) * (b - a) / (kb - ka)
        g = k_at(model, c, r, g)
        a, ka, b, kb = b, kb, c, g.k
        if abs(g.k - 1) < tol:
            break
    return c


def integral_worth(model, h):
    travel = model.mix["reg"]["travel"]
    k_in, k_out = k_at(model, h, 0.0).k, k_at(model, h, travel).k
    return 1 / k_in - 1 / k_out


# ----------------------------------------------------------------------------- calibrate
def calibrate():
    cfg = load(2.0)
    rr = cfg["control"]["regulating_rod"]
    target = float(rr["published_worth_pcm"]) * 1e-5
    print(f"Calibrating reg_rod Sa2 to {target * 1e5:.1f} pcm at the critical bank height "
          f"(2 cm mesh, rod at {rr['withdrawn']} cm)")
    hist = []

    def worth(sa2):
        c = copy.deepcopy(cfg)
        c["materials"]["reg_rod"]["Sa2"] = sa2
        m = build(c)
        h = critical_bank(m, float(rr["withdrawn"]))
        w = integral_worth(m, h)
        hist.append((sa2, h, w))
        print(f"  Sa2 = {sa2:.5f}: critical bank {h:.4f} cm, worth {w * 1e5:.3f} pcm", flush=True)
        return w

    a = float(cfg["materials"]["reg_rod"]["Sa2"])
    b = a * 1.2
    wa, wb = worth(a), worth(b)
    for _ in range(10):                      # secant in ln(Sa2)
        lc = np.log(b) + (target - wb) * (np.log(b) - np.log(a)) / (wb - wa)
        c = float(np.exp(lc))
        wc = worth(c)
        a, wa, b, wb = b, wb, c, wc
        if abs(wc - target) < 0.02e-5:
            break
    sa2, h, w = hist[-1]
    print(f"Result: Sa2 = {sa2:.4f} /cm -> worth {w * 1e5:.2f} pcm; critical bank {h:.3f} cm "
          f"(put both in the YAML: materials.reg_rod.Sa2, transient.start_height)")


# ----------------------------------------------------------------------------- [1]
def test_worth():
    print("[1] Regulating rod worth (2 cm mesh, static eigenvalues)")
    cfg = load(2.0)
    rr = cfg["control"]["regulating_rod"]
    model = build(cfg)
    r0, travel = float(rr["withdrawn"]), float(rr["travel"])
    t0 = time.time()
    h_c = critical_bank(model, r0)
    W = integral_worth(model, h_c)
    pub = float(rr["published_worth_pcm"]) * 1e-5
    print(f"    critical bank (rod at {r0:g} cm): {h_c:.3f} cm")
    print(f"    integral worth {W * 1e5:.2f} pcm vs published HEU value {pub * 1e5:.0f} pcm "
          f"(the calibration target): {abs(W / pub - 1):.1e}")
    assert abs(W / pub - 1) < 2e-3
    # worth curve at the critical bank
    hs = np.linspace(0.0, travel, 23)
    ks, g = [], None
    for r in hs:
        g = k_at(model, h_c, r, g)
        ks.append(g.k)
    rho = 1 - ks[0] / np.array(ks)
    d = np.gradient(rho, hs)
    i0 = int(np.argmin(np.abs(hs - r0)))
    up = rho[-1] - float(np.interp(r0, hs, rho))
    print(f"    worth curve: differential worth max {d.max() * 1e5:.2f} pcm/cm at {hs[d.argmax()]:.1f} cm, "
          f"{d[i0] * 1e5:.2f} pcm/cm at the nominal {r0:g} cm; above the bank tip "
          f"({h_c:.1f} cm) < {np.abs(d[hs > h_c + 3]).max() * 1e5:.2f} pcm/cm (shadowed)")
    vmax = float(rr["drive_speed"])
    print(f"    tech spec 3.2.2.4: available positive reactivity from {r0:g} cm = {up * 1e5:.1f} pcm "
          f"< 500 pcm;  3.2.2.1: max rate {d.max() * vmax * 1e5:.2f} pcm/s < 50 pcm/s")
    assert up < 5e-3 and d.max() * vmax < 5e-4
    # shadowing: integral worth vs bank height
    print("    integral worth vs bank height (published trend, LEU analysis: about doubles as the "
          "bank is withdrawn through its range):")
    rows = []
    for h in (12.0, 15.0, h_c, 35.0, 53.0):
        w = integral_worth(model, h)
        rows.append((h, w))
        print(f"      bank {h:5.1f} cm: {w * 1e5:7.1f} pcm")
    ws = [w for _, w in rows]
    assert all(np.diff(ws) > 0), "worth should grow as the bank is withdrawn"
    # live mesh
    m4 = build(load(4.0))
    h4 = float(cfg["transient"]["start_height"])
    print(f"    4 cm mesh (live mode), bank {h4:g} cm: integral worth {integral_worth(m4, h4) * 1e5:.1f} pcm "
          f"({time.time() - t0:.0f} s)")
    np.savez("vc_worth.npz", r=hs, rho=rho, bank=np.array(rows))


# ----------------------------------------------------------------------------- [2]
def test_inhour():
    cfg = load(4.0)
    r_a = float(cfg["control"]["regulating_rod"]["withdrawn"]) - 7.0   # 10 cm, ending just
    r_b = r_a + 10.0                                                  # below the bank tip
    print(f"[2] Rod withdrawal {r_a:g} -> {r_b:g} cm at zero power: reactivity and stable period")
    model = build(cfg)
    kin = kinetics_from_config(cfg)
    h0 = float(cfg["transient"]["start_height"])
    reg = regulating_rod_program(cfg, h0=r_a)
    reg.command(10.0, r_b)
    t0 = time.time()
    s = TransientSolver(model, BladeProgram(h0, 0.2, 60, 53), kin=kin, power_w=1.0, dt_max=20.0,
                        reg_program=reg, tol=1e-8)
    s.run(1600.0)
    h = s.hist.arrays()
    k1 = k_at(model, h0, r_b).k
    rho_static = 1 - s.k0 / k1
    rho_end = h["rho"][-1]
    m = h["t"] >= 1200.0
    T_sim = (h["t"][m][-1] - h["t"][m][0]) / np.log(h["power"][m][-1] / h["power"][m][0])
    T_inh = inhour_period(rho_static, h["gen_time"][-1], kin)
    print(f"    {len(h['t']) - 1} steps ({time.time() - t0:.0f} s); rod at {h['reg_rod'][-1]:.2f} cm")
    print(f"    reactivity: transient {rho_end * 1e5:.4f} pcm, static 1 - k0/k1 {rho_static * 1e5:.4f} pcm "
          f"-> {abs(rho_end / rho_static - 1):.1e}")
    print(f"    stable period (1200-1600 s): simulated {T_sim:.2f} s, inhour {T_inh:.2f} s "
          f"-> {abs(T_sim / T_inh - 1):.1e}")
    assert abs(rho_end / rho_static - 1) < 2e-3 and abs(T_sim / T_inh - 1) < 5e-3


# ----------------------------------------------------------------------------- [3]
def closed_loop(cfg, model, kin, P0, thermal, commands, t_end, method="pcqs", dt_max=5.0,
                bank_moves=(), **kw):
    reg = regulating_rod_program(cfg)
    ctl = controller_from_config(cfg, reg)
    for c in commands:
        ctl.schedule(*c)
    prog = BladeProgram(float(cfg["transient"]["start_height"]), 0.2, 60, 53)
    for t, target in bank_moves:
        prog.command(t, target)
    s = TransientSolver(model, prog, kin=kin, method=method, power_w=P0, thermal=thermal,
                        reg_program=reg, controller=ctl, dt_max=dt_max, **kw)
    s.run(t_end)
    return s, ctl


def reg_for_k(model, h, k_target, r_lo, r_hi, k_of=None):
    """Independent static search: regulating rod position with k = k_target."""
    k_of = k_of or (lambda r: k_at(model, h, r).k)
    a, b = r_lo, r_hi
    ka, kb = k_of(a), k_of(b)
    for _ in range(30):
        c = b + (k_target - kb) * (b - a) / (kb - ka)
        kc = k_of(c)
        a, ka, b, kb = b, kb, c, kc
        if abs(kc - k_target) < 1e-11:
            break
    return c


def settle_report(s, ctl, P_set, t_step, t_stop=np.inf):
    """Final power, overshoot (fraction of the step) and time to stay within 0.1 % of the
    setpoint, looking only at t_step <= t < t_stop."""
    h = s.hist.arrays()
    P = h["power"] * s.P0
    win = (h["t"] >= t_step) & (h["t"] < t_stop)
    tw, Pw = h["t"][win], P[win]
    over = (Pw.max() - P_set) / (P_set - P[0]) if P_set > P[0] else (Pw.min() - P_set) / (P_set - P[0])
    bad = np.nonzero(np.abs(Pw / P_set - 1) > 1e-3)[0]
    t_set = tw[bad[-1] + 1] - t_step if len(bad) and bad[-1] + 1 < len(Pw) else np.nan
    return P[-1], over, t_set


def test_zero_power():
    print("[3] Closed loop at zero power (4 cm, 1 W, no feedback)")
    cfg = load(4.0)
    model = build(cfg)
    kin = kinetics_from_config(cfg)
    h0 = float(cfg["transient"]["start_height"])
    r0 = float(cfg["control"]["regulating_rod"]["withdrawn"])
    t0 = time.time()
    h1 = h0 + 0.05
    s, ctl = closed_loop(cfg, model, kin, 1.0, None, [(0.0, "auto", (True, None)),
                                                       (10.0, "setpoint", 1.1)], 1400.0,
                         bank_moves=[(400.0, h1)])
    h = s.hist.arrays()
    P_end, over, t_set = settle_report(s, ctl, 1.1, 10.0, 400.0)
    i400 = int(np.searchsorted(h["t"], 400.0))
    P400 = h["power"][i400]
    dist = np.max(np.abs(h["power"][i400:] / 1.1 - 1))
    r_end = h["reg_rod"][-1]
    r_static = reg_for_k(model, h1, s.k0, r0 - 4.0, r0)
    d = (k_at(model, h1, r_end + 0.05).k - k_at(model, h1, r_end - 0.05).k) / 0.1 / s.k0
    rho_bank = 1 - k_at(model, h0, r0).k / k_at(model, h1, r0).k
    print(f"    {len(h['t']) - 1} steps ({time.time() - t0:.0f} s); {len(ctl.log)} controller samples; "
          f"rod {r0:g} -> {r_end:.4f} cm")
    print(f"    setpoint 1.1 W at 10 s: final power {P_end:.9f} W -> |P/P_set - 1| = {abs(P_end / 1.1 - 1):.1e}; "
          f"overshoot {over * 100:.1f} % of the step; within 0.1 % after {t_set:.0f} s")
    print(f"    bank +0.05 cm at 400 s ({rho_bank * 1e5:+.1f} pcm; P/P_set - 1 was {P400 / 1.1 - 1:+.1e}): "
          f"largest power excursion after it {dist * 100:.2f} %")
    print(f"    final rod {r_end:.4f} cm vs static search for k = k0: {r_static:.4f} cm "
          f"(difference {abs(r_end - r_static):.1e} cm = {abs(r_end - r_static) * d * 1e5:.1e} pcm)")
    assert abs(P_end / 1.1 - 1) < 1e-5 and abs(r_end - r_static) * d < 1e-7
    # scram: controller to manual, rod stays put
    reg2 = ctl.reg
    s.scram("manual scram")
    s.run(s.t + 5.0)
    msg = ctl.messages[-1][1]
    print(f"    scram at {s.t - 5:.0f} s: controller '{msg}', auto = {ctl.auto}; rod "
          f"{s.hist.reg_rod[-1]:.4f} cm (was {r_end:.4f}); blades {s.hist.blade[-1]:.1f} cm")
    assert not ctl.auto and abs(s.hist.reg_rod[-1] - r_end) < 1e-12 and reg2.scram_time is None
    np.savez("vc_zero_power.npz", t=h["t"], P=h["power"], reg=h["reg_rod"])


# ----------------------------------------------------------------------------- [4]
def static_reg_with_feedback(cfg, model, h, P, k_target, r_lo, r_hi):
    th = ChannelThermal(model, cfg)

    def k_of(r):
        g = None
        for _ in range(60):
            prob = Problem(model.problem.mesh, th.apply(model.problem_at(h, r).xs), model.problem.bc)
            g = solve_k(prob, tol_k=1e-12, tol_src=1e-11, guess=g)
            S = (prob.xs["nSf1"] * g.phi1 + prob.xs["nSf2"] * g.phi2).ravel()
            Tf, Tc = th.steady_state(th.node_power(S, P / S.sum()))
            dT = np.abs(Tc - th.Tc).max() + np.abs(Tf - th.Tf).max()
            th.Tf, th.Tc = Tf, Tc
            if dT < 1e-9:
                break
        return g.k
    return reg_for_k(model, h, k_target, r_lo, r_hi, k_of)


def test_at_power():
    print("[4] Closed loop at 2 MW with feedback (decay heat off), setpoint 2.2 MW at 10 s")
    cfg = load(4.0)
    cfg = {k: v for k, v in cfg.items() if k != "decay_heat"}   # needs days to settle
    model = build(cfg)
    kin = kinetics_from_config(cfg)
    h0 = float(cfg["transient"]["start_height"])
    r0 = float(cfg["control"]["regulating_rod"]["withdrawn"])
    t0 = time.time()
    s, ctl = closed_loop(cfg, model, kin, 2e6, ChannelThermal(model, cfg),
                         [(0.0, "auto", (True, None)), (10.0, "setpoint", 2.2e6)], 900.0)
    h = s.hist.arrays()
    P_end, over, t_set = settle_report(s, ctl, 2.2e6, 10.0)
    r_end = h["reg_rod"][-1]
    print(f"    {len(h['t']) - 1} steps ({time.time() - t0:.0f} s); rod {r0:g} -> {r_end:.4f} cm; "
          f"feedback {(h['rho_feedback'][-1] - h['rho_feedback'][0]) * 1e5:+.2f} pcm")
    print(f"    final power {P_end / 1e6:.7f} MW -> |P/P_set - 1| = {abs(P_end / 2.2e6 - 1):.1e}; "
          f"overshoot {over * 100:.1f} % of the step; within 0.1 % after {t_set:.0f} s")
    r_static = static_reg_with_feedback(cfg, model, h0, 2.2e6, s.k0, r_end - 0.5, r_end + 0.5)
    d = (k_at(model, h0, r_end + 0.05).k - k_at(model, h0, r_end - 0.05).k) / 0.1 / s.k0
    print(f"    final rod {r_end:.4f} cm vs static equilibrium search {r_static:.4f} cm "
          f"(difference {abs(r_end - r_static):.1e} cm = {abs(r_end - r_static) * d * 1e5:.1e} pcm)")
    assert abs(P_end / 2.2e6 - 1) < 1e-5 and abs(r_end - r_static) * d < 2e-7
    np.savez("vc_at_power.npz", t=h["t"], P=h["power"], reg=h["reg_rod"], Tf=h["fuel_max"])


# ----------------------------------------------------------------------------- [5]
RUNS = {
    "direct_h": ("direct", 0.025),
    "direct_2h": ("direct", 0.05),
    "direct_4h": ("direct", 0.1),
    "pcqs_1.0": ("pcqs", 1.0),
    "pcqs_0.5": ("pcqs", 0.5),
}
T5 = 40.0


def timed_run(name):
    cfg = load(4.0)
    model = build(cfg)
    kin = kinetics_from_config(cfg)
    method, dt = RUNS[name]
    t0 = time.time()
    s, ctl = closed_loop(cfg, model, kin, 1e6, ChannelThermal(model, cfg),
                         [(0.0, "auto", (True, None)), (2.0, "setpoint", 1.05e6)], T5,
                         method=method, dt_max=dt, dlnp_max=0.0 if method == "direct" else 0.25)
    h = s.hist.arrays()
    np.savez(f"vc_{name}.npz", t=h["t"], P=h["power"], reg=h["reg_rod"], Tf=h["fuel_max"],
             wall=time.time() - t0)
    print(f"    {name:10s} {len(h['t']) - 1:5d} steps {time.time() - t0:6.0f} s", flush=True)


def compare():
    print("[5] Closed loop at 1 MW with feedback, setpoint +5 % at 2 s: PCQS vs extrapolated direct")
    d = {n: np.load(f"vc_{n}.npz") for n in RUNS}
    ts = np.arange(0.0, T5 + 1e-9, 1.0)                       # controller sample times

    def at(n, key):
        t = d[n]["t"]
        j = np.searchsorted(t, ts - 1e-7)
        assert np.all(np.abs(t[j] - ts) < 1e-7), f"{n} misses a sample time"
        return d[n][key][j]

    out, errs = {}, {}
    for key in ("P", "reg", "Tf"):
        a, b, c = at("direct_h", key), at("direct_2h", key), at("direct_4h", key)
        ref = (8 * a - 6 * b + c) / 3
        unc = np.max(np.abs((2 * a - b) - ref))
        out[key] = ref
        print(f"    reference {key:3s}: 2nd-order Richardson (25/50/100 ms), uncertainty "
              f"{unc / (np.abs(ref).max() if key == 'P' else 1):.1e}{'' if key == 'P' else (' cm' if key == 'reg' else ' K')}")
    for n in ("direct_4h", "direct_h", "pcqs_1.0", "pcqs_0.5"):
        eP = np.max(np.abs(at(n, "P") / out["P"] - 1))
        er = np.max(np.abs(at(n, "reg") - out["reg"]))
        eT = np.max(np.abs(at(n, "Tf") - out["Tf"]))
        errs[n] = eP
        print(f"    {RUNS[n][0]:7s} dt={RUNS[n][1]:<6} power {eP:.1e}, rod {er:.1e} cm, hottest plate "
              f"{eT:.1e} K  ({float(d[n]['wall']):.0f} s)")
    print(f"    reference: power {out['P'][-1]:.5f} x 1 MW at {T5:g} s, rod {out['reg'][0]:.3f} -> "
          f"{out['reg'][-1]:.3f} cm")
    plot(d, ts, out)
    assert errs["pcqs_1.0"] < 1e-4 and errs["pcqs_0.5"] < errs["direct_h"]


def plot(d, ts, ref, out="results/verify_control.png"):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    z, a4 = np.load("vc_zero_power.npz"), np.load("vc_at_power.npz")
    a = axs[0]
    a.plot(z["t"], z["P"], "C3", label="zero power: 1 W -> 1.1 W setpoint")
    a.plot(a4["t"], a4["P"], "C0", label="2 MW -> 2.2 MW, with feedback")
    a.axhline(1.1, color="C3", ls=":", lw=1)
    a.set(xlabel="time [s]", ylabel="power / initial", title="Automatic control: setpoint +10 %")
    a.legend(fontsize=8)
    a2 = a.twinx()
    a2.plot(z["t"], z["reg"], "C3--", lw=0.8)
    a2.plot(a4["t"], a4["reg"], "C0--", lw=0.8)
    a2.set_ylabel("regulating rod [cm] (dashed)")
    a = axs[1]
    a.plot(ts, ref["P"], "k", lw=2.5, label="reference (extrapolated direct)")
    for n, c in (("pcqs_1.0", "C3"), ("pcqs_0.5", "C1")):
        a.plot(d[n]["t"], d[n]["P"], "o", ms=3, color=c, label=f"PCQS dt {n[5:]} s")
    a.set(xlabel="time [s]", ylabel="power [MW]", title="1 MW, setpoint +5 % at 2 s")
    a.legend(fontsize=8)
    a = axs[2]
    for n, c in (("direct_4h", "C4"), ("direct_h", "C2"), ("pcqs_1.0", "C3"), ("pcqs_0.5", "C1")):
        t = d[n]["t"]
        j = np.searchsorted(t, ts - 1e-7)
        e = np.abs(d[n]["P"][j] / ref["P"] - 1)
        a.semilogy(ts[1:], np.maximum(e[1:], 1e-16), "o-", ms=2.5, lw=1, color=c, label=n)
    a.set(xlabel="time [s]", ylabel="|relative power error|", title="Error vs reference at sample times")
    a.legend(fontsize=8)
    for a in axs:
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"    saved {out}")


if __name__ == "__main__":
    args, jobs = pop_jobs(sys.argv[1:])
    args = args or ["worth", "inhour", "zero", "power"]
    if args[0] == "run":            # run [NAME ...]: all reference runs if no names, in parallel
        run_parallel(timed_run, args[1:] or list(RUNS), jobs)
        sys.exit()
    if "calibrate" in args:
        calibrate()
    if "worth" in args:
        test_worth()
    if "inhour" in args:
        test_inhour()
    if "zero" in args:
        test_zero_power()
    if "power" in args:
        test_at_power()
    if "compare" in args:
        compare()
    print("\nControl verification finished.")
