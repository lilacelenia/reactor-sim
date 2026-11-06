"""
verify_feedback.py -- checks the thermal-hydraulics and feedback coupling (4 cm mesh).

[1] Reactivity sensitivities: the per-node d(rho)/dT used inside the amplitude equations
    must predict the exact reactivity change for a random temperature perturbation.
[2] Channel thermal model: backward Euler must conserve energy to round-off and relax to
    the exact steady-state solution.
[3] Steady state at power: nothing changes -> power and temperatures stay constant.
[4] Coupled transient (5 MW, 0.8 cm withdrawal): quasi-static methods vs a
    Richardson-extrapolated direct reference. A deliberately strong test: power triples.
[5] Feedback equilibrium: after a small withdrawal from 2 MW the transient settles; its
    final power must match an independent static search for the power at which the
    eigenvalue with steady-state temperatures returns to critical.

Loss of flow (decay heat, pump coastdown, natural circulation):
[6] Decay heat: the group equations reproduce the ANS-5.1 closed form for finite operation
    followed by shutdown; the standard's shutdown fractions are compared with published MITR
    analyses (independent codes).
[7] Pump coastdown inside the transient solver (1 W, nothing else changes): pump speed vs
    the exact tan / hyperbolic solutions, and core flow vs the exact Riccati solution of the
    loop momentum equation (no friction torque, so a closed form exists).
[8] Natural circulation: pumps tripped at constant fission power (feedback coefficients
    zeroed so power cannot change); the transient settles at the flow and temperatures of an
    independent algebraic steady-state solution. Also compared with the published MITR
    natural-convection result the NCV resistance was calibrated to.
[9] Loss-of-flow accident at 6 MW (pump trip at 1 s, scram at 2 s, to 30 s): PCQS vs a
    Richardson-extrapolated direct reference at shared time points (power, heat
    including decay, flow, hottest plate); energy conservation of every run.

    python3 verify_feedback.py                 # [1]-[3] and [5] (~5 min)
    python3 verify_feedback.py lof             # [6]-[8] (~10 min)
    python3 verify_feedback.py run [NAME ...] [--jobs N]  # [4], [9] reference runs (all if no
                                                         # names), several processes at once
    python3 verify_feedback.py compare         # [4] compare and plot
    python3 verify_feedback.py compare_lof     # [9] compare and plot
"""
import sys
import time

import numpy as np

from core_builder import build, load_config
from parallel import pop_jobs, run_parallel
from diffusion3d import Problem, solve_k
from kinetics3d import BladeProgram, PlantProgram, TransientSolver, _Ops, kinetics_from_config
from thermal import ChannelThermal, DecayHeat

H0 = float(load_config("mitr2_like.yaml")["transient"]["start_height"])   # critical bank height, cm


def _run_one(name):
    """One reference run in its own process (parallel.run_parallel)."""
    cfg, model, kin = setup()
    (lof_transient if name in LOF_RUNS else transient)(name, cfg, model, kin)


def setup():
    cfg = load_config("mitr2_like.yaml")
    cfg["mesh"]["cell_size"] = 4.0
    model = build(cfg)
    return cfg, model, kinetics_from_config(cfg)


def at_power(cfg, model, kin, P, h=H0, method="pcqs", **kw):
    th = ChannelThermal(model, cfg)
    return TransientSolver(model, BladeProgram(h, 0.2, 60, 53), kin=kin, method=method,
                           thermal=th, power_w=P, **kw)


# ----------------------------------------------------------------------------- [1]
def test_sensitivity(cfg, model, kin):
    print("[1] Reactivity sensitivities vs exact recalculation")
    s = at_power(cfg, model, kin, 5e6)
    th = s.thermal
    prob = s._problem(H0)
    a_f, a_c, rho_s = th.rho_sensitivity(prob.xs, s.k0, s.w1, s.w2, s.psi1, s.psi2, prob.bc)
    print(f"    reactivity from the sensitivity routine {rho_s:+.2e} vs solver {s.rho:+.2e} (both ~0)")
    assert abs(rho_s - s.rho) < 1e-9
    rng = np.random.default_rng(1)
    Tf0, Tc0 = th.Tf.copy(), th.Tc.copy()
    print(f"    total d rho/dT: coolant {a_c.sum() * 1e5:+.2f} pcm/K, fuel {a_f.sum() * 1e5:+.3f} pcm/K "
          f"(uniform rise, adjoint-weighted)")
    for amp in (1.0, 0.1):
        dTf = amp * rng.uniform(-1, 1, th.n) * 3
        dTc = amp * rng.uniform(-1, 1, th.n)
        pred = a_f @ dTf + a_c @ dTc
        ops = _Ops(Problem(prob.mesh, th.apply(model.problem_at(H0).xs, Tf0 + dTf, Tc0 + dTc), prob.bc),
                   s.k0, kin)
        exact, _ = s._pk_params(ops, s.psi1, s.psi2)
        exact -= s.rho
        rel = abs(pred / exact - 1)
        print(f"    random dT ~ {amp:4.1f} K: predicted {pred * 1e5:+.4f} pcm, exact {exact * 1e5:+.4f} pcm, "
              f"rel. diff {rel:.1e}")
        assert rel < 2e-3 * amp + 1e-6, "sensitivity mismatch"


# ----------------------------------------------------------------------------- [2]
def test_thermal(cfg, model):
    print("[2] Channel thermal model")
    th = ChannelThermal(model, cfg)
    rng = np.random.default_rng(2)
    q = rng.uniform(0.5, 1.5, th.n) * 6e6 / th.n
    E0, inflow, out = th.stored_energy(), 0.0, 0.0
    dt = 0.05
    for _ in range(400):                         # 20 s >> thermal time constants (~0.3 s)
        th.step_be(dt, q)
        inflow += q.sum() * dt
        out += th.outflow_power() * dt
    err = abs(inflow - (th.stored_energy() - E0) - out) / inflow
    Tf, Tc = th.steady_state(q)
    dT = max(np.abs(Tf - th.Tf).max(), np.abs(Tc - th.Tc).max())
    print(f"    energy balance error over 400 steps: {err:.1e} (relative)")
    print(f"    after 20 s at constant power: max |T - exact steady state| = {dT:.1e} K")
    print(f"    6 MW, nonuniform: core outlet {Tc.reshape(th.ne, th.nl)[:, -1].mean() - 273.15:.2f} C "
          f"= inlet + P/(m cp) = {th.T_in + q.sum() / (th.mcp * th.ne) - 273.15:.2f} C")
    assert err < 1e-12 and dT < 1e-8


# ----------------------------------------------------------------------------- [3]
def test_steady(cfg, model, kin):
    print("[3] Steady state at 5 MW")
    for method in ("direct", "pcqs"):
        s = at_power(cfg, model, kin, 5e6, method=method, dt_max=1.0)
        T0 = s.thermal.Tf.copy()
        s.run(5.0)
        dP = np.max(np.abs(np.array(s.hist.power) - 1))
        dT = np.abs(s.thermal.Tf - T0).max()
        print(f"    {method:7s}: max |P/P0 - 1| = {dP:.1e}, max plate temperature drift {dT:.1e} K")
        assert dP < 1e-6 and dT < 1e-4


# ----------------------------------------------------------------------------- [4]
RUNS = {
    "direct_h": ("direct", 0.01),
    "direct_2h": ("direct", 0.02),
    "direct_4h": ("direct", 0.04),
    "pcqs_1.0": ("pcqs", 1.0),
    "pcqs_0.25": ("pcqs", 0.25),
    "adiabatic_0.25": ("adiabatic", 0.25),
    "nofb_pcqs_0.25": ("pcqs", 0.25),
}
T_END = 20.0


def transient(name, cfg, model, kin):
    method, dt = RUNS[name]
    prog = BladeProgram(H0, 0.2, 60, 53)
    prog.command(1.0, H0 + 0.8)                  # +0.8 cm over 4 s
    th = None if name.startswith("nofb") else ChannelThermal(model, cfg)
    t0 = time.time()
    s = TransientSolver(model, prog, kin=kin, method=method, dt_max=dt, dh_max=10.0,
                        dlnp_max=0.0 if method == "direct" else 0.25, thermal=th, power_w=5e6)
    s.run(T_END)
    h = s.hist.arrays()
    np.savez(f"vf_{name}.npz", t=h["t"], P=h["power"], wall=time.time() - t0,
             **({"Tf": h["fuel_max"], "Tout": h["coolant_outlet"], "rho_fb": h["rho_feedback"]} if th else {}))
    print(f"    {name:15s} {len(h['t']) - 1:5d} steps {time.time() - t0:6.0f} s", flush=True)


def compare():
    print("[4] Coupled transient: 5 MW, withdraw 0.8 cm over 1-5 s (power triples)")
    d = {n: np.load(f"vf_{n}.npz") for n in RUNS}
    t = d["direct_4h"]["t"]
    Ph = np.interp(t, d["direct_h"]["t"], d["direct_h"]["P"])
    P2 = np.interp(t, d["direct_2h"]["t"], d["direct_2h"]["P"])
    ref = (8 * Ph - 6 * P2 + d["direct_4h"]["P"]) / 3
    unc = np.max(np.abs((2 * Ph - P2) / ref - 1))
    Tref = (8 * np.interp(t, d["direct_h"]["t"], d["direct_h"]["Tf"])
            - 6 * np.interp(t, d["direct_2h"]["t"], d["direct_2h"]["Tf"]) + d["direct_4h"]["Tf"]) / 3
    print(f"    reference: 2nd-order Richardson of direct runs (dt = 10, 20, 40 ms), "
          f"estimated uncertainty {unc:.1e}")
    errs = {}
    for n in ("direct_4h", "direct_h", "pcqs_1.0", "pcqs_0.25", "adiabatic_0.25"):
        tt = d[n]["t"]
        j = np.searchsorted(t, tt - 1e-7)
        ok = (j < len(t)) & (np.abs(t[np.minimum(j, len(t) - 1)] - tt) < 1e-7)
        eP = np.max(np.abs(d[n]["P"][ok] / ref[j[ok]] - 1))
        eT = np.max(np.abs(d[n]["Tf"][ok] - Tref[j[ok]]))
        errs[n] = eP
        print(f"    {RUNS[n][0]:9s} dt_max={RUNS[n][1]:<5} {len(tt) - 1:5d} steps: max rel. power error "
              f"{eP:.1e}, hottest-plate temperature error {eT:.1e} K  ({float(d[n]['wall']):.0f} s)")
    print(f"    final power: with feedback {ref[-1] * 5:.2f} MW, without {d['nofb_pcqs_0.25']['P'][-1] * 5:.2f} MW")
    plot(d, t, ref)
    assert errs["pcqs_0.25"] < errs["direct_h"]
    return errs


def plot(d, t, ref, out="verify_feedback.png"):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    a = axs[0]
    a.plot(t, ref * 5, "k", lw=2.5, label="reference (extrapolated direct)")
    for n, lab, c in (("pcqs_1.0", "PCQS, dt 1 s", "C3"), ("pcqs_0.25", "PCQS, dt 0.25 s", "C1"),
                      ("adiabatic_0.25", "adiabatic, dt 0.25 s", "C2"),
                      ("nofb_pcqs_0.25", "no feedback (PCQS)", "C7")):
        a.plot(d[n]["t"], d[n]["P"] * 5, "o-" if "nofb" not in n else "--", ms=2.5, lw=1, color=c, label=lab)
    a.set(xlabel="time [s]", ylabel="power [MW]", title="5 MW, withdraw 0.8 cm over 1-5 s", ylim=(0, 25))
    a.legend(fontsize=8)
    a = axs[1]
    for n, lab, c in (("direct_h", "direct, dt 10 ms", "C4"), ("pcqs_1.0", "PCQS, dt 1 s", "C3"),
                      ("pcqs_0.25", "PCQS, dt 0.25 s", "C1"), ("adiabatic_0.25", "adiabatic, dt 0.25 s", "C2")):
        tt = d[n]["t"]
        j = np.searchsorted(t, tt - 1e-7)
        ok = (j < len(t)) & (np.abs(t[np.minimum(j, len(t) - 1)] - tt) < 1e-7) & (tt > 0)
        a.semilogy(tt[ok], np.abs(d[n]["P"][ok] / ref[j[ok]] - 1), "o-", ms=2.5, lw=1, color=c, label=lab)
    a.set(xlabel="time [s]", ylabel="|relative power error|", title="Error vs reference")
    a.legend(fontsize=8)
    a = axs[2]
    h = d["pcqs_0.25"]
    a.plot(h["t"], h["Tf"] - 273.15, label="hottest plate")
    a.plot(h["t"], h["Tout"] - 273.15, label="core outlet (mixed)")
    a2 = a.twinx()
    a2.plot(h["t"], h["rho_fb"] * 1e5, "C3--", label="feedback reactivity")
    a2.set_ylabel("feedback reactivity [pcm]", color="C3")
    a.set(xlabel="time [s]", ylabel="temperature [C]", title="Temperatures and feedback (PCQS)")
    a.legend(loc="upper left", fontsize=8)
    for a in axs:
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=110)


# ----------------------------------------------------------------------------- [5]
def static_power_for_criticality(cfg, model, kin, h, k_target, P_lo, P_hi):
    """Independent static calculation: power P such that the eigenvalue with steady-state
    temperatures at P equals k_target (secant on P, Picard on temperatures)."""
    th = ChannelThermal(model, cfg)
    cache = {}

    def k_of(P):
        r = None
        for _ in range(50):
            prob = Problem(model.problem.mesh, th.apply(model.problem_at(h).xs), model.problem.bc)
            r = solve_k(prob, tol_k=1e-11, tol_src=1e-10, guess=r)
            S = (prob.xs["nSf1"] * r.phi1 + prob.xs["nSf2"] * r.phi2).ravel()
            Tf, Tc = th.steady_state(th.node_power(S, P / S.sum()))
            dT = np.abs(Tc - th.Tc).max() + np.abs(Tf - th.Tf).max()
            th.Tf, th.Tc = Tf, Tc
            if dT < 1e-8:
                break
        cache[P] = r.k
        return r.k

    a, b = P_lo, P_hi
    ka, kb = k_of(a), k_of(b)
    for _ in range(20):
        c = b + (k_target - kb) * (b - a) / (kb - ka)
        kc = k_of(c)
        a, ka, b, kb = b, kb, c, kc
        if abs(kc - k_target) < 1e-9:
            break
    return c


def test_equilibrium(cfg, model, kin):
    print("[5] Feedback equilibrium: 2 MW, withdraw 0.1 cm at 1 s, run to 1500 s")
    # decay heat off: its long-lived groups need days to come to equilibrium at a new power,
    # so after 1500 s the heat deposited is still below the fission power
    cfg = {k: v for k, v in cfg.items() if k != "decay_heat"}
    print("    (decay heat off for this check; flow from the primary-loop model)")
    prog = BladeProgram(H0, 0.2, 60, 53)
    prog.command(1.0, H0 + 0.1)
    th = ChannelThermal(model, cfg)
    t0 = time.time()
    s = TransientSolver(model, prog, kin=kin, method="pcqs", dt_max=20.0, dh_max=0.5,
                        thermal=th, power_w=2e6)
    s.run(1500.0)
    P_end = s.hist.power[-1] * 2e6
    slope = (s.hist.power[-1] - s.hist.power[-6]) / (s.hist.t[-1] - s.hist.t[-6]) * 2e6
    print(f"    transient: {len(s.hist.t) - 1} steps ({time.time() - t0:.0f} s); "
          f"P(1500 s) = {P_end / 1e6:.4f} MW, still changing by {slope:.2f} W/s")
    P_static = static_power_for_criticality(cfg, model, kin, H0 + 0.1, s.k0, 2e6, 3e6)
    rel = abs(P_end / P_static - 1)
    print(f"    static search: eigenvalue with steady temperatures returns to k0 at "
          f"{P_static / 1e6:.4f} MW;  difference {rel:.1e}")
    assert rel < 2e-3


# ----------------------------------------------------------------------------- [6]
def test_decay_heat(cfg):
    print("[6] Decay heat (ANS-5.1-1979, U-235, 23 groups)")
    dh = DecayHeat(cfg["decay_heat"], 1)
    print(f"    equilibrium decay heat / fission power: {dh.gamma:.4%}")
    # 30 days at 1 W, then shutdown, in uneven steps -> closed form F(t, T) / Q
    T_op = 30 * 86400.0
    dh.D[:] = 0.0
    t = 0.0
    for dt in np.geomspace(1.0, T_op / 20, 60):
        dt = min(dt, T_op - t)
        if dt <= 0:
            break
        dh.advance_constant(dt, np.array([1.0]))
        t += dt
    while t < T_op - 1e-6:
        dh.advance_constant(min(T_op / 20, T_op - t), np.array([1.0]))
        t = min(t + T_op / 20, T_op)
    errs = []
    t_s = 0.0
    for tt in (1.0, 16.0, 300.0, 3600.0, 86400.0):
        dh.advance_constant(tt - t_s, np.array([0.0]))
        t_s = tt
        exact = dh.fraction_after_shutdown(tt, T_op)
        errs.append(abs(dh.D.sum() / exact - 1))
    print(f"    30 days at power, then shutdown: group equations vs closed form at 1 s ... 1 day: "
          f"max rel. diff {max(errs):.1e}")
    assert max(errs) < 1e-9
    # published MITR analyses (different codes and histories)
    f16 = dh.fraction_after_shutdown(16.0)
    f300 = dh.fraction_after_shutdown(300.0, 4200 / 7.35 * 86400)
    print(f"    16 s after shutdown from long operation: {f16:.3%}; MITR 2003 analysis (DKPOWR): "
          f"325 kW / 7.4 MW = {325 / 7400:.3%}  (diff {f16 / (325 / 7400) - 1:+.1%})")
    print(f"    300 s after 4200 MWd at 7.35 MW: {f300:.3%}; LEU PSAR (ORIGEN-S): "
          f"186.2 kW / 7.35 MW = {186.2 / 7350:.3%}  (diff {f300 / (186.2 / 7350) - 1:+.1%})")


# ----------------------------------------------------------------------------- [7]
def coastdown_flow_exact(lp, t):
    """Exact core flow after all pumps trip at t = 0 (no friction torque, NCVs shut, no
    buoyancy): I dW/dt = dp0/(1 + t/tau)^2 - R W^2 with I = I_c + I_p, R = R_c + R_p.
    W = c/(1+x) + 1/v (x = t/tau), c the particular-solution root, v from a linear ODE."""
    I, R, tau, W0 = lp.I_c + lp.I_p, lp.R_c + lp.R_p, lp.tau_cd, lp.W0
    dp0 = R * W0**2
    c = (I / tau + np.sqrt((I / tau) ** 2 + 4 * R * dp0)) / (2 * R)
    m = 2 * R * c * tau / I
    x1 = 1.0 + np.asarray(t) / tau
    v0 = 1.0 / (W0 - c)
    v = x1**m * (v0 + R * tau / (I * (1 - m)) * (x1 ** (1 - m) - 1))
    return c / x1 + 1.0 / v


def test_coastdown(cfg, model, kin):
    print("[7] Pump coastdown in the transient solver (1 W; trip at 1 s)")
    for fric in (0.0, None):
        c = {**cfg, "primary_loop": dict(cfg["primary_loop"])}
        if fric is not None:
            c["primary_loop"]["pump_friction_ratio"] = fric
        for method in ("pcqs", "direct"):
            plant = PlantProgram()
            plant.set(1.0, "pumps", 0)
            th = ChannelThermal(model, c)
            s = TransientSolver(model, BladeProgram(H0, 0.2, 60, 53), kin=kin, method=method,
                                thermal=th, power_w=1.0, plant=plant, dt_max=2.0)
            s.run(40.0)
            lp = th.loop
            h = s.hist.arrays()
            after = h["t"] >= 1.0
            tt = h["t"][after] - 1.0
            e_w = np.max(np.abs(h["pump_speed"][after] - lp.coastdown_speed(tt)))
            msg = f"    f = {lp.f_fric:<4g} {method:6s} {after.sum():3d} steps: pump speed max error {e_w:.1e}"
            if lp.f_fric == 0.0:
                Wx = coastdown_flow_exact(lp, tt)
                e_f = np.max(np.abs(h["core_flow"][after] / Wx - 1))
                msg += f", core flow max rel. error {e_f:.1e} (flow {Wx[-1] / lp.W0:.1%} at 39 s)"
                assert e_f < (1e-6 if method == "pcqs" else 2e-2)
            else:
                msg += f" (pumps stop {lp.stop_time:.1f} s after the trip)"
            print(msg)
            assert e_w < 1e-6


# ----------------------------------------------------------------------------- [8]
def nc_steady_state(th, q):
    """Independent algebraic natural-circulation steady state for node heat q [W]:
    find W with  B - R_c W^2 = R_v W_v^2,  -Dp - B_p = R_p(W_p) W_p|W_p|,  W = W_v + W_p,
    temperatures from the exact steady channel solution at flow W."""
    from scipy.optimize import brentq
    lp = th.loop

    def parts(W):
        Tf, Tc = th.steady_state(q, flow=W / th.W0)
        B, Bp = lp.buoyancy(Tc, th.T_in)
        mdp = B - lp.R_c * W**2
        Wv = np.sqrt(max(mdp, 0.0) / lp.R_v)
        drive = mdp - Bp
        Wp = np.sign(drive) * np.sqrt(abs(drive) / (lp.R_p if drive >= 0 else lp.R_p * lp.k_rev))
        return W - Wv - Wp, Tf, Tc

    W = brentq(lambda w: parts(w)[0], 0.05, 20.0, xtol=1e-14, rtol=1e-14)
    _, Tf, Tc = parts(W)
    return W, Tf, Tc


def test_natural_circulation(cfg, model, kin):
    print("[8] Natural circulation: pumps tripped at constant fission power")
    c = {**cfg, "feedback": {**cfg["feedback"], "materials": {}}}     # power cannot change
    for P, T_in in ((1.0e5, None), (1.0e5, 333.15)):
        cc = {**c, "thermal_hydraulics": dict(c["thermal_hydraulics"])}
        if T_in is not None:
            cc["thermal_hydraulics"]["inlet_temperature"] = T_in
        plant = PlantProgram()
        plant.set(1.0, "pumps", 0)
        th = ChannelThermal(model, cc)
        s = TransientSolver(model, BladeProgram(H0, 0.2, 60, 53), kin=kin, method="pcqs",
                            thermal=th, power_w=P, plant=plant, dt_max=20.0)
        s.run(1200.0)
        q = th.heat(th.node_power(s.ops.fission(s.phi1, s.phi2), s.watt_per_fission))
        W, Tf, Tc = nc_steady_state(th, q)
        h = s.hist
        slope = abs(h.core_flow[-1] - h.core_flow[-4]) / (h.t[-1] - h.t[-4])
        eW = abs(th.loop.Wc / W - 1)
        eT = max(np.abs(th.Tf - Tf).max(), np.abs(th.Tc - Tc).max())
        out = Tc.reshape(th.ne, th.nl)[:, -1]
        print(f"    {P / 1e3:.0f} kW, inlet {th.T_in - 273.15:.0f} C: transient core flow {th.loop.Wc:.4f} kg/s "
              f"(NCVs {'open' if th.loop.open else 'shut'}, drift {slope:.1e} kg/s2) vs algebraic "
              f"{W:.4f} kg/s: rel. diff {eW:.1e}; temperatures {eT:.1e} K")
        print(f"        fuel-channel flow {W * th.loop.fuel_frac:.3f} kg/s, mixed outlet {out.mean() - 273.15:.1f} C, "
              f"hottest channel outlet {out.max() - 273.15:.1f} C, hottest plate {Tf.max() - 273.15:.1f} C")
        assert eW < 1e-4 and eT < 1e-2 and th.loop.open
    print("    published (LEU PSAR Table 4-18, NCVs only, 100 kW, 60 C inlet, MULCH-II): 1.35 kg/s "
          "through the core, hottest outlet 94.6 C")


# ----------------------------------------------------------------------------- [9]
LOF_RUNS = {"lof_direct_h": ("direct", 0.01), "lof_direct_2h": ("direct", 0.02),
            "lof_direct_4h": ("direct", 0.04), "lof_pcqs_0.4": ("pcqs", 0.4),
            "lof_pcqs_0.2": ("pcqs", 0.2)}
LOF_DROP = 0.44                    # s, blade drop from H0 (scram speed H0 / 0.44 s; published < 1 s)
LOF_END = 30.0
LOF_KEYS = ("power", "thermal_power", "core_flow", "fuel_max", "coolant_outlet_max", "decay_heat",
            "energy_in", "energy_out", "ncv_open")


def lof_transient(name, cfg, model, kin):
    method, dt = LOF_RUNS[name]
    # the drop time is a multiple of 40 ms and PCQS takes 44 steps of 10 ms during the drop,
    # so every run has its step ends on the reference grid
    prog = BladeProgram(H0, 0.2, H0 / LOF_DROP, 53)
    prog.scram(2.0)                              # ~1 s after the trip, as in the MITR analyses
    plant = PlantProgram()
    plant.set(1.0, "pumps", 0)
    th = ChannelThermal(model, cfg)
    t0 = time.time()
    s = TransientSolver(model, prog, kin=kin, method=method, thermal=th, power_w=6e6, plant=plant,
                        dt_max=dt, dlnp_max=0.0, dflow_max=0.0,
                        dh_max=H0 / 44 if method == "pcqs" else 100.0)
    E0 = th.stored_energy()
    s.run(LOF_END)
    h = s.hist.arrays()
    bal = (th.E_in - th.E_out - (th.stored_energy() - E0)) / th.E_in
    np.savez(f"vf_{name}.npz", t=h["t"], wall=time.time() - t0, balance=bal,
             **{k: h[k] for k in LOF_KEYS})
    print(f"    {name:15s} {len(h['t']) - 1:5d} steps {time.time() - t0:6.0f} s, energy balance {bal:.1e}",
          flush=True)


def compare_lof():
    print(f"[9] Loss of flow at 6 MW: pumps trip at 1 s, scram at 2 s, to {LOF_END:g} s")
    d = {n: np.load(f"vf_{n}.npz") for n in LOF_RUNS}
    t = d["lof_direct_4h"]["t"]

    def at(n, k):
        return np.interp(t, d[n]["t"], d[n][k])

    ref = {k: (8 * at("lof_direct_h", k) - 6 * at("lof_direct_2h", k) + d["lof_direct_4h"][k]) / 3
           for k in ("power", "thermal_power", "core_flow", "fuel_max")}
    unc = np.max(np.abs((2 * at("lof_direct_h", "power") - at("lof_direct_2h", "power")) / ref["power"] - 1))
    print(f"    reference: 2nd-order Richardson of direct runs (dt = 10, 20, 40 ms), estimated "
          f"power uncertainty {unc:.1e}")
    for n in LOF_RUNS:
        print(f"    {n:15s} energy balance (in - out - stored change) / in = {float(d[n]['balance']):.1e}")
        assert abs(float(d[n]["balance"])) < (1e-8 if "pcqs" in n else 1e-10)
    # the 0.44 s blade drop is reported separately: there the power falls 60 % in 40 ms and the
    # direct runs are not yet in their asymptotic range (40 -> 5 ms still changes it ~1 %), so the
    # extrapolated reference is not trustworthy inside the drop
    windows = (("before the scram", 0.0, 2.0 + 1e-9), ("blade drop", 2.0 + 1e-9, 2.0 + LOF_DROP - 1e-9),
               ("after the drop", 2.0 + LOF_DROP - 1e-9, LOF_END + 1.0))
    errs = {}
    for n in ("lof_direct_4h", "lof_direct_h", "lof_pcqs_0.4", "lof_pcqs_0.2"):
        tt = d[n]["t"]
        j = np.searchsorted(t, tt - 1e-7)
        ok = (j < len(t)) & (np.abs(t[np.minimum(j, len(t) - 1)] - tt) < 1e-7)
        print(f"    {n:15s} {len(tt) - 1:4d} steps ({ok.sum():4d} shared, {float(d[n]['wall']):.0f} s):")
        for wname, a, b in windows:
            sel = ok & (tt > a) & (tt <= b)
            jj = j[sel]
            e = {k: np.max(np.abs(d[n][k][sel] / ref[k][jj] - 1)) for k in ("power", "thermal_power", "core_flow")}
            eT = np.max(np.abs(d[n]["fuel_max"][sel] - ref["fuel_max"][jj]))
            errs[n, wname] = e
            print(f"        {wname:17s} max rel. error power {e['power']:.1e}, heat {e['thermal_power']:.1e}, "
                  f"flow {e['core_flow']:.1e}; hottest plate {eT:.1e} K")
    r = d["lof_direct_h"]
    i_open = np.argmax(r["ncv_open"]) if r["ncv_open"].any() else None
    print(f"    reference run: decay heat {r['decay_heat'][0] / 1e3:.0f} kW at 6 MW, "
          f"{np.interp(18.0, r['t'], r['decay_heat']) / 1e3:.0f} kW 16 s after the scram; "
          f"flow {np.interp(20.0, r['t'], r['core_flow']):.1f} kg/s when the pumps have stopped; NCVs open at "
          + (f"{r['t'][i_open]:.2f} s" if i_open is not None else "-")
          + f"; hottest plate {r['fuel_max'][0] - 273.15:.1f} C at start, max after the scram "
            f"{r['fuel_max'][r['t'] > 3].max() - 273.15:.1f} C")
    plot_lof(d, t, ref)
    for w in ("before the scram", "after the drop"):
        assert errs["lof_pcqs_0.2", w]["power"] < errs["lof_direct_h", w]["power"]
    return errs


def plot_lof(d, t, ref, out="verify_loss_of_flow.png"):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    a = axs[0]
    a.semilogy(t, ref["power"] * 6e6, "k", lw=2.5, label="fission power (reference)")
    a.semilogy(t, ref["thermal_power"], "C3", lw=1.5, label="heat to fuel incl. decay (reference)")
    h = d["lof_pcqs_0.2"]
    a.semilogy(h["t"], h["power"] * 6e6, "C1o", ms=2.5, label="fission power, PCQS 0.2 s")
    a.semilogy(h["t"], h["decay_heat"], "C2--", label="decay heat (PCQS)")
    a.set(xlabel="time [s]", ylabel="power [W]", title="Power: pumps trip at 1 s, scram at 2 s")
    a.legend(fontsize=8)
    a = axs[1]
    a.semilogy(t, ref["core_flow"], "k", lw=2.5, label="core flow (reference)")
    a.semilogy(h["t"], h["core_flow"], "C1o", ms=2.5, label="PCQS 0.2 s")
    a.set(xlabel="time [s]", ylabel="core flow [kg/s]", title="Coastdown and natural circulation")
    a.legend(fontsize=8)
    a = axs[2]
    a.plot(t, ref["fuel_max"] - 273.15, "k", lw=2.5, label="hottest plate (reference)")
    a.plot(h["t"], h["fuel_max"] - 273.15, "C1o", ms=2.5, label="hottest plate, PCQS 0.2 s")
    a.plot(h["t"], h["coolant_outlet_max"] - 273.15, "C0--", label="hottest channel outlet (PCQS)")
    a.set(xlabel="time [s]", ylabel="temperature [C]", title="Temperatures")
    a.legend(fontsize=8)
    for a in axs:
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=110)


if __name__ == "__main__":
    args, jobs = pop_jobs(sys.argv[1:])
    args = args or ["unit", "equilibrium"]
    if args[0] == "run":            # run [NAME ...]: all reference runs if no names, in parallel
        names = args[1:] or list(LOF_RUNS) + list(RUNS)
        run_parallel(_run_one, names, jobs)
        sys.exit()
    cfg, model, kin = setup()
    if "unit" in args:
        test_sensitivity(cfg, model, kin)
        test_thermal(cfg, model)
        test_steady(cfg, model, kin)
    if "equilibrium" in args:
        test_equilibrium(cfg, model, kin)
    if "lof" in args:
        test_decay_heat(cfg)
        test_coastdown(cfg, model, kin)
        test_natural_circulation(cfg, model, kin)
    if "compare" in args:
        compare()
    if "compare_lof" in args:
        compare_lof()
    print("\nFeedback verification finished.")
