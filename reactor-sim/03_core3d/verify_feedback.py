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

    python3 verify_feedback.py                 # [1]-[3] and [5] (~5 min)
    python3 verify_feedback.py run NAME [...]  # [4] individual runs (to run in parallel)
    python3 verify_feedback.py compare         # [4] compare and plot
"""
import sys
import time

import numpy as np

from core_builder import build, load_config
from diffusion3d import Problem, solve_k
from kinetics3d import BladeProgram, TransientSolver, _Ops, kinetics_from_config
from thermal import ChannelThermal


def setup():
    cfg = load_config("mitr2_like.yaml")
    cfg["mesh"]["cell_size"] = 4.0
    model = build(cfg)
    return cfg, model, kinetics_from_config(cfg)


def at_power(cfg, model, kin, P, h=25.9, method="pcqs", **kw):
    th = ChannelThermal(model, cfg)
    return TransientSolver(model, BladeProgram(h, 0.2, 60, 53), kin=kin, method=method,
                           thermal=th, power_w=P, **kw)


# ----------------------------------------------------------------------------- [1]
def test_sensitivity(cfg, model, kin):
    print("[1] Reactivity sensitivities vs exact recalculation")
    s = at_power(cfg, model, kin, 5e6)
    th = s.thermal
    prob = s._problem(25.9)
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
        ops = _Ops(Problem(prob.mesh, th.apply(model.problem_at(25.9).xs, Tf0 + dTf, Tc0 + dTc), prob.bc),
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
    prog = BladeProgram(25.9, 0.2, 60, 53)
    prog.command(1.0, 26.7)                      # +0.8 cm over 4 s
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
    prog = BladeProgram(25.9, 0.2, 60, 53)
    prog.command(1.0, 26.0)
    th = ChannelThermal(model, cfg)
    t0 = time.time()
    s = TransientSolver(model, prog, kin=kin, method="pcqs", dt_max=20.0, dh_max=0.5,
                        thermal=th, power_w=2e6)
    s.run(1500.0)
    P_end = s.hist.power[-1] * 2e6
    slope = (s.hist.power[-1] - s.hist.power[-6]) / (s.hist.t[-1] - s.hist.t[-6]) * 2e6
    print(f"    transient: {len(s.hist.t) - 1} steps ({time.time() - t0:.0f} s); "
          f"P(1500 s) = {P_end / 1e6:.4f} MW, still changing by {slope:.2f} W/s")
    P_static = static_power_for_criticality(cfg, model, kin, 26.0, s.k0, 2e6, 3e6)
    rel = abs(P_end / P_static - 1)
    print(f"    static search: eigenvalue with steady temperatures returns to k0 at "
          f"{P_static / 1e6:.4f} MW;  difference {rel:.1e}")
    assert rel < 2e-3


if __name__ == "__main__":
    args = sys.argv[1:] or ["unit", "equilibrium"]
    cfg, model, kin = setup()
    if args[0] == "run":
        for n in args[1:]:
            transient(n, cfg, model, kin)
        sys.exit()
    if "unit" in args:
        test_sensitivity(cfg, model, kin)
        test_thermal(cfg, model)
        test_steady(cfg, model, kin)
    if "equilibrium" in args:
        test_equilibrium(cfg, model, kin)
    if "compare" in args:
        compare()
    print("\nFeedback verification finished.")
