"""
verify_kinetics.py -- checks the time-dependent solvers in kinetics3d.

[1] Steady state: with nothing moving, power must stay constant (all methods).
[2] Exact separable reference: a bare homogeneous cube whose absorption changes uniformly
    in time. The flux stays exactly (time function) x (fundamental mesh mode), so the 3D
    problem reduces to 8 ODEs (2 groups + 6 precursors) that can be solved to 1e-10.
    The direct solver must converge to it at first order in dt; the quasi-static solvers
    must be accurate even with large steps.
[3] MITR-II-like core on a coarse 4 cm mesh: blade withdrawal, hold, scram. Quasi-static
    methods and a coarse direct run are compared against a fine-step direct solution.
"""
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp

from core_builder import build, load_config
from parallel import pop_jobs, run_parallel
from diffusion3d import FACES, Mesh, Problem
from kinetics3d import BladeProgram, KineticsData, TransientSolver

H0 = float(load_config("mitr2_like.yaml")["transient"]["start_height"])   # critical bank height, cm

KIN = KineticsData()


# ----------------------------------------------------------------------------- [2] cube
class CubeModel:
    """Minimal stand-in for CoreModel: uniform cube whose thermal absorption depends on the
    'blade' control variable h:  Sa2(h) = Sa2_0 * (1 - kappa * (h - 1))."""
    MAT = dict(D1=1.5, D2=0.4, Sa1=0.01, Sa2=0.08, Ss12=0.02, nSf1=0.004, nSf2=0.135)

    def __init__(self, a=120.0, n=12, kappa=0.006):
        self.mesh = Mesh(n, n, n, a / n, a / n, a / n)
        self.bc = {f: ("zero",) for f in FACES}
        self.kappa = kappa
        self.W = sp.csr_matrix(np.ones((self.mesh.n_cells, 1)))
        self.problem = self.problem_at(1.0)
        # discrete buckling of the fundamental mode: Kronecker sum of three 1D operators
        h = a / n
        T = (np.diag(np.full(n, 2.0)) - np.diag(np.ones(n - 1), 1) - np.diag(np.ones(n - 1), -1))
        T[0, 0] = T[-1, -1] = 3.0                      # phi = 0 at the face (half a cell out)
        self.B2 = 3 * np.linalg.eigvalsh(T / h**2)[0]

    def sa2(self, h):
        return self.MAT["Sa2"] * (1 - self.kappa * (h - 1.0))

    def problem_at(self, h):
        xs = {k: np.full(self.mesh.shape, v, float) for k, v in self.MAT.items()}
        xs["Sa2"][:] = self.sa2(h)
        return Problem(self.mesh, xs, self.bc)

    def k_exact(self, h):
        m, B2 = self.MAT, self.B2
        r1, r2 = m["D1"] * B2 + m["Sa1"] + m["Ss12"], m["D2"] * B2 + self.sa2(h)
        return m["nSf1"] / r1 + m["Ss12"] * m["nSf2"] / (r1 * r2)


def cube_reference(model, prog, t_end, k0):
    """Solve the 8 separable ODEs exactly, splitting at every kink in the control motion."""
    m, B2, k = model.MAT, model.B2, KIN
    r1 = m["D1"] * B2 + m["Sa1"] + m["Ss12"]

    def rhs(t, y):
        T1, T2, c = y[0], y[1], y[2:]
        S = (m["nSf1"] * T1 + m["nSf2"] * T2) / k0
        r2 = m["D2"] * B2 + model.sa2(prog.position(t))
        dT1 = k.v1 * (-r1 * T1 + (1 - k.beta) * S + k.lam_i @ c)
        dT2 = k.v2 * (m["Ss12"] * T1 - r2 * T2)
        return np.concatenate(([dT1, dT2], k.beta_i * S - k.lam_i * c))

    T1 = 1.0
    T2 = m["Ss12"] / (m["D2"] * B2 + model.sa2(prog.h0)) * T1
    S0 = (m["nSf1"] * T1 + m["nSf2"] * T2) / k0
    y = np.concatenate(([T1, T2], k.beta_i * S0 / k.lam_i))
    knots = [0.0] + prog.breakpoints(0.0, t_end) + [t_end]
    ts, Ps = [], []
    for a, b in zip(knots[:-1], knots[1:]):
        sol = solve_ivp(rhs, (a, b), y, method="Radau", rtol=1e-11, atol=1e-14, dense_output=True)
        y = sol.y[:, -1]
        ts.append(sol)
    def power(t):
        for sol in ts:
            if sol.t[0] - 1e-12 <= t <= sol.t[-1] + 1e-12:
                T1, T2 = sol.sol(t)[:2]
                return (m["nSf1"] * T1 + m["nSf2"] * T2) / k0 / S0
        raise ValueError(t)
    return np.vectorize(power)


def cube_program():
    # control variable: 1 = critical; ramp to 2 (positive reactivity), hold, then "scram" to 0
    prog = BladeProgram(h0=1.0, drive_speed=2.0, scram_speed=20.0, travel=2.0)
    prog.command(0.5, 2.0)
    prog.scram(4.0)
    return prog


def test_cube():
    print("[2] Bare cube vs exact separable solution")
    model = CubeModel()
    prog = cube_program()
    t_end = 6.0
    k0_exact = model.k_exact(1.0)
    rho_max = 1 - k0_exact / model.k_exact(2.0)
    rho_min = 1 - k0_exact / model.k_exact(0.0)
    print(f"    k0 exact = {k0_exact:.7f};  reactivity range {rho_min/KIN.beta:+.2f} $ .. {rho_max/KIN.beta:+.2f} $")
    ref = cube_reference(model, prog, t_end, k0_exact)

    probe = np.array([2.0, 3.9, 6.0])           # fixed times for the convergence-order check

    def err(hist):
        t, P = np.array(hist.t), np.array(hist.power)
        Pref = ref(t)
        at_probe = np.abs(np.interp(probe, t, P) / ref(probe) - 1)
        return np.max(np.abs(P / Pref - 1)), at_probe, t, P

    results = {}
    runs = [("direct", dict(dt_max=0.02, dh_max=0.04)), ("direct", dict(dt_max=0.01, dh_max=0.02)),
            ("direct", dict(dt_max=0.005, dh_max=0.01)),
            ("pcqs", dict(dt_max=0.5, dh_max=0.25)), ("pcqs", dict(dt_max=0.1, dh_max=0.05)),
            ("adiabatic", dict(dt_max=0.1, dh_max=0.05))]
    for method, kw in runs:
        t0 = time.time()
        s = TransientSolver(model, cube_program(), method=method, tol=1e-9, **kw)
        if method == "direct" and abs(s.k0 - k0_exact) > 1e-8:
            raise AssertionError(f"k0 {s.k0} vs exact {k0_exact}")
        s.run(t_end)
        e, e_probe, t, P = err(s.hist)
        results[(method, kw["dt_max"])] = (e, e_probe)
        print(f"    {method:9s} dt_max={kw['dt_max']:<6} steps={len(t) - 1:4d}  max rel. power error = {e:.2e}"
              f"   ({time.time() - t0:.1f} s)")
    d = np.array([results[("direct", h)][1] for h in (0.02, 0.01, 0.005)])
    order = np.log2(d[:-1] / d[1:])
    print(f"    direct method convergence order in dt at t = {probe} s:\n      {np.round(order, 2).tolist()} (expect 1)")
    print("    (the peak error, during the scram drop, is still pre-asymptotic at these steps)")
    assert np.all(order > 0.9), "direct solver not first order"
    assert results[("pcqs", 0.5)][0] < results[("direct", 0.005)][0] / 100, "PCQS should beat direct by far"
    assert results[("pcqs", 0.1)][0] < 1e-4
    return results


# ----------------------------------------------------------------------------- [1] steady
def test_steady(model):
    print("[1] Steady state (nothing moves)")
    for method in ("direct", "pcqs", "adiabatic"):
        s = TransientSolver(model, BladeProgram(H0, 0.2, 60, 53), method=method, dt_max=1.0)
        s.run(3.0)
        dev = np.max(np.abs(np.array(s.hist.power) - 1))
        print(f"    {method:9s}: max |P - 1| over 3 s = {dev:.1e}")
        assert dev < 1e-6


# ----------------------------------------------------------------------------- [3] MITR
MITR_T_END = 6.0
MITR_RUNS = {   # name: (method, dt_max, dh_max)
    "direct_h": ("direct", 0.005, 0.05),      # three nested step sizes -> Richardson reference
    "direct_2h": ("direct", 0.01, 0.1),
    "direct_4h": ("direct", 0.02, 0.2),
    "direct_coarse": ("direct", 0.1, 0.5),
    "pcqs_1.0": ("pcqs", 1.0, 1.0),
    "pcqs_0.25": ("pcqs", 0.25, 0.25),
    "adiabatic_0.25": ("adiabatic", 0.25, 0.25),
}


def mitr_model():
    cfg = load_config("mitr2_like.yaml")
    cfg["mesh"]["cell_size"] = 4.0
    return build(cfg)


def mitr_program(h_crit=H0):
    prog = BladeProgram(h0=h_crit, drive_speed=0.2, scram_speed=60.0, travel=53.0)
    prog.command(0.5, h_crit + 0.4)          # 0.4 cm withdrawal at drive speed (2 s)
    prog.scram(4.0)
    return prog


def mitr_run(name, model):
    method, dt, dh = MITR_RUNS[name]
    t0 = time.time()
    s = TransientSolver(model, mitr_program(), method=method, dt_max=dt, dh_max=dh)
    s.run(MITR_T_END)
    np.savez(f"vk_{name}.npz", t=s.hist.t, P=s.hist.power, rho=s.hist.rho, wall=time.time() - t0)
    print(f"    {name:15s} {len(s.hist.t) - 1:5d} steps  {time.time() - t0:6.0f} s", flush=True)


def _mitr_run_one(name):
    """One MITR run in its own process (parallel.run_parallel)."""
    mitr_run(name, mitr_model())


def mitr_compare():
    print("[3] MITR-II-like core, 4 cm mesh: withdraw 0.4 cm at 0.5 s, scram at 4 s")
    d = {n: np.load(f"vk_{n}.npz") for n in MITR_RUNS}
    t = d["direct_4h"]["t"]                               # coarsest of the nested set
    Ph = np.interp(t, d["direct_h"]["t"], d["direct_h"]["P"])
    P2 = np.interp(t, d["direct_2h"]["t"], d["direct_2h"]["P"])
    P4 = d["direct_4h"]["P"]
    ref2 = (8 * Ph - 6 * P2 + P4) / 3                    # cancels O(dt) and O(dt^2) errors
    ref1 = 2 * Ph - P2
    ref_unc = np.max(np.abs(ref1 / ref2 - 1))
    print(f"    reference: 2nd-order Richardson extrapolation of direct runs (dt = 5, 10, 20 ms);")
    print(f"    estimated reference uncertainty {ref_unc:.1e} (1st- vs 2nd-order extrapolation)")
    print("    errors are evaluated only at time points shared exactly with the reference grid")
    print("    (no interpolation: power falls 6x in 0.1 s during the scram, so interpolating")
    print("    between grid points would itself create percent-level 'errors')")
    t_scram, t_in = 4.0, 4.0 + (H0 + 0.4) / 60.0
    errs = {}
    for n in ("direct_coarse", "direct_4h", "direct_h", "pcqs_1.0", "pcqs_0.25", "adiabatic_0.25"):
        tt, P = d[n]["t"], d[n]["P"]
        j = np.clip(np.searchsorted(t, tt), 0, len(t) - 1)
        j = np.where(np.abs(t[j] - tt) < 1e-7, j, np.where(np.abs(t[j - 1] - tt) < 1e-7, j - 1, -1))
        ok = j >= 0
        e = np.abs(P[ok] / ref2[j[ok]] - 1)
        tc = tt[ok]
        before = e[tc <= t_scram].max()
        drop = e[(tc > t_scram) & (tc <= t_in)].max()
        after = e[tc > t_in].max()
        errs[n] = max(before, drop, after)
        method, dt, dh = MITR_RUNS[n]
        print(f"    {method:9s} dt_max={dt:<6} {len(tt) - 1:5d} steps | max rel. power error: "
              f"before scram {before:.1e}, during drop {drop:.1e}, after {after:.1e}"
              f"  ({float(d[n]['wall']):.0f} s, {ok.sum()} points)")
    np.savez("verify_kinetics_mitr.npz", t_ref=t, P_ref=ref2,
             **{f"{n}_t": d[n]["t"] for n in MITR_RUNS}, **{f"{n}_P": d[n]["P"] for n in MITR_RUNS})
    plot_mitr()
    assert errs["pcqs_0.25"] < errs["direct_h"], "PCQS (0.25 s) should beat direct (5 ms)"
    assert errs["pcqs_1.0"] < errs["direct_4h"], "PCQS (1 s) should beat direct (20 ms)"
    return errs


def plot_mitr(npz="verify_kinetics_mitr.npz", out="verify_kinetics_mitr.png"):
    import matplotlib.pyplot as plt
    d = np.load(npz)
    tr, Pr = d["t_ref"], d["P_ref"]
    runs = [("direct_coarse", "direct, dt 0.1 s", "C0"), ("direct_h", "direct, dt 5 ms", "C4"),
            ("pcqs_1.0", "PCQS, dt 1 s", "C3"), ("pcqs_0.25", "PCQS, dt 0.25 s", "C1"),
            ("adiabatic_0.25", "adiabatic, dt 0.25 s", "C2")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    a1.semilogy(tr, Pr, "k", lw=2.5, label="reference (extrapolated direct)")
    for n, lab, c in runs:
        t, P = d[f"{n}_t"], d[f"{n}_P"]
        a1.semilogy(t, P, "o-", ms=2.5, lw=1, color=c, label=lab)
        j = np.clip(np.searchsorted(tr, t), 1, len(tr) - 1)
        j = np.where(np.abs(tr[j] - t) < 1e-7, j, np.where(np.abs(tr[j - 1] - t) < 1e-7, j - 1, -1))
        ok = (j >= 0) & (t > 0)                         # only time points shared exactly
        a2.semilogy(t[ok], np.abs(P[ok] / Pr[j[ok]] - 1), "o-", ms=2.5, lw=1, color=c, label=lab)
    a1.set(xlabel="time [s]", ylabel="power / initial",
           title="Coarse MITR-II-like core: withdraw 0.4 cm, hold, scram at 4 s")
    a2.set(xlabel="time [s]", ylabel="|relative error| vs reference", title="Error of each method (at time points shared with the reference)")
    for a in (a1, a2):
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)


if __name__ == "__main__":
    # python3 verify_kinetics.py                 -> everything, sequentially
    # python3 verify_kinetics.py cube steady     -> fast tests only
    # python3 verify_kinetics.py run [NAME ...] -> MITR runs (all if no names), in parallel
    # python3 verify_kinetics.py compare         -> compare saved MITR runs and plot
    # --jobs N: processes at a time (default: performance cores, or REACTOR_JOBS)
    args, jobs = pop_jobs(sys.argv[1:])
    args = args or ["cube", "steady", "mitr"]
    if args[0] == "run":
        run_parallel(_mitr_run_one, args[1:] or list(MITR_RUNS), jobs)
        sys.exit()
    if "cube" in args:
        test_cube()
    if "steady" in args:
        test_steady(mitr_model())
    if "mitr" in args:
        run_parallel(_mitr_run_one, list(MITR_RUNS), jobs)
    if "mitr" in args or "compare" in args:
        mitr_compare()
    print("\nKinetics verification finished.")
