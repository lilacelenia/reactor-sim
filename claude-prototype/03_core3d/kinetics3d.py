"""
kinetics3d.py -- time-dependent 3D two-group neutron kinetics.

Equations (all fission neutrons, prompt and delayed, are born in the fast group):
    (1/v1) dphi1/dt = -A1 phi1 + (1-beta) F phi + sum_i lam_i C_i
    (1/v2) dphi2/dt = -A2 phi2 + Ss12 phi1
    dC_i/dt         = beta_i F phi - lam_i C_i                      (6 precursor groups, per cell)
with A_g the diffusion + removal operators, F phi = (nSf1 phi1 + nSf2 phi2) / k0. Dividing by
the initial eigenvalue k0 makes the starting state an exact steady state, the standard way to
start a transient from a computed critical (or near-critical) configuration.

Three solution methods, sharing all the machinery:

  "direct"     Fully implicit (backward Euler) in time for flux and precursors. The reference:
               converges to the exact time-dependent solution as dt -> 0, but needs small steps.
  "pcqs"       Predictor-corrector quasi-static. Flux = amplitude p(t) x shape psi(r, t).
               Each macro step: (1) predict the flux with one large implicit step, (2) take its
               shape, (3) integrate the point-kinetics amplitude equations exactly with
               reactivity and generation time computed from that shape (adjoint-weighted),
               (4) rescale the flux, update precursors consistently. Large steps, accurate.
  "adiabatic"  Same amplitude machinery, but the shape is the static eigen-shape at the
               current blade position (ignores shape lag; fine for slow rod motion).

Thermal feedback (optional, thermal.ChannelThermal): per-channel plate and coolant
temperatures change the fuel cross sections; see TransientSolver for how they are coupled.

Point-kinetics parameters (exact definitions given a shape psi, adjoint weight w at t = 0):
    rho    = <w, (F - L) psi> / <w, F psi>
    Lambda = <w, psi/v>       / <w, F psi>
    beta_eff = beta  (delayed and prompt neutrons share a spectrum in this two-group model)
"""
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import solve_ivp

from diffusion3d import Problem, build_loss_matrix, inner_rtol, pcg, solve_k

KEEPIN_BETA = np.array([0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273])
KEEPIN_LAMBDA = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])


# ----------------------------------------------------------------------------- inputs
@dataclass
class KineticsData:
    beta_i: np.ndarray = field(default_factory=lambda: KEEPIN_BETA.copy())
    lam_i: np.ndarray = field(default_factory=lambda: KEEPIN_LAMBDA.copy())
    v1: float = 1.0e7           # fast neutron speed [cm/s]
    v2: float = 2.5e5           # thermal neutron speed [cm/s]
    nu: float = 2.43            # neutrons per fission (U-235 thermal)
    energy_per_fission_j: float = 200.0 * 1.602176634e-13   # recoverable energy, 200 MeV

    @property
    def beta(self):
        return float(self.beta_i.sum())


def kinetics_from_config(cfg):
    k = cfg.get("kinetics", {}) or {}
    data = KineticsData()
    if "beta" in k:
        data.beta_i = np.asarray(k["beta"], float)
    if "decay_constants" in k:
        data.lam_i = np.asarray(k["decay_constants"], float)
    if "velocities" in k:
        data.v1, data.v2 = map(float, k["velocities"])
    if "nu" in k:
        data.nu = float(k["nu"])
    if "energy_per_fission_mev" in k:
        data.energy_per_fission_j = float(k["energy_per_fission_mev"]) * 1.602176634e-13
    if len(data.beta_i) != len(data.lam_i):
        raise ValueError("beta and decay_constants must have the same length")
    return data


class BladeProgram:
    """Shim-bank position h(t): moves toward the latest commanded target at the drive speed;
    a scram drives it to 0 at the scram speed and locks out further commands.

        prog = BladeProgram(h0=25.9, drive_speed=0.2, scram_speed=60, travel=53)
        prog.command(t=5.0, target=26.3)
        prog.scram(t=90.0)
    """

    def __init__(self, h0, drive_speed, scram_speed, travel):
        self.h0, self.travel = float(h0), float(travel)
        self.drive, self.scram_speed = float(drive_speed), float(scram_speed)
        self.events = []                       # (time, target, speed)
        self.scram_time = None
        self._start = (0.0, self.h0, self.h0, self.drive)   # state the events start from

    def command(self, t, target, speed=None):
        """Drive toward `target` from time t at the drive speed (or at `speed`, if given:
        the automatic controller sets the regulating rod's speed every sample)."""
        if self.scram_time is not None and t >= self.scram_time:
            return
        spd = self.drive if speed is None else float(abs(speed))
        self.events.append((float(t), float(np.clip(target, 0, self.travel)), spd))
        self.events.sort(key=lambda e: e[0])

    def scram(self, t):
        if self.scram_time is None or t < self.scram_time:
            self.scram_time = float(t)
            self.events = [e for e in self.events if e[0] < t] + [(float(t), 0.0, self.scram_speed)]

    def compact(self, t):
        """Forget the motion before the last command at or before t (positions at times >= t
        are unchanged). Keeps position() cheap when commands come every second (controller)."""
        last = None
        for seg in self._segments():
            if seg[0] <= t:
                last = seg
            else:
                break
        if last is not None:
            self._start = last
            self.events = [e for e in self.events if e[0] > last[0]]

    def _segments(self):
        """Yield (t_start, h_start, target, speed) for each constant-command interval."""
        t, h, target, speed = self._start
        for te, tgt, spd in self.events:
            yield t, h, target, speed
            h = self._move(h, target, speed, te - t)
            t, target, speed = te, tgt, spd
        yield t, h, target, speed

    @staticmethod
    def _move(h, target, speed, dt):
        step = speed * max(dt, 0.0)
        return target if abs(target - h) <= step else h + np.sign(target - h) * step

    def position(self, t):
        h = self._start[1]
        for ts, hs, target, speed in self._segments():
            if ts > t:
                break
            h = self._move(hs, target, speed, t - ts)
        return h

    def breakpoints(self, t0, t1):
        """Times in (t0, t1) where the motion has a kink (command times, arrival times)."""
        out = []
        for ts, hs, target, speed in self._segments():
            if t0 < ts < t1:
                out.append(ts)
            if speed > 0 and target != hs:
                arrive = ts + abs(target - hs) / speed
                if t0 < arrive < t1:
                    out.append(arrive)
        return sorted(set(out))

    def speed_at(self, t):
        """Blade speed magnitude just after time t."""
        dt = 1e-6
        return abs(self.position(t + dt) - self.position(t)) / dt


class PlantProgram:
    """Step changes, at given times, of the coolant inlet temperature [K], the relative flow
    and the number of primary pumps running.

    Without a primary-loop model, 'flow' is the relative flow itself (an instant step) and
    'pumps' is ignored. With one (thermal.PrimaryLoop), 'flow' is the speed target of the
    running pumps (flow follows through the loop dynamics) and 'pumps' trips or restarts
    pumps: {pumps: 0} trips all of them and the flow coasts down."""
    KINDS = ("inlet_temperature", "flow", "pumps")

    def __init__(self, inlet_temperature=None, flow=1.0):
        self.T_in0, self.flow0 = inlet_temperature, float(flow)
        self.events = []                       # (time, kind, value)

    def set(self, t, kind, value):
        if kind not in self.KINDS:
            raise ValueError(kind)
        self.events.append((float(t), kind, float(value)))
        self.events.sort(key=lambda e: e[0])

    def state(self, t, T_in_default):
        """(inlet temperature, flow or pump speed target, pumps running or None = all)."""
        T_in = T_in_default if self.T_in0 is None else self.T_in0
        flow, pumps = self.flow0, None
        for te, kind, v in self.events:
            if te <= t + 1e-12:
                if kind == "inlet_temperature":
                    T_in = v
                elif kind == "flow":
                    flow = v
                else:
                    pumps = v
        return T_in, flow, pumps

    def breakpoints(self, t0, t1):
        return sorted({te for te, _, _ in self.events if t0 < te < t1})


# ----------------------------------------------------------------------------- linear algebra
def _cg(A, b, x0, rtol, dinv):
    """Jacobi-preconditioned CG (diffusion3d.pcg); dinv = 1 / diag(A)."""
    if not np.any(b):
        return np.zeros_like(b)
    return pcg(A, b, x0, rtol, dinv)


class _Ops:
    """Operators at one blade position (and, optionally, one time step size)."""

    def __init__(self, problem, k0, kin, dt=None):
        m, xs = problem.mesh, problem.xs
        self.n = m.n_cells
        flat = {k: np.asarray(v, float).ravel() for k, v in xs.items()}
        self.nsf1, self.nsf2 = flat["nSf1"] / k0, flat["nSf2"] / k0
        self.ss12 = flat["Ss12"]
        self.A1 = build_loss_matrix(m, xs["D1"], xs["Sa1"] + xs["Ss12"], problem.bc)
        self.A2 = build_loss_matrix(m, xs["D2"], xs["Sa2"], problem.bc)
        self.kin = kin
        self.dt = dt
        if dt is not None:
            self.M1 = (self.A1 + sp.identity(self.n) / (kin.v1 * dt)).tocsr()
            self.M2 = (self.A2 + sp.identity(self.n) / (kin.v2 * dt)).tocsr()
        else:                                   # steady state: no time-absorption terms
            self.M1, self.M2 = self.A1.tocsr(), self.A2.tocsr()
        self.P1 = 1.0 / self.M1.diagonal()          # Jacobi preconditioners (as arrays)
        self.P2 = 1.0 / self.M2.diagonal()

    def fission(self, phi1, phi2):
        return self.nsf1 * phi1 + self.nsf2 * phi2

    def loss_inner(self, w1, w2, psi1, psi2):
        """<w, L psi> with L the full loss operator (diffusion, removal, downscatter)."""
        return w1 @ (self.A1 @ psi1) + w2 @ (self.A2 @ psi2 - self.ss12 * psi1)


# ----------------------------------------------------------------------------- adjoint
def solve_adjoint(problem, tol_k=1e-9, tol_src=1e-8, max_outer=5000):
    """Fundamental-mode adjoint fluxes (w1, w2) and eigenvalue.
    Adjoint equations:  A1 w1 - Ss12 w2 = (1/k) nSf1 w1,   A2 w2 = (1/k) nSf2 w1."""
    ops = _Ops(problem, 1.0, KineticsData())
    P1, P2 = 1 / ops.A1.diagonal(), 1 / ops.A2.diagonal()
    A1, A2 = ops.A1.tocsr(), ops.A2.tocsr()
    w1, w2 = np.ones(ops.n), np.ones(ops.n)
    k, dsrc = 1.0, 1.0
    for it in range(max_outer):
        src = w1.copy()                                   # adjoint "fission source"
        rtol = inner_rtol(dsrc, tol_src * 0.01)
        w2 = _cg(A2, ops.nsf2 * src / k, w2, rtol, P2)
        w1 = _cg(A1, ops.nsf1 * src / k + ops.ss12 * w2, w1, rtol, P1)
        wgt = ops.nsf1 + ops.nsf2                          # any positive weight works
        k_new = k * (wgt @ w1) / (wgt @ src)
        norm = np.linalg.norm(w1)
        dsrc = np.linalg.norm(w1 / norm - src / np.linalg.norm(src))
        w1, w2 = w1 / norm, w2 / norm
        dk, k = abs(k_new - k), k_new
        if dk < tol_k and dsrc < tol_src and rtol <= tol_src * 0.01:
            break
    return w1, w2, k


# ----------------------------------------------------------------------------- results
@dataclass
class History:
    t: list = field(default_factory=list)
    blade: list = field(default_factory=list)
    reg_rod: list = field(default_factory=list)        # regulating rod position (if modeled)
    power: list = field(default_factory=list)          # total fission power / initial
    amplitude: list = field(default_factory=list)      # adjoint-weighted amplitude p(t)
    rho: list = field(default_factory=list)            # reactivity [dk/k]
    gen_time: list = field(default_factory=list)       # Lambda [s]
    element_power: list = field(default_factory=list)  # per element, relative to initial core avg
    gmres_its: list = field(default_factory=list)
    wall: list = field(default_factory=list)
    # thermal-hydraulics (filled only when a thermal model is attached)
    rho_feedback: list = field(default_factory=list)   # rho(T) - rho(T_ref), same shape
    fuel_max: list = field(default_factory=list)       # hottest plate node [K]
    fuel_avg: list = field(default_factory=list)       # heat-capacity-weighted plate temp [K]
    coolant_outlet: list = field(default_factory=list) # mixed core outlet [K]
    coolant_outlet_max: list = field(default_factory=list)  # hottest channel outlet [K]
    inlet: list = field(default_factory=list)
    flow: list = field(default_factory=list)
    heat_removed: list = field(default_factory=list)   # coolant heat removal / initial power
    channel_outlet: list = field(default_factory=list) # per channel outlet [K]
    thermal_power: list = field(default_factory=list)  # heat deposited in the fuel [W]
    decay_heat: list = field(default_factory=list)     # fission-product decay heat [W]
    core_flow: list = field(default_factory=list)      # core flow [kg/s]
    pump_speed: list = field(default_factory=list)     # mean primary pump speed (relative)
    ncv_open: list = field(default_factory=list)       # natural convection valves open
    energy_in: list = field(default_factory=list)      # heat deposited since t = 0 [J]
    energy_out: list = field(default_factory=list)     # heat carried out since t = 0 [J]
    events: list = field(default_factory=list)

    def arrays(self):
        return {k: np.asarray(v) for k, v in self.__dict__.items() if k != "events"}


# ----------------------------------------------------------------------------- solver
class TransientSolver:
    """
    model      : core_builder.CoreModel (provides problem_at(h) and the element tally matrix W)
    program    : BladeProgram
    method     : "direct", "pcqs" or "adiabatic"
    dt_max     : largest time step [s]
    dh_max     : largest blade travel per step [cm] (refines steps while blades move)
    dlnp_max   : largest expected change of ln(power) per step (refines fast transients)
    n_rho      : reactivity/Lambda sample points inside each quasi-static macro step
    trip_power : automatic scram when power/initial exceeds this (None = off)
    thermal    : thermal.ChannelThermal for temperature feedback (None = no feedback)
    reg_program: BladeProgram of the regulating rod (None = it stays at the model's position)
    controller : controller.PowerController driving reg_program (None = manual only). It is
                 sampled at the end of each step (sample and command times are step breakpoints)
    dhr_max    : largest regulating rod travel per step [cm] (its worth per cm is ~100x below
                 the bank's)
    power_w    : initial power [W]; sets the temperatures when thermal is attached
    plant      : PlantProgram (inlet temperature / flow changes)
    source     : external neutron source per cell [n/(cm^3 s)], emitted into the fast group
                 (e.g. a Pu-Be startup source). With a source the problem is solved in
                 absolute units and physical cross sections (no division by k0): the start is
                 the source-driven steady state of a subcritical core, its power follows from
                 the source strength (power_w is ignored), and fluxes are in n/(cm^2 s).

    Feedback coupling. Direct method: temperatures are advanced after each flux step
    (backward Euler, lagged by one small step). Quasi-static methods: the node temperatures
    are integrated *inside* the amplitude equations, with reactivity
        rho(t) = rho_0(t) + sum_k a_k (T_k(t) - T_k(t_n))
    where rho_0 is the exact reactivity with temperatures frozen at t_n and a_k = d rho/d T_k
    are exact first-order sensitivities. At the end of every step the reactivity is
    recomputed exactly with the new temperatures, so linearization errors do not accumulate.

    Decay heat and the primary loop (when the thermal model has them) are integrated the same
    way: inside the amplitude equations for the quasi-static methods (the node decay heat
    through 2 x 23 scalar convolution integrals of the amplitude, exact for the
    piecewise-linear shape; pump speeds and loop flows as extra unknowns, with the natural
    convection valves switching at located events), and after each flux step for the
    direct method (decay heat exactly for the end-of-step power, loop with frozen
    temperatures). dflow_max limits the relative change of core flow or pump speed per step.
    """

    def __init__(self, model, program, kin=None, method="pcqs", dt_max=1.0, dh_max=0.5,
                 dlnp_max=0.25, n_rho=4, tol=1e-7, trip_power=None, trip_delay=0.0,
                 thermal=None, power_w=1.0, plant=None, source=None, verbose=False,
                 dflow_max=0.05, reg_program=None, controller=None, dhr_max=2.0):
        if method not in ("direct", "pcqs", "adiabatic"):
            raise ValueError(method)
        self.model, self.prog = model, program
        self.reg, self.controller, self.dhr_max = reg_program, controller, dhr_max
        if controller is not None and reg_program is None:
            raise ValueError("an automatic controller needs a regulating rod program")
        self.kin = kin or KineticsData()
        self.method, self.dt_max, self.dh_max, self.dlnp_max = method, dt_max, dh_max, dlnp_max
        self.dflow_max = dflow_max
        self.n_rho, self.tol = n_rho, tol
        self.trip_power, self.trip_delay = trip_power, trip_delay
        self.thermal, self.P0 = thermal, float(power_w)
        self.plant = plant or PlantProgram()
        self.Q = None if source is None else np.asarray(source, float).ravel()
        self.verbose = verbose
        self.hist = History()
        if self.Q is None:
            self._init_steady_state()
        else:
            self._init_source_state()
        self._control_update()

    # ---------------- control rod positions
    def _pos(self, t):
        """Absorber positions at time t: shim bank height, or (bank, regulating rod)."""
        h = self.prog.position(t)
        return h if self.reg is None else (h, self.reg.position(t))

    def _bare_problem(self, pos):
        return self.model.problem_at(*pos) if isinstance(pos, tuple) else self.model.problem_at(pos)

    @property
    def scrammed(self):
        st = self.prog.scram_time
        return st is not None and self.t >= st - 1e-12

    def _control_update(self):
        """Automatic controller: apply due commands, sample on its grid (end of a step)."""
        if self.controller is not None:
            self.controller.update(self.t, self.hist.power[-1] * self.P0, self.scrammed)

    # ---------------- problems with feedback
    def _problem(self, h, Tf=None, Tc=None):
        prob = self._bare_problem(h)
        if self.thermal is not None:
            prob = Problem(prob.mesh, self.thermal.apply(prob.xs, Tf, Tc), prob.bc)
        return prob

    def _set_plant(self, t):
        th = self.thermal
        if th is None:
            return
        th.T_in, flow, pumps = self.plant.state(t, th.T_in)
        if th.loop is None:
            th.flow = flow
        else:
            th.loop.set_command(flow, pumps)
            th.flow = th.loop.relative_flow

    def _init_thermal_extras(self, q_fission):
        """Decay heat in equilibrium (long operation) and the loop in forced-flow steady state."""
        th = self.thermal
        if th.decay is not None:
            th.decay.set_equilibrium(q_fission)
        if th.loop is not None:
            th.loop.init_steady(th.Tc, th.T_in)
            self._set_plant(0.0)
        th.E_in = th.E_out = 0.0

    # ---------------- initial state
    def _init_steady_state(self):
        t0 = time.time()
        h0 = self._pos(0.0)
        th = self.thermal
        self._set_plant(0.0)
        r = solve_k(self._problem(h0), tol_k=1e-10, tol_src=1e-9)
        if th is not None:
            # flux shape and temperatures must agree: iterate (feedback is weak -> fast)
            for it in range(50):
                prob = self._problem(h0)
                r = solve_k(prob, tol_k=1e-10, tol_src=1e-9, guess=r)
                S = (prob.xs["nSf1"] * r.phi1 + prob.xs["nSf2"] * r.phi2).ravel()
                q = th.node_power(S, self.P0 / S.sum())
                Tf, Tc = th.steady_state(q)
                dT = max(np.abs(Tf - th.Tf).max(), np.abs(Tc - th.Tc).max())
                th.Tf, th.Tc = Tf, Tc
                if dT < 1e-7:
                    break
            r = solve_k(self._problem(h0), tol_k=1e-11, tol_src=1e-10, guess=r)
            prob = self._problem(h0)
            S = (prob.xs["nSf1"] * r.phi1 + prob.xs["nSf2"] * r.phi2).ravel()
            self._init_thermal_extras(th.node_power(S, self.P0 / S.sum()))
        prob = self._problem(h0)
        self.k0 = r.k
        self.phi1, self.phi2 = r.phi1.ravel().copy(), r.phi2.ravel().copy()
        self.ops = _Ops(prob, self.k0, self.kin)
        w1, w2, k_adj = solve_adjoint(prob)
        if abs(k_adj - self.k0) > 1e-6:
            raise RuntimeError(f"adjoint eigenvalue {k_adj} != forward {self.k0}")
        self.w1, self.w2 = w1, w2
        kin = self.kin
        S = self.ops.fission(self.phi1, self.phi2)
        self.C = np.outer(kin.beta_i / kin.lam_i, S)       # equilibrium precursors (6, n)
        self.K0 = self._wv(self.phi1, self.phi2)           # <w, phi/v>, conserved by the shape
        self.S0 = S.sum()                                   # total fission rate, for power
        self.watt_per_fission = self.P0 / self.S0
        self.t, self.p = 0.0, 1.0
        self.psi1, self.psi2 = self.phi1.copy(), self.phi2.copy()
        self.rho, self.Lam = self._pk_params(self.ops, self.psi1, self.psi2)
        self._eig_guess = r
        self._last_growth, self._last_dt = 1.0, None
        self._record(gmres_its=0, wall=time.time() - t0)
        if self.verbose:
            extra = ""
            if th is not None:
                extra = (f"\n  at {self.P0:.3g} W: hottest plate {th.Tf.max() - 273.15:.1f} C, "
                         f"core outlet {self.hist.coolant_outlet[-1] - 273.15:.1f} C, "
                         f"feedback reactivity {self.hist.rho_feedback[-1] * 1e5:+.1f} pcm")
            print(f"  steady state: k0 = {self.k0:.6f}, rho(0) = {self.rho*1e5:+.2e} pcm, "
                  f"Lambda = {self.Lam*1e6:.1f} us, beta = {kin.beta*1e5:.0f} pcm  "
                  f"({time.time()-t0:.1f} s){extra}")

    def _init_source_state(self):
        """Steady state of a subcritical core driven by the external source (absolute units)."""
        t0 = time.time()
        kin, th = self.kin, self.thermal
        h0 = self._pos(0.0)
        self._set_plant(0.0)
        prob = self._problem(h0)
        r = solve_k(prob, tol_k=1e-9, tol_src=1e-8)
        self.k_initial = r.k
        if r.k >= 1.0:
            raise ValueError(f"a source-driven start needs a subcritical core (k = {r.k:.5f} "
                             f"with the blades at {h0:.2f} cm)")
        self.k0 = 1.0                                     # physical cross sections
        self.ops = _Ops(prob, 1.0, kin)
        # all fission neutrons count at steady state (delayed ones are in equilibrium)
        f1, f2, _ = self._solve_fixed(self.ops, 1.0, self.Q, np.zeros(self.ops.n), None)
        self.phi1, self.phi2 = f1, f2
        S = self.ops.fission(f1, f2)
        self.watt_per_fission = (self.model.problem.mesh.cell_volume * kin.energy_per_fission_j
                                 / kin.nu)                # W per (nu Sigma_f phi) summed over cells
        self.S0 = S.sum()
        self.P0 = self.S0 * self.watt_per_fission
        if th is not None:
            q = th.node_power(S, self.watt_per_fission)
            th.Tf, th.Tc = th.steady_state(q)
            self._init_thermal_extras(q)
            self.ops = _Ops(self._problem(h0), 1.0, kin)
        w1, w2, k_adj = solve_adjoint(prob)
        if abs(k_adj - r.k) > 1e-6:
            raise RuntimeError(f"adjoint eigenvalue {k_adj} != forward {r.k}")
        self.w1, self.w2 = w1, w2
        self.C = np.outer(kin.beta_i / kin.lam_i, S)
        self.K0 = self._wv(f1, f2)
        self.t, self.p = 0.0, 1.0
        self.psi1, self.psi2 = f1.copy(), f2.copy()
        self.rho, self.Lam = self._pk_params(self.ops, self.psi1, self.psi2)
        self._eig_guess = r
        self._last_growth, self._last_dt = 1.0, None
        self._record(gmres_its=0, wall=time.time() - t0)
        if self.verbose:
            src = self.Q.sum() * self.model.problem.mesh.cell_volume
            print(f"  source-driven steady state: source {src:.3g} n/s, k_eff = {r.k:.5f}, "
                  f"power {self.P0:.3g} W, rho = {self.rho * 1e5:+.0f} pcm "
                  f"(${self.rho / kin.beta:+.2f}), Lambda = {self.Lam * 1e6:.1f} us  "
                  f"({time.time() - t0:.1f} s)")

    # ---------------- inner products
    def _wv(self, f1, f2):
        return self.w1 @ f1 / self.kin.v1 + self.w2 @ f2 / self.kin.v2

    def _pk_params(self, ops, psi1, psi2):
        wF = self.w1 @ ops.fission(psi1, psi2)
        rho = (wF - ops.loss_inner(self.w1, self.w2, psi1, psi2)) / wF
        Lam = self._wv(psi1, psi2) / wF
        return rho, Lam

    # ---------------- implicit (backward Euler) flux step
    def _implicit_step(self, ops, phi1_old, phi2_old, C_old, guess):
        kin, dt = self.kin, ops.dt
        lam = kin.lam_i
        a = 1.0 - kin.beta + np.sum(kin.beta_i * lam * dt / (1 + lam * dt))
        q1 = phi1_old / (kin.v1 * dt) + (lam / (1 + lam * dt)) @ C_old
        if self.Q is not None:
            q1 = q1 + self.Q
        q2 = phi2_old / (kin.v2 * dt)
        f1, f2, n_mv = self._solve_fixed(ops, a, q1, q2, guess)
        S = ops.fission(f1, f2)
        C_new = (C_old + dt * np.outer(kin.beta_i, S)) / (1 + lam * dt)[:, None]
        return f1, f2, C_new, n_mv

    def _solve_fixed(self, ops, a, q1, q2, guess):
        """Solve  M1 phi1 = a F phi + q1,  M2 phi2 = Ss12 phi1 + q2  for the fission source
        s = F phi with GMRES (each iteration is one two-group sweep). The inner solves stay at
        a fixed tolerance: GMRES needs a (nearly) fixed operator, and relaxing them per GMRES
        residual (inexact GMRES) was tried and cost more iterations than it saved."""
        inner = self.tol * 1e-2

        def sweep(s, with_q):
            b1 = a * s + (q1 if with_q else 0.0)
            f1 = _cg(ops.M1, b1, None, inner, ops.P1)
            b2 = ops.ss12 * f1 + (q2 if with_q else 0.0)
            f2 = _cg(ops.M2, b2, None, inner, ops.P2)
            return f1, f2

        rhs_1, rhs_2 = sweep(np.zeros(ops.n), True)
        rhs = ops.fission(rhs_1, rhs_2)
        n_mv = [0]

        def matvec(s):
            n_mv[0] += 1
            f1, f2 = sweep(s, False)
            return s - ops.fission(f1, f2)

        A = spla.LinearOperator((ops.n, ops.n), matvec=matvec, dtype=float)
        s, info = spla.gmres(A, rhs, x0=guess, rtol=self.tol, atol=0.0, restart=60, maxiter=100)
        if info != 0:
            raise RuntimeError(f"GMRES did not converge (info={info})")
        f1, f2 = sweep(s, True)                 # flux consistent with the converged source
        return f1, f2, n_mv[0]

    # ---------------- amplitude (point kinetics) over one macro step
    def _amplitude(self, dt, c0, rho_t, lam_t, s_pts, fb=None, qs=0.0):
        """Integrate PK from p=1 (scaled) over [0, dt] with piecewise-linear rho and 1/Lambda.
        Also returns the integrals needed to update spatial precursors:
            I0_i = int p (1-s) e^{-lam_i (dt - t)} dt,   I1_i = int p s e^{-lam_i (dt - t)} dt.
        With fb (feedback data), the channel temperatures are integrated alongside."""
        kin = self.kin
        beta, bi, lam = kin.beta, kin.beta_i, kin.lam_i
        G = len(lam)
        tp = s_pts * dt
        inv_lam = 1.0 / np.asarray(lam_t)
        y0 = np.concatenate(([1.0], c0, np.zeros(2 * G)))

        if fb is None:
            def f(t, y):
                p, c = y[0], y[1:1 + G]
                r = np.interp(t, tp, rho_t)
                iL = np.interp(t, tp, inv_lam)   # 1/Lambda ~ <w, F psi> is linear in psi
                s = t / dt
                return np.concatenate(([(r - beta) * iL * p + lam @ c + qs], bi * iL * p - lam * c,
                                       p * (1 - s) - lam * y[1 + G:1 + 2 * G], p * s - lam * y[1 + 2 * G:]))
            sol = solve_ivp(f, (0.0, dt), y0, method="Radau", rtol=1e-9, atol=1e-13,
                            first_step=min(dt, 1e-5))
            if not sol.success:
                raise RuntimeError(sol.message)
            y = sol.y[:, -1]
            return y[0], y[1 + G:1 + 2 * G], y[1 + 2 * G:], None

        # ---- coupled amplitude + channel temperatures (+ decay heat, primary loop)
        th = self.thermal
        N = th.n
        dec, lp = th.decay, th.loop
        a_f, a_c, G0, G1 = fb["a_f"], fb["a_c"], fb["G0"], fb["G1"]
        Tf0, Tc0 = th.Tf.copy(), th.Tc.copy()
        Cf, Cc, hA, Tin = th.Cf, th.Cc, th.hA, th.T_in
        below = th.below
        hb = below >= 0
        ne, nl = th.ne, th.nl
        out_nodes = np.arange(ne) * nl + nl - 1
        Gd = 0 if dec is None else len(dec.lam)
        npmp = 0 if lp is None else lp.n_pumps
        iJ0 = 1 + 3 * G + np.arange(Gd)
        iJ1 = iJ0 + Gd
        iTf = 1 + 3 * G + 2 * Gd + np.arange(N)
        iTc = iTf + N
        iom = 1 + 3 * G + 2 * Gd + 2 * N + np.arange(npmp)
        iWp = 1 + 3 * G + 2 * Gd + 2 * N + npmp
        iWv = iWp + 1
        iE = (iWv + 1) if lp is not None else (1 + 3 * G + 2 * Gd + 2 * N)
        nY = iE + 2
        loop_y0 = lp.get_state() if lp is not None else np.zeros(0)
        y0 = np.concatenate((y0, np.zeros(2 * Gd), Tf0, Tc0, loop_y0, [0.0, 0.0]))
        if dec is not None:
            one_m_g = 1.0 - dec.gamma
            lamd, lam_b, D0 = dec.lam, dec.lam_b, dec.D
            sG0, sG1 = G0.sum(), G1.sum()
        else:
            one_m_g = 1.0
        mcp_fixed = th.mcp
        cp_e = th.cp_e

        def common(t, y):
            Tf, Tc = y[iTf], y[iTc]
            s = t / dt
            r = np.interp(t, tp, rho_t) + a_f @ (Tf - Tf0) + a_c @ (Tc - Tc0)
            qf = (1 - s) * G0 + s * G1
            heat = one_m_g * y[0] * qf
            if Gd:
                J0, J1 = y[iJ0], y[iJ1]
                heat = heat + np.exp(-lamd * t) @ D0 + G0 * (lam_b @ J0) + G1 * (lam_b @ J1)
            if lp is not None:
                Wc = y[iWp] + y[iWv]
                mcp = cp_e * max(Wc, 0.0)
            else:
                mcp = mcp_fixed
            return Tf, Tc, s, r, np.interp(t, tp, inv_lam), qf, heat, mcp

        def make(is_open):
            def f(t, y):
                p, c = y[0], y[1:1 + G]
                Tf, Tc, s, r, iL, qf, heat, mcp = common(t, y)
                Tb = np.full(N, Tin)
                Tb[hb] = Tc[below[hb]]
                parts = [[(r - beta) * iL * p + lam @ c + qs], bi * iL * p - lam * c,
                         p * (1 - s) - lam * y[1 + G:1 + 2 * G], p * s - lam * y[1 + 2 * G:1 + 3 * G]]
                if Gd:
                    parts += [p * (1 - s) - lamd * y[iJ0], p * s - lamd * y[iJ1]]
                parts += [(heat - hA * (Tf - Tc)) / Cf, (hA * (Tf - Tc) + mcp * (Tb - Tc)) / Cc]
                if lp is not None:
                    do, dwp, dwv, _ = lp.derivs(y[iom], y[iWp], y[iWv], lp.buoyancy(Tc, Tin), is_open)
                    parts += [do, [dwp, dwv if is_open else 0.0]]
                parts += [[heat.sum(), mcp * np.sum(Tc[out_nodes] - Tin)]]
                return np.concatenate(parts)

            def jac(t, y):
                p = y[0]
                Tf, Tc, s, r, iL, qf, heat, mcp = common(t, y)
                R, C, V = [], [], []

                def add(rr, cc, vv):
                    rr, cc = np.broadcast_arrays(np.asarray(rr), np.asarray(cc))
                    R.append(rr.ravel())
                    C.append(cc.ravel())
                    V.append(np.broadcast_to(np.asarray(vv, float), rr.shape).ravel())

                gi = np.arange(G)
                ic_, iI0, iI1 = 1 + gi, 1 + G + gi, 1 + 2 * G + gi
                add(0, 0, (r - beta) * iL)
                add(0, ic_, lam)
                add(0, iTf, a_f * p * iL)
                add(0, iTc, a_c * p * iL)
                add(ic_, 0, bi * iL)
                add(ic_, ic_, -lam)
                add(iI0, 0, 1 - s)
                add(iI0, iI0, -lam)
                add(iI1, 0, s)
                add(iI1, iI1, -lam)
                add(iTf, 0, one_m_g * qf / Cf)
                add(iTf, iTf, -hA / Cf)
                add(iTf, iTc, hA / Cf)
                add(iTc, iTf, hA / Cc)
                add(iTc, iTc, -(hA + mcp) / Cc)
                add(iTc[hb], iTc[below[hb]], mcp / Cc[hb])
                add(iE, 0, one_m_g * qf.sum())
                add(iE + 1, iTc[out_nodes], mcp)
                if Gd:
                    add(iJ0, 0, 1 - s)
                    add(iJ0, iJ0, -lamd)
                    add(iJ1, 0, s)
                    add(iJ1, iJ1, -lamd)
                    add(iTf[:, None], iJ0[None, :], (G0 / Cf)[:, None] * lam_b[None, :])
                    add(iTf[:, None], iJ1[None, :], (G1 / Cf)[:, None] * lam_b[None, :])
                    add(iE, iJ0, lam_b * sG0)
                    add(iE, iJ1, lam_b * sG1)
                if lp is not None:
                    Tb = np.full(N, Tin)
                    Tb[hb] = Tc[below[hb]]
                    if y[iWp] + y[iWv] > 0:
                        dmcp = cp_e * (Tb - Tc) / Cc
                        add(iTc, iWp, dmcp)
                        add(iTc, iWv, dmcp)
                        dE = cp_e * np.sum(Tc[out_nodes] - Tin)
                        add(iE + 1, iWp, dE)
                        add(iE + 1, iWv, dE)
                    _, _, dB, dBp = lp.buoyancy(Tc, Tin, grad=True)
                    d_om, Jw, F = lp.partials(y[iom], y[iWp], y[iWv], is_open)
                    add(iom, iom, d_om)
                    wc = np.concatenate((iom, [iWp, iWv]))
                    add(iWp, wc, Jw[0])
                    add(iWp, iTc, F[0, 0] * dB + F[0, 1] * dBp)
                    if is_open:
                        add(iWv, wc, Jw[1])
                        add(iWv, iTc, F[1, 0] * dB + F[1, 1] * dBp)
                return sp.csc_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))),
                                     shape=(nY, nY))
            return f, jac

        atol = np.concatenate((np.full(1 + 3 * G + 2 * Gd, 1e-13), np.full(2 * N, 1e-7),
                               np.full(npmp, 1e-10), np.full(2 if lp is not None else 0, 1e-9 * th.W0),
                               [1e-6, 1e-6]))
        t0, y, is_open = 0.0, y0, (lp.open if lp is not None else False)
        for _ in range(50):                        # NCV switches inside this step
            f, jac = make(is_open)
            events = None
            if lp is not None and lp.n_valves > 0:
                def ev(t, yy, o=is_open):
                    return lp.event_value(yy[iom], yy[iWp], yy[iWv], lp.buoyancy(yy[iTc], Tin), o)
                ev.terminal, ev.direction = True, -1
                events = [ev]
            sol = solve_ivp(f, (t0, dt), y, method="Radau", rtol=1e-9, atol=atol, jac=jac,
                            events=events, first_step=min(dt - t0, 1e-5))
            if not sol.success:
                raise RuntimeError(sol.message)
            if events and sol.status == 1:
                t0, y = float(sol.t_events[0][0]), sol.y_events[0][0].copy()
                is_open = not is_open
                if not is_open:
                    y[iWv] = 0.0
                if dt - t0 > 1e-12:
                    continue
            else:
                y = sol.y[:, -1]
            break
        extra = dict(Tf=y[iTf], Tc=y[iTc], J0=y[iJ0], J1=y[iJ1], E_in=y[iE], E_out=y[iE + 1],
                     loop_y=y[iWp - npmp:iE].copy() if lp is not None else None, loop_open=is_open)
        return y[0], y[1 + G:1 + 2 * G], y[1 + 2 * G:1 + 3 * G], extra

    # ---------------- one step
    def _choose_dt(self, t_end):
        t = self.t
        dt = min(self.dt_max, t_end - t)
        speed = self.prog.speed_at(t)
        if speed > 0:
            dt = min(dt, self.dh_max / speed)
        if self.reg is not None:
            speed = self.reg.speed_at(t)
            if speed > 0:
                dt = min(dt, self.dhr_max / speed)
        if self._last_dt and self.dlnp_max:
            rate = abs(np.log(self._last_growth)) / self._last_dt     # |d ln P / dt|
            if rate > 0:
                dt = min(dt, self.dlnp_max / rate)
        th = self.thermal
        if th is not None and th.loop is not None and self.dflow_max:
            # relative rate of change of core flow and pump speed right now
            self._set_plant(t)
            lp = th.loop
            do, dwp, dwv, _ = lp.derivs(lp.omega, lp.Wp, lp.Wv, lp.buoyancy(th.Tc, th.T_in), lp.open)
            rate = max(abs(dwp + (dwv if lp.open else 0.0)) / max(abs(lp.Wc), 1e-3 * lp.W0),
                       np.max(np.abs(do) / np.maximum(np.abs(lp.omega), 0.05)) if lp.n_pumps else 0.0)
            if rate > 0:
                dt = min(dt, self.dflow_max / rate)
        bps = self.prog.breakpoints(t + 1e-9, t + dt) + self.plant.breakpoints(t + 1e-9, t + dt)
        if self.reg is not None:
            bps += self.reg.breakpoints(t + 1e-9, t + dt)
        if self.controller is not None:
            bps += self.controller.breakpoints(t + 1e-9, t + dt)
        if bps:
            dt = min(bps) - t
        return max(dt, 1e-6)

    def step(self, dt):
        t_wall = time.time()
        kin, th = self.kin, self.thermal
        self._set_plant(self.t)                  # inlet / flow for this step
        t1 = self.t + dt
        h1 = self._pos(t1)
        prob1 = self._problem(h1)                # temperatures still at t_n
        ops1 = _Ops(prob1, self.k0, kin, dt)
        F_old = self.ops.fission(self.phi1, self.phi2)
        guess = F_old * self._last_growth ** (dt / self._last_dt) if self._last_dt else F_old
        its = 0

        if self.method == "direct":
            f1, f2, C1, its = self._implicit_step(ops1, self.phi1, self.phi2, self.C, guess)
            self.phi1, self.phi2, self.C = f1, f2, C1
            if th is not None:
                self._thermal_step_direct(dt, th.node_power(ops1.fission(f1, f2), self.watt_per_fission))
                ops_end = _Ops(self._problem(h1), self.k0, kin)
            else:
                ops_end = ops1
            K = self._wv(f1, f2)
            self.p = K / self.K0
            self.psi1, self.psi2 = f1 / self.p, f2 / self.p
            self.rho, self.Lam = self._pk_params(ops_end, self.psi1, self.psi2)
        else:
            # ---- shape at the end of the step
            if self.method == "pcqs":
                f1, f2, _, its = self._implicit_step(ops1, self.phi1, self.phi2, self.C, guess)
            else:
                r = solve_k(prob1, tol_k=self.tol, tol_src=self.tol * 10, guess=self._eig_guess)
                self._eig_guess = r
                f1, f2 = r.phi1.ravel(), r.phi2.ravel()
                its = r.outer_iterations
            scale = self.K0 / self._wv(f1, f2)
            psi1_new, psi2_new = f1 * scale, f2 * scale

            # ---- reactivity (temperatures frozen at t_n) and Lambda across the step
            s_pts = np.linspace(0.0, 1.0, self.n_rho + 1)
            rho_t, lam_t = [self.rho], [self.Lam]
            for s in s_pts[1:]:
                ops_s = ops1 if s == 1.0 else _Ops(self._problem(self._pos(self.t + s * dt)),
                                                   self.k0, kin)
                ps1 = (1 - s) * self.psi1 + s * psi1_new
                ps2 = (1 - s) * self.psi2 + s * psi2_new
                r_, L_ = self._pk_params(ops_s, ps1, ps2)
                rho_t.append(r_)
                lam_t.append(L_)

            fb = None
            if th is not None:
                # exact sensitivities d rho / d T_k at the mid-step shape and blade position
                pm1, pm2 = 0.5 * (self.psi1 + psi1_new), 0.5 * (self.psi2 + psi2_new)
                prob_m = self._problem(self._pos(self.t + 0.5 * dt))
                a_f, a_c, _ = th.rho_sensitivity(prob_m.xs, self.k0, self.w1, self.w2, pm1, pm2,
                                                 prob_m.bc)
                wpf = self.watt_per_fission * self.p
                fb = dict(a_f=a_f, a_c=a_c,
                          G0=th.node_power(self.ops.fission(self.psi1, self.psi2), wpf),
                          G1=th.node_power(ops1.fission(psi1_new, psi2_new), wpf))

            # ---- amplitude: point kinetics with precursors projected from the spatial ones
            c0 = (self.C @ self.w1) / self.K0 / self.p          # scaled so p(t_n) = 1
            qs = 0.0 if self.Q is None else (self.w1 @ self.Q) / self.K0 / self.p
            p_ratio, I0, I1, ex = self._amplitude(dt, c0, np.array(rho_t), np.array(lam_t),
                                                  s_pts, fb, qs)
            p_old = self.p
            self.p = p_old * p_ratio
            if th is not None:
                th.Tf, th.Tc = ex["Tf"], ex["Tc"]
                th.E_in += ex["E_in"]
                th.E_out += ex["E_out"]
                if th.decay is not None:
                    dec = th.decay
                    dec.D = (np.exp(-dec.lam * dt)[:, None] * dec.D
                             + dec.lam_b[:, None] * (np.outer(ex["J0"], fb["G0"]) + np.outer(ex["J1"], fb["G1"])))
                if th.loop is not None:
                    th.loop.set_state(ex["loop_y"], ex["loop_open"])
                    th.flow = th.loop.relative_flow
                ops_end = _Ops(self._problem(h1), self.k0, kin)
            else:
                ops_end = ops1

            # ---- spatial precursors, exactly integrated with the fine-grained amplitude
            Fpsi0 = self.ops.fission(self.psi1, self.psi2)
            Fpsi1 = ops_end.fission(psi1_new, psi2_new)
            decay = np.exp(-kin.lam_i * dt)
            self.C = (decay[:, None] * self.C
                      + p_old * kin.beta_i[:, None] * (np.outer(I0, Fpsi0) + np.outer(I1, Fpsi1)))
            self.psi1, self.psi2 = psi1_new, psi2_new
            self.phi1, self.phi2 = self.p * psi1_new, self.p * psi2_new
            # exact reactivity with the new temperatures (no accumulated linearization error)
            self.rho, self.Lam = self._pk_params(ops_end, self.psi1, self.psi2)

        power_old = self.hist.power[-1]
        self.ops, self.t = ops_end, t1
        self._record(its, time.time() - t_wall)
        self._last_growth = max(min(self.hist.power[-1] / power_old, 10.0), 0.1)
        self._last_dt = dt
        self._check_trip()
        self._control_update()

    def _thermal_step_direct(self, dt, q_fission):
        """Direct method: loop (frozen temperatures), decay heat (end-of-step fission power),
        then backward Euler for the node temperatures; energy bookkeeping exact per step."""
        th = self.thermal
        if th.loop is not None:
            th.loop.advance(dt, th.loop.buoyancy(th.Tc, th.T_in))
            th.flow = th.loop.relative_flow
        if th.decay is not None:
            th.decay.advance_constant(dt, q_fission)
        q = th.heat(q_fission)
        th.step_be(dt, q)
        th.E_in += q.sum() * dt
        th.E_out += th.outflow_power() * dt

    def scram(self, reason=None):
        """Scram now (operator or an external trip): blades drop from the current time and
        the automatic controller drops to manual at once."""
        if self.prog.scram_time is None and reason:
            self.hist.events.append((self.t, reason))
        self.prog.scram(self.t)
        self._control_update()

    def _check_trip(self):
        if self.trip_power is None or self.prog.scram_time is not None:
            return
        if self.hist.power[-1] > self.trip_power:
            self.prog.scram(self.t + self.trip_delay)
            self.hist.events.append((self.t, f"high-power trip at {self.hist.power[-1]:.3g} x initial"))

    def run(self, t_end, callback=None):
        while self.t < t_end - 1e-9:
            self.step(self._choose_dt(t_end))
            if self.verbose:
                h = self.hist
                extra = ""
                if self.thermal is not None:
                    extra = (f"  Tfmax = {h.fuel_max[-1] - 273.15:6.2f} C  "
                             f"out = {h.coolant_outlet[-1] - 273.15:6.2f} C")
                reg = f"  reg = {h.reg_rod[-1]:6.2f} cm" if h.reg_rod else ""
                print(f"  t = {self.t:8.3f} s  blade = {h.blade[-1]:6.2f} cm{reg}  "
                      f"P = {h.power[-1] * self.P0:.5e} W  rho = {h.rho[-1] * 1e5:+8.1f} pcm{extra}  "
                      f"its = {h.gmres_its[-1]:3d}  ({h.wall[-1]:.2f} s)", flush=True)
            if callback:
                callback(self)
        return self.hist

    # ---------------- bookkeeping
    def _record(self, gmres_its, wall):
        h = self.hist
        S = self.ops.fission(self.phi1, self.phi2)
        h.t.append(self.t)
        pos = self._pos(self.t)
        h.blade.append(self.prog.position(self.t))
        if self.reg is not None:
            h.reg_rod.append(self.reg.position(self.t))
        h.power.append(S.sum() / self.S0)
        h.amplitude.append(self.p)
        h.rho.append(self.rho)
        h.gen_time.append(self.Lam)
        fueled_avg0 = getattr(self, "_elem_norm", None)
        P = self.model.W.T @ S
        if fueled_avg0 is None:
            self._elem_norm = fueled_avg0 = P[P > 0].mean()
        h.element_power.append(P / fueled_avg0)
        h.gmres_its.append(gmres_its)
        h.wall.append(wall)
        th = self.thermal
        if th is not None:
            # feedback reactivity: same shape, cross sections at T vs at the reference temperature
            ops_ref = _Ops(self._bare_problem(pos), self.k0, self.kin)
            rho_ref, _ = self._pk_params(ops_ref, self.psi1, self.psi2)
            h.rho_feedback.append(self.rho - rho_ref)
            h.fuel_max.append(th.Tf.max())
            h.fuel_avg.append(np.sum(th.Cf * th.Tf) / th.Cf.sum())
            out = th.Tc.reshape(th.ne, th.nl)[:, -1]
            h.channel_outlet.append(out.copy())
            h.coolant_outlet.append(out.mean())          # equal flow per channel -> mixed mean
            h.coolant_outlet_max.append(out.max())
            h.inlet.append(th.T_in)
            h.flow.append(th.flow)
            h.heat_removed.append(th.outflow_power() / self.P0)
            h.thermal_power.append(float(th.heat(th.node_power(S, self.watt_per_fission)).sum()))
            h.decay_heat.append(th.decay_power)
            lp = th.loop
            h.core_flow.append(lp.Wc if lp is not None else th.flow * th.W0)
            h.pump_speed.append(float(np.maximum(lp.omega, 0.0).mean()) if lp is not None and lp.n_pumps else th.flow)
            h.ncv_open.append(bool(lp.open) if lp is not None else False)
            h.energy_in.append(th.E_in)
            h.energy_out.append(th.E_out)

    @property
    def power_density(self):
        """Current fission density per cell (relative to the initial core-average)."""
        S = self.ops.fission(self.phi1, self.phi2)
        return (S / (self.S0 / np.count_nonzero(self.ops.nsf2))).reshape(self.model.problem.mesh.shape)
