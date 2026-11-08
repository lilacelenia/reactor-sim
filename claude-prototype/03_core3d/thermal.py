"""
thermal.py -- per-channel thermal-hydraulics and temperature feedback on cross sections.

Every fuel element is one coolant channel, split axially into the mesh layers that contain
fuel. Each (element, layer) node has two temperatures:

    T_f  plate (fuel) temperature:  C_f dT_f/dt = q - hA (T_f - T_c)
    T_c  coolant temperature:       C_c dT_c/dt = hA (T_f - T_c) + m cp (T_c,below - T_c)

Coolant flows upward (inlet at the bottom); T_c of a node is its mixed outlet temperature
(donor-cell / upwind scheme), so at steady state T_c rises by q / (m cp) per node.
q is the fission power deposited in that node, from the 3D flux.

Feedback: the cross sections of the feedback materials (the homogenized fuel, which includes
the channel water) change linearly with the node temperatures:
    Sigma_x(T) = Sigma_x0 * (1 + c_x * (T_c - T_ref) + f_x * (T_f - T_ref))
with c_x, f_x the fractional derivatives in the config (D is changed the same way, then mixed
harmonically like the other D's). Cells partly filled with fuel get the change in proportion
to their fuel volume fraction, and a cell shared by two elements gets the fission-weighted
mix of their temperatures.

The module also gives the exact first-order sensitivity of the adjoint-weighted reactivity to
every node temperature, d rho / d T_k, used to put the feedback inside the fast amplitude
integration of the quasi-static method.

Optional, when the config has the sections:

decay_heat (DecayHeat): fission-product decay heat with the 23-group ANS-5.1-1979 U-235 fit.
    Per node and group, D_k [W] obeys  dD_k/dt = lam_k (b_k q_f - D_k),  b_k = alpha_k/(lam_k Q),
    with q_f the node fission power. The heat deposited in a node is
        q = (1 - gamma) q_f + sum_k D_k,        gamma = sum_k b_k  (~6.6 %),
    so at equilibrium (long operation) q = q_f, and after a scram from long operation
    sum_k D_k = P0 sum_k b_k exp(-lam_k t), the standard's infinite-irradiation curve.

primary_loop (PrimaryLoop): pump coastdown and natural circulation. Lumped momentum balance of
    three flow paths between the lower plenum (below the core) and the tank above it:
        core (up)          I_c dW_c/dt = Dp - R_c W_c|W_c| + B(T)
        pump loop (down)   I_p dW_p/dt = dp_pump (mean pump speed)^2 - R_p(W_p) W_p|W_p| - B_p - Dp
        NCVs (down)        I_v dW_v/dt = -R_v W_v|W_v| - Dp             (only while open)
    with W_c = W_p + W_v and Dp the lower-plenum pressure above the tank's (dynamic part; the
    cold hydrostatic head is taken out). Dp follows from continuity. B is the buoyancy head of
    the heated channels plus the chimney above the core. B_p is the adverse head of the pump
    loop, whose hot leg (mixed core outlet water) runs down to the heat exchangers; check
    valves make R_p much larger for reverse flow. So once the pumps stop, the natural
    circulation runs through the NCVs, as the MITR analyses assume. After a trip each pump
    speed w (relative) obeys  tau dw/dt = -(w^2 + f): hydraulic torque ~ w^2 plus a constant
    friction torque f (relative), so  w = sqrt(f) tan(atan(1/sqrt(f)) - sqrt(f) t/tau)  and the
    pump stops at t = tau atan(1/sqrt(f))/sqrt(f) (f = 0: w = 1/(1 + t/tau)). The pump head is
    dp_pump (mean speed)^2 (affinity laws). The natural convection valves
    (ball valves held shut by the forced-flow pressure) open when Dp drops to zero and shut
    again if flow through them would reverse.
"""
import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp

from diffusion3d import FACES, XS_NAMES

CP_WATER = 4180.0          # J/(kg K)
GRAVITY = 9.80665          # m/s^2
MEV_J = 1.602176634e-13


# ----------------------------------------------------------------------------- water
_KELL = (999.83952, 16.945176, -7.9870401e-3, -46.170461e-6, 105.56302e-9, -280.54253e-12)
_KELL_B = 16.879850e-3


def water_density(T_kelvin):
    """Liquid water density [kg/m3] at 1 atm, Kell (1975), 0-150 C (< 0.01 % from IAPWS)."""
    t = np.asarray(T_kelvin, float) - 273.15
    num = _KELL[0] + t * (_KELL[1] + t * (_KELL[2] + t * (_KELL[3] + t * (_KELL[4] + t * _KELL[5]))))
    return num / (1.0 + _KELL_B * t)


def water_density_dT(T_kelvin):
    """d(density)/dT [kg/(m3 K)] of water_density."""
    t = np.asarray(T_kelvin, float) - 273.15
    num = _KELL[0] + t * (_KELL[1] + t * (_KELL[2] + t * (_KELL[3] + t * (_KELL[4] + t * _KELL[5]))))
    dnum = _KELL[1] + t * (2 * _KELL[2] + t * (3 * _KELL[3] + t * (4 * _KELL[4] + t * 5 * _KELL[5])))
    den = 1.0 + _KELL_B * t
    return (dnum * den - num * _KELL_B) / den**2


# ----------------------------------------------------------------------------- decay heat
class DecayHeat:
    """Fission-product decay heat, one set of groups per thermal node (see module docstring)."""

    def __init__(self, cfg_dh, n_nodes):
        alpha = np.asarray(cfg_dh["alpha_mev_per_fission_s"], float)
        self.lam = np.asarray(cfg_dh["lambda_per_s"], float)
        if alpha.shape != self.lam.shape:
            raise ValueError("decay_heat: alpha and lambda must have the same length")
        self.Q_mev = float(cfg_dh["energy_per_fission_mev"])
        self.b = alpha / (self.lam * self.Q_mev)           # equilibrium fraction per group
        self.gamma = float(self.b.sum())
        self.lam_b = self.lam * self.b
        self.D = np.zeros((len(self.lam), n_nodes))         # W per group and node

    def set_equilibrium(self, q_fission):
        self.D = np.outer(self.b, q_fission)

    def heat(self, q_fission, D=None):
        """Heat deposited per node [W] for node fission powers q_fission."""
        D = self.D if D is None else D
        return (1.0 - self.gamma) * q_fission + D.sum(axis=0)

    def advance_constant(self, dt, q_fission):
        """Exact update over dt with the node fission power held at q_fission."""
        e = np.exp(-self.lam * dt)[:, None]
        self.D = e * self.D + (1.0 - e) * np.outer(self.b, q_fission)

    def fraction_after_shutdown(self, t, T_op=np.inf):
        """ANS-5.1 closed form: decay power / operating fission power, t s after shutdown
        from T_op s of operation at constant power."""
        t = np.asarray(t, float)[..., None]
        f = self.b * np.exp(-self.lam * t)
        if np.isfinite(T_op):
            f = f * (1.0 - np.exp(-self.lam * T_op))
        return f.sum(axis=-1)


# ----------------------------------------------------------------------------- primary loop
class PrimaryLoop:
    """Pump coastdown and natural circulation (see module docstring).

    State: pump speeds omega (relative), W_p (pump-loop flow), W_v (NCV flow) [kg/s], and
    whether the NCVs are open. Commands: number of pumps running and their speed target."""

    def __init__(self, cfg, th):
        pl, thc = cfg["primary_loop"], cfg["thermal_hydraulics"]
        self.th = th
        self.W0 = float(thc["total_flow_kg_s"])
        self.fuel_frac = float(thc["fuel_flow_fraction"])
        self.n_pumps = int(pl["pumps"])
        self.dp0 = float(pl["pump_head_pa"])
        self.R_c = float(pl["core_pressure_drop_pa"]) / self.W0**2
        self.tau_cd = float(pl["coastdown_time_constant_s"])
        self.tau_up = float(pl["spinup_time_constant_s"])
        self.f_fric = float(pl.get("pump_friction_ratio", 0.0))
        self.I_c = float(pl["inertia_core_per_m"])
        self.I_p = float(pl["inertia_pump_loop_per_m"])
        self.I_v = float(pl["inertia_ncv_per_m"])
        self.H_ch = float(pl["chimney_height_m"])
        self.H_hx = float(pl.get("pump_loop_drop_m", 0.0))
        self.k_rev = float(pl.get("check_valve_reverse_factor", 1.0))
        self.n_valves = int(pl.get("natural_convection_valves", 0))
        self.dz = th.dz_m                                   # heated length per layer [m]
        self.R_p = (self.dp0 - self.R_c * self.W0**2) / self.W0**2   # refined in init_steady
        if self.R_p <= 0:
            raise ValueError("primary_loop: pump head must exceed the core pressure drop")
        self.R_v = np.inf
        self.calibration = None
        if self.n_valves > 0:
            self.R_v = self._calibrate_ncv(pl["nc_calibration"], th.T_in)
        self.reset()

    def reset(self):
        self.omega = np.ones(self.n_pumps)
        self.running = np.ones(self.n_pumps, bool)
        self.target = 1.0
        self.Wp, self.Wv, self.open = self.W0, 0.0, False

    # ---------------- commands and state
    def set_command(self, speed_target, n_running):
        self.target = float(speed_target)
        n = self.n_pumps if n_running is None else int(round(n_running))
        self.running = np.arange(self.n_pumps) < n

    @property
    def Wc(self):
        return self.Wp + self.Wv

    @property
    def relative_flow(self):
        return max(self.Wc, 0.0) / self.W0

    def get_state(self):
        return np.concatenate((self.omega, [self.Wp, self.Wv]))

    def set_state(self, y, is_open):
        n = self.n_pumps
        self.omega, self.Wp, self.Wv, self.open = y[:n].copy(), float(y[n]), float(y[n + 1]), bool(is_open)

    # ---------------- buoyancy
    def buoyancy(self, Tc, T_in, grad=False):
        """Heads [Pa] relative to water at the inlet temperature: B, driving the core (heated
        channels plus chimney), and B_p, opposing the pump loop. With grad=True also their
        gradients with respect to every coolant node."""
        ne, nl = self.th.ne, self.th.nl
        T = Tc.reshape(ne, nl)
        Tm = np.empty((ne, nl))                    # node-average temperatures
        Tm[:, 0] = T_in
        Tm[:, 1:] = T[:, :-1]
        Tm += T
        Tm *= 0.5
        T_mix = T[:, -1].mean()
        # one density evaluation for everything (called many times per step by the ODE solver)
        rho = water_density(np.concatenate((Tm.ravel(), T[:, -1], [T_in, T_mix])))
        rin, r_mix = rho[-2], rho[-1]
        B = (GRAVITY * ((rin * self.dz.sum()) - (rho[:ne * nl].reshape(ne, nl) @ self.dz).sum() / ne)
             + GRAVITY * self.H_ch * (rin - rho[ne * nl:-2].mean()))
        Bp = GRAVITY * self.H_hx * (rin - r_mix)
        if not grad:
            return B, Bp
        dr = water_density_dT(Tm) * self.dz
        g = -0.5 * GRAVITY * dr / ne
        g[:, :-1] -= 0.5 * GRAVITY * dr[:, 1:] / ne
        g[:, -1] -= GRAVITY * self.H_ch * water_density_dT(T[:, -1]) / ne
        gp = np.zeros((ne, nl))
        gp[:, -1] = -GRAVITY * self.H_hx * water_density_dT(T_mix) / ne
        return B, Bp, g.ravel(), gp.ravel()

    def _Rp(self, Wp):
        return self.R_p if Wp >= 0 else self.R_p * self.k_rev

    def _calibrate_ncv(self, cal, T_in_default):
        """NCV resistance R_v such that the natural-circulation core flow equals the published
        value at the published power and inlet temperature (uniform axial power, equal
        channels, continuous temperature profile)."""
        Q, W_fuel = float(cal["power_w"]), float(cal["core_flow_kg_s"])
        T_in = float(cal.get("inlet_temperature_c", T_in_default - 273.15)) + 273.15
        dT = Q / (W_fuel * CP_WATER)
        x, w = np.polynomial.legendre.leggauss(64)
        z = 0.5 * (x + 1.0)                                  # fraction of the heated length
        H = self.dz.sum()
        rin = water_density(T_in)
        B = GRAVITY * (H * 0.5 * np.sum(w * (rin - water_density(T_in + dT * z)))
                       + self.H_ch * (rin - water_density(T_in + dT)))
        Bp = GRAVITY * self.H_hx * (rin - water_density(T_in + dT))
        Wc = W_fuel / self.fuel_frac
        minus_dp = B - self.R_c * Wc**2                      # = R_v W_v^2
        drive_p = minus_dp - Bp                              # = R_p(W_p) W_p |W_p|
        Wp = np.sign(drive_p) * np.sqrt(abs(drive_p) / (self.R_p if drive_p >= 0 else self.R_p * self.k_rev))
        if minus_dp <= 0 or Wp >= Wc:
            raise ValueError("primary_loop: natural-convection calibration is inconsistent "
                             f"(buoyancy {B:.1f} Pa, pump-loop flow {Wp:.3g} kg/s >= core flow {Wc:.3g})")
        self.calibration = dict(power_w=Q, T_in=T_in, core_flow=Wc, fuel_flow=W_fuel,
                                buoyancy_pa=B, pump_loop_flow=Wp)
        return minus_dp / (Wc - Wp) ** 2

    def init_steady(self, Tc, T_in):
        """Make the current forced-flow state an exact steady state: all pumps at full speed,
        NCVs shut, flow W0 with the buoyancy of the given temperatures."""
        self.reset()
        B, Bp = self.buoyancy(Tc, T_in)
        self.R_p = (self.dp0 - self.R_c * self.W0**2 + B - Bp) / self.W0**2

    def coastdown_speed(self, t):
        """Exact speed of a pump t s after a trip from full speed."""
        t = np.asarray(t, float)
        f, tau = self.f_fric, self.tau_cd
        if f == 0.0:
            return 1.0 / (1.0 + t / tau)
        r = np.sqrt(f)
        return np.where(t < self.stop_time, r * np.tan(np.arctan(1.0 / r) - r * t / tau), 0.0)

    @property
    def stop_time(self):
        f = self.f_fric
        return np.inf if f == 0.0 else self.tau_cd * np.arctan(1.0 / np.sqrt(f)) / np.sqrt(f)

    # ---------------- dynamics
    def derivs(self, omega, Wp, Wv, heads, is_open):
        """Time derivatives (omega', W_p', W_v') and the plenum pressure Dp [Pa];
        heads = (B, B_p) from buoyancy()."""
        B, Bp = heads[0], heads[1]
        m = np.maximum(omega, 0.0).mean() if self.n_pumps else 0.0
        Wc = Wp + Wv
        ic, ip, iv = 1.0 / self.I_c, 1.0 / self.I_p, (1.0 / self.I_v if is_open else 0.0)
        a_c = self.R_c * Wc * abs(Wc) - B
        a_p = self.dp0 * m * m - self._Rp(Wp) * Wp * abs(Wp) - Bp
        a_v = -self.R_v * Wv * abs(Wv) if is_open else 0.0
        dp = (a_c * ic + a_p * ip + a_v * iv) / (ic + ip + iv)
        domega = np.where(self.running, (self.target - omega) / self.tau_up,
                          np.where(omega > 0, -(omega * omega + self.f_fric) / self.tau_cd, 0.0))
        return domega, (a_p - dp) * ip, (a_v - dp) * iv, dp

    def partials(self, omega, Wp, Wv, is_open):
        """Jacobian pieces: d(omega')/d omega (diagonal), d(W_p', W_v')/d(omega_i, W_p, W_v) as
        a 2 x (n+2) array, and a 2 x 2 array F with d(W_p', W_v')/dT = F[:, 0] dB/dT + F[:, 1] dB_p/dT."""
        n = self.n_pumps
        m = np.maximum(omega, 0.0).mean() if n else 0.0
        Wc = Wp + Wv
        ic, ip, iv = 1.0 / self.I_c, 1.0 / self.I_p, (1.0 / self.I_v if is_open else 0.0)
        S = ic + ip + iv
        d_om = np.where(self.running, -1.0 / self.tau_up,
                        np.where(omega > 0, -2.0 * omega / self.tau_cd, 0.0))
        J = np.zeros((2, n + 2))
        dap = np.where(omega > 0, 2.0 * self.dp0 * m / n, 0.0)   # d a_p / d omega_i
        J[0, :n] = ip * (dap - ip * dap / S)
        J[1, :n] = iv * (-ip * dap / S)
        dac = 2.0 * self.R_c * abs(Wc)
        dapW = -2.0 * self._Rp(Wp) * abs(Wp)
        ddp = (ic * dac + ip * dapW) / S
        J[0, n], J[1, n] = ip * (dapW - ddp), iv * (-ddp)
        davW = -2.0 * self.R_v * abs(Wv) if is_open else 0.0
        ddp = (ic * dac + iv * davW) / S
        J[0, n + 1], J[1, n + 1] = ip * (-ddp), iv * (davW - ddp)
        # d a_c/dT = -dB/dT, d a_p/dT = -dB_p/dT  ->  d dp/dT = -(ic dB + ip dB_p) / S
        F = np.array([[ip * ic / S, ip * (ip / S - 1.0)],
                      [iv * ic / S, iv * ip / S]])
        return d_om, J, F

    def event_value(self, omega, Wp, Wv, heads, is_open):
        """Crosses zero (downward) when the NCVs switch: plenum pressure with the valves
        shut (opening), or flow through them (closing)."""
        if is_open:
            return Wv
        return self.derivs(omega, Wp, Wv, heads, False)[3]

    def advance(self, dt, heads, rtol=1e-10, max_switch=50):
        """Integrate the loop over dt with fixed buoyancy heads (used by the direct method)."""
        B = heads
        n = self.n_pumps
        y, t, is_open = self.get_state(), 0.0, self.open
        # The heads are piecewise constant (updated between steps), so a head update can move
        # the switching function past zero at a step boundary, where no crossing inside the
        # step would be seen. Switch at once in that case.
        if self.n_valves > 0 and self.event_value(y[:n], y[n], y[n + 1], B, is_open) < 0:
            is_open = not is_open
            if not is_open:
                y[n + 1] = 0.0
        for _ in range(max_switch):
            def f(_t, yy, o=is_open):
                do, dwp, dwv, _ = self.derivs(yy[:n], yy[n], yy[n + 1], B, o)
                return np.concatenate((do, [dwp, dwv if o else 0.0]))

            def jac(_t, yy, o=is_open):
                d_om, J, _ = self.partials(yy[:n], yy[n], yy[n + 1], o)
                M = np.zeros((n + 2, n + 2))
                M[np.arange(n), np.arange(n)] = d_om
                M[n] = J[0]
                if o:
                    M[n + 1] = J[1]
                return M

            events = None
            if self.n_valves > 0:
                def ev(_t, yy, o=is_open):
                    return self.event_value(yy[:n], yy[n], yy[n + 1], B, o)
                ev.terminal, ev.direction = True, -1
                events = [ev]
            sol = solve_ivp(f, (t, dt), y, method="Radau", jac=jac, rtol=rtol,
                            atol=np.concatenate((np.full(n, 1e-12), [1e-10 * self.W0] * 2)),
                            events=events, first_step=min(dt - t, 1e-4))
            if not sol.success:
                raise RuntimeError(sol.message)
            if events and sol.status == 1:
                t, y = float(sol.t_events[0][0]), sol.y_events[0][0].copy()
                is_open = not is_open
                if not is_open:
                    y[n + 1] = 0.0
                if dt - t > 1e-12:
                    continue
            else:
                y = sol.y[:, -1]
            break
        self.set_state(y, is_open)


class ChannelThermal:
    def __init__(self, model, cfg):
        th = cfg["thermal_hydraulics"]
        fb = cfg.get("feedback") or {}
        mesh = model.problem.mesh
        self.mesh = mesh
        z0 = float(cfg["core"]["fuel_bottom"])
        H = float(cfg["core"]["fuel_height"])

        # ---- nodes: fueled elements x mesh layers overlapping the fuel
        zl = mesh.z0 + mesh.dz * np.arange(mesh.nz)
        overlap = np.clip(np.minimum(zl + mesh.dz, z0 + H) - np.maximum(zl, z0), 0.0, None)
        self.layers = np.nonzero(overlap > 1e-9)[0]
        hfrac = overlap[self.layers] / H                    # share of the fueled length per layer
        W = model.W.tocoo()
        elem_tot = np.bincount(W.col, weights=W.data, minlength=model.W.shape[1])
        self.elems = np.nonzero(elem_tot > 0)[0]
        ne, nl = len(self.elems), len(self.layers)
        self.ne, self.nl, self.n = ne, nl, ne * nl
        ie_of = np.full(model.W.shape[1], -1)
        ie_of[self.elems] = np.arange(ne)
        il_of = np.full(mesh.nz, -1)
        il_of[self.layers] = np.arange(nl)
        kz = W.row % mesh.nz
        ok = (ie_of[W.col] >= 0) & (il_of[kz] >= 0)
        node = ie_of[W.col[ok]] * nl + il_of[kz[ok]]
        # M: cell -> node weights (rows sum to 1 for every fuel-bearing cell)
        self.M = sp.csr_matrix((W.data[ok], (W.row[ok], node)), shape=(mesh.n_cells, self.n))
        self.MT = self.M.T.tocsr()
        self.labels = [model.labels[e] for e in self.elems]

        # ---- per-node thermal parameters
        pe = th["per_element"]
        hf = np.tile(hfrac, ne)
        self.Cf = float(pe["plate_heat_capacity_j_k"]) * hf
        self.Cc = float(pe["coolant_heat_capacity_j_k"]) * hf
        self.hA = float(pe["hA_w_k"]) * hf
        self.dz_m = overlap[self.layers] / 100.0             # heated length per layer [m]
        self.W0 = float(th["total_flow_kg_s"])
        self.cp_e = float(th["fuel_flow_fraction"]) / ne * CP_WATER   # W/K per element per kg/s
        self.mcp_nominal = self.cp_e * self.W0               # W/K per element
        self.flow = 1.0                                      # relative flow
        self.T_in = float(th["inlet_temperature"])
        self.below = np.arange(self.n) - 1                   # node feeding each node
        self.below[np.arange(ne) * nl] = -1                  # bottom nodes: fed by the inlet

        # ---- feedback data
        self.T_ref = float(fb.get("reference_temperature", self.T_in))
        frac_base = model.mix["frac_base"]
        mats = list(cfg["materials"])
        self.fb = []
        for mname, d in (fb.get("materials") or {}).items():
            f = frac_base[..., mats.index(mname)].ravel()
            vals = {x: float(cfg["materials"][mname].get(x, 0.0)) for x in XS_NAMES}
            cc = {x: float(v) for x, v in (d.get("coolant_temperature") or {}).items()}
            ff = {x: float(v) for x, v in (d.get("fuel_temperature") or {}).items()}
            for x in list(cc) + list(ff):
                if x not in XS_NAMES:
                    raise ValueError(f"unknown cross section '{x}' in feedback for {mname}")
            self.fb.append((mname, f, vals, cc, ff))

        self.Tf = np.full(self.n, self.T_in)
        self.Tc = np.full(self.n, self.T_in)

        # ---- optional decay heat and primary-loop hydraulics
        self.decay = DecayHeat(cfg["decay_heat"], self.n) if cfg.get("decay_heat") else None
        self.loop = PrimaryLoop(cfg, self) if cfg.get("primary_loop") else None
        self.E_in = 0.0          # heat deposited in the nodes since the start [J]
        self.E_out = 0.0         # heat carried out by the coolant since the start [J]

    # ------------------------------------------------------------------ conveniences
    @property
    def mcp(self):
        return self.mcp_nominal * self.flow

    def node_power(self, fission_per_cell, scale):
        """Node powers [W] from a per-cell fission-rate vector; `scale` converts to watts."""
        return scale * (self.MT @ fission_per_cell)

    def heat(self, q_fission):
        """Heat deposited per node [W]: prompt fission energy plus decay heat."""
        return q_fission if self.decay is None else self.decay.heat(q_fission)

    @property
    def decay_power(self):
        """Total fission-product decay heat [W] (0 without a decay-heat model)."""
        return 0.0 if self.decay is None else float(self.decay.D.sum())

    def steady_state(self, q, T_in=None, flow=None):
        """Exact steady temperatures for node powers q [W]."""
        T_in = self.T_in if T_in is None else T_in
        mcp = self.mcp_nominal * (self.flow if flow is None else flow)
        qe = q.reshape(self.ne, self.nl)
        Tc = T_in + np.cumsum(qe, axis=1) / mcp
        Tf = Tc + qe / self.hA.reshape(self.ne, self.nl)
        return Tf.ravel(), Tc.ravel()

    def stored_energy(self, Tf=None, Tc=None):
        Tf = self.Tf if Tf is None else Tf
        Tc = self.Tc if Tc is None else Tc
        return np.sum(self.Cf * Tf + self.Cc * Tc)

    def outflow_power(self, Tc=None):
        """Heat carried out of the core by the coolant [W] (relative to the inlet)."""
        Tc = self.Tc if Tc is None else Tc
        return self.mcp * np.sum(Tc.reshape(self.ne, self.nl)[:, -1] - self.T_in)

    def step_be(self, dt, q):
        """Backward-Euler update of all node temperatures with end-of-step powers q [W].
        Exactly energy-conserving: q dt = d(stored) + outflow dt, step by step."""
        ne, nl = self.ne, self.nl
        Tf0, Tc0 = self.Tf.reshape(ne, nl), self.Tc.reshape(ne, nl)
        Cf, Cc, hA = (a.reshape(ne, nl) for a in (self.Cf, self.Cc, self.hA))
        qe = q.reshape(ne, nl)
        Tf, Tc = np.empty_like(Tf0), np.empty_like(Tc0)
        T_below = np.full(ne, self.T_in)
        mcp = self.mcp
        for j in range(nl):
            a11, a12 = Cf[:, j] / dt + hA[:, j], -hA[:, j]
            a21, a22 = -hA[:, j], Cc[:, j] / dt + hA[:, j] + mcp
            b1 = Cf[:, j] / dt * Tf0[:, j] + qe[:, j]
            b2 = Cc[:, j] / dt * Tc0[:, j] + mcp * T_below
            det = a11 * a22 - a12 * a21
            Tf[:, j] = (b1 * a22 - a12 * b2) / det
            Tc[:, j] = (a11 * b2 - a21 * b1) / det
            T_below = Tc[:, j]
        self.Tf, self.Tc = Tf.ravel(), Tc.ravel()

    # ------------------------------------------------------------------ feedback on xs
    def apply(self, xs, Tf=None, Tc=None):
        """Cross sections with temperature feedback (returns new arrays; xs is not modified)."""
        Tf = self.Tf if Tf is None else Tf
        Tc = self.Tc if Tc is None else Tc
        shape = self.mesh.shape
        dTf = self.M @ (Tf - self.T_ref)
        dTc = self.M @ (Tc - self.T_ref)
        out = {k: np.asarray(v, float).ravel().copy() for k, v in xs.items()}
        for _, f, vals, cc, ff in self.fb:
            for x in set(cc) | set(ff):
                r = cc.get(x, 0.0) * dTc + ff.get(x, 0.0) * dTf
                if x in ("D1", "D2"):
                    D0 = vals[x]
                    out[x] = 1.0 / (1.0 / out[x] + f * (1.0 / (D0 * (1 + r)) - 1.0 / D0))
                else:
                    out[x] = out[x] + f * vals[x] * r
        return {k: v.reshape(shape) for k, v in out.items()}

    def rho_sensitivity(self, xs, k0, w1, w2, psi1, psi2, bc, Tf=None, Tc=None):
        """Exact first-order d(rho)/d(T_f,k) and d(rho)/d(T_c,k) for every node, for
        rho = 1 - <w, L psi> / <w, F psi> evaluated with cross sections xs (which must
        already include the feedback at Tf, Tc)."""
        Tf = self.Tf if Tf is None else Tf
        Tc = self.Tc if Tc is None else Tc
        x = {k: np.asarray(v, float).ravel() for k, v in xs.items()}
        wF = w1 @ ((x["nSf1"] * psi1 + x["nSf2"] * psi2) / k0)
        wL = (w1 @ ((x["Sa1"] + x["Ss12"]) * psi1) + w2 @ (x["Sa2"] * psi2) - w2 @ (x["Ss12"] * psi1)
              + _diffusion_form(self.mesh, x["D1"], w1, psi1, bc)
              + _diffusion_form(self.mesh, x["D2"], w2, psi2, bc))
        rho = 1.0 - wL / wF
        # d<w,L psi>/d xs (per cell) and d<w,F psi>/d xs
        dL = {"Sa1": w1 * psi1, "Ss12": (w1 - w2) * psi1, "Sa2": w2 * psi2,
              "D1": _diffusion_form_dD(self.mesh, x["D1"], w1, psi1, bc),
              "D2": _diffusion_form_dD(self.mesh, x["D2"], w2, psi2, bc)}
        dF = {"nSf1": w1 * psi1 / k0, "nSf2": w1 * psi2 / k0}
        drho = {k: (-dL.get(k, 0.0) + (1.0 - rho) * dF.get(k, 0.0)) / wF for k in XS_NAMES}

        dTf_cell = self.M @ (Tf - self.T_ref)
        dTc_cell = self.M @ (Tc - self.T_ref)
        Sf = np.zeros(self.mesh.n_cells)
        Sc = np.zeros(self.mesh.n_cells)
        for _, f, vals, cc, ff in self.fb:
            for k in set(cc) | set(ff):
                if k in ("D1", "D2"):
                    D0 = vals[k]
                    r = cc.get(k, 0.0) * dTc_cell + ff.get(k, 0.0) * dTf_cell
                    Dm = D0 * (1 + r)
                    dD_dr = x[k] ** 2 * f * D0 / Dm**2          # d D_cell / d r
                else:
                    dD_dr = f * vals[k]                          # d Sigma_cell / d r
                Sc += drho[k] * dD_dr * cc.get(k, 0.0)
                Sf += drho[k] * dD_dr * ff.get(k, 0.0)
        return self.MT @ Sf, self.MT @ Sc, rho


# ----------------------------------------------------------------------------- helpers
def _faces(mesh):
    for axis, h, (lo_f, hi_f) in ((0, mesh.dx, ("x-", "x+")), (1, mesh.dy, ("y-", "y+")),
                                  (2, mesh.dz, ("z-", "z+"))):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis], hi[axis] = slice(0, -1), slice(1, None)
        first = [slice(None)] * 3
        last = [slice(None)] * 3
        first[axis], last[axis] = 0, -1
        yield h, tuple(lo), tuple(hi), (tuple(first), lo_f), (tuple(last), hi_f)


def _bc_coeff(D, h, bc):
    if bc[0] == "reflective":
        return np.zeros_like(D), np.zeros_like(D)
    if bc[0] == "zero":
        return 2 * D / h**2, np.full_like(D, 2 / h**2)
    g = bc[1]
    den = h / (2 * D) + 1 / g
    return 1 / (h * den), (h / (2 * D**2)) / (h * den**2)


def _diffusion_form(mesh, D, w, psi, bc):
    """<w, -div D grad psi> = sum over faces c (w_i - w_j)(psi_i - psi_j) + boundary terms."""
    D3, w3, p3 = (a.reshape(mesh.shape) for a in (D, w, psi))
    tot = 0.0
    for h, lo, hi, *bnd in _faces(mesh):
        c = 2 * D3[lo] * D3[hi] / ((D3[lo] + D3[hi]) * h**2)
        tot += np.sum(c * (w3[lo] - w3[hi]) * (p3[lo] - p3[hi]))
        for sl, face in bnd:
            cb, _ = _bc_coeff(D3[sl], h, bc[face])
            tot += np.sum(cb * w3[sl] * p3[sl])
    return tot


def _diffusion_form_dD(mesh, D, w, psi, bc):
    """Per-cell derivative of _diffusion_form with respect to that cell's D."""
    D3, w3, p3 = (a.reshape(mesh.shape) for a in (D, w, psi))
    out = np.zeros(mesh.shape)
    for h, lo, hi, *bnd in _faces(mesh):
        Dl, Dh = D3[lo], D3[hi]
        g = (w3[lo] - w3[hi]) * (p3[lo] - p3[hi])
        den = (Dl + Dh) ** 2 * h**2
        out[lo] += 2 * Dh**2 / den * g
        out[hi] += 2 * Dl**2 / den * g
        for sl, face in bnd:
            _, dcb = _bc_coeff(D3[sl], h, bc[face])
            out[sl] += dcb * w3[sl] * p3[sl]
    return out.ravel()
