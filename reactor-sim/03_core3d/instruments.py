"""
instruments.py -- nuclear instrumentation, trips and the 1/M approach to criticality.

Modeled on the published MITR nuclear safety system (see the YAML for sources):
  * Channels 1-2: source-range fission chambers (pulse counting) for startup, used as
    period channels; above ~1 kW the period function switches to an ion chamber, and the
    fission chambers stop working properly above ~5 kW (counting losses).
  * Channel 3: period channel on an ion chamber, on scale from ~500 W.
  * Channels 4-6: linear power (ion chambers); 5 and 6 have low-range amplifiers < 100 kW.
  * Scrams: short period 10 s; high flux 6.5 MW, or 80 kW without primary flow.
Detector positions, sensitivities, dead time, meter time constants and the startup source
strength are estimates.

Each detector samples the thermal flux in a small region of the mesh. Its signal is
calibrated in watts at a reference critical configuration ("heat balance"), so it reads
true power there and drifts with the flux shape elsewhere -- e.g. near the startup source,
or when the blades move. Fission chambers turn that into a count rate with dead-time losses
and counting statistics.
"""
from dataclasses import dataclass, field

import numpy as np

from diffusion3d import solve_k


@dataclass
class ChannelReading:
    name: str
    kind: str              # "fission_chamber" | "ion_chamber"
    value: float           # count rate [cps] (fission chamber, with meter noise) or power [W]
    expected: float        # noise-free value
    on_scale: bool
    status: str            # "", "below range", "off scale high", "saturated"
    period: float = np.inf # [s], period channels only (inf = steady / not a period channel)

    def text(self):
        if self.kind == "fission_chamber":
            v = f"{self.value:,.1f} cps" if self.value < 100 else f"{self.value:,.0f} cps"
        else:
            v = _fmt_w(self.value)
        if self.status:
            v += f"  [{self.status}]"
        return v


@dataclass
class OneOverM:
    """Inverse multiplication from counted rates: 1/M = CR_ref / CR."""
    points: list = field(default_factory=list)       # (blade cm, count rate, sigma, 1/M, sigma_1/M)

    def add(self, blade, counts, seconds):
        cr = counts / seconds
        sig = np.sqrt(max(counts, 1)) / seconds
        if not self.points:
            ref, ref_sig = cr, sig
        else:
            ref, ref_sig = self.points[0][1], self.points[0][2]
        inv = ref / cr
        inv_sig = inv * np.hypot(sig / cr, ref_sig / ref) if self.points else 0.0
        self.points.append((float(blade), cr, sig, inv, inv_sig))

    def ecp(self, n_fit=3):
        """Estimated critical position: weighted straight-line fit of 1/M against blade
        height through the last n_fit points, extrapolated to 1/M = 0 (None if the trend
        does not point down)."""
        if len(self.points) < 2:
            return None
        pts = np.array([(h, m, max(sm, 0.01 * m, 1e-6)) for h, _, _, m, sm in self.points[-n_fit:]])
        h, m, sm = pts.T
        if np.ptp(h) == 0:
            return None
        w = 1.0 / sm**2
        A = np.vstack([np.ones_like(h), h]).T * np.sqrt(w)[:, None]
        (b0, b1), *_ = np.linalg.lstsq(A, m * np.sqrt(w), rcond=None)
        if b1 >= 0:
            return None
        return -b0 / b1

    def clear(self):
        self.points.clear()


class InstrumentSystem:
    def __init__(self, cfg, model, kin, seed=1):
        ic = cfg["instrumentation"]
        self.cfg, self.model, self.kin = cfg, model, kin
        mesh = model.problem.mesh
        self.rng = np.random.default_rng(seed)
        xc, yc, zc = (mesh.centers(a) for a in "xyz")
        X, Y, Z = np.meshgrid(xc, yc, zc, indexing="ij")
        self.det = {}
        for name, d in ic["detectors"].items():
            p, r = np.asarray(d["position"], float), float(d.get("radius", mesh.dx))
            dist = np.sqrt((X - p[0]) ** 2 + (Y - p[1]) ** 2 + (Z - p[2]) ** 2).ravel()
            sel = np.nonzero(dist <= r)[0]
            if sel.size == 0:
                sel = np.array([int(np.argmin(dist))])
            self.det[name] = sel                 # cells averaged by this detector

        # calibration: reference critical-ish eigen state, xs at reference temperatures
        h_cal = float(ic.get("calibration_height", cfg.get("transient", {}).get("start_height", 25.9)))
        prob = model.problem_at(h_cal)
        r = solve_k(prob, tol_k=1e-8, tol_src=1e-7)
        S = (prob.xs["nSf1"] * r.phi1 + prob.xs["nSf2"] * r.phi2).ravel()
        phi2 = r.phi2.ravel()
        self.g_ref = {n: phi2[sel].mean() / S.sum() for n, sel in self.det.items()}

        tr = ic["trips"]
        self.short_period = float(tr["short_period_s"])
        self.high_power = float(tr["high_power_w"])
        self.high_power_no_flow = float(tr["high_power_no_flow_w"])
        self.no_flow_below = float(tr.get("no_flow_below", 0.05))
        self.tau_period = float(ic.get("period_meter_time_constant_s", 1.0))
        self.tau_meter = float(ic.get("count_rate_meter_time_constant_s", 2.0))
        self.channels = ic["channels"]
        for ch in self.channels:
            if ch["detector"] not in self.det:
                raise ValueError(f"channel {ch['name']}: unknown detector {ch['detector']}")
            if ch["kind"] == "fission_chamber":
                ch["_cps_per_w"] = float(ch["full_scale_cps"]) / float(ch["full_scale_w"])
        self.reset()

    def reset(self):
        self._last = None                 # (t, {detector: signal_w})
        self._omega = {ch["name"]: 0.0 for ch in self.channels if ch.get("period")}
        self._count = None                # active counting interval
        self.one_over_m = OneOverM()
        self.readings = []
        self.trip = None

    # ------------------------------------------------------------------ signals
    def detector_signals(self, solver):
        """Detector signals in watts (true power x local flux shape factor vs calibration)."""
        ops = solver.ops
        phi1, phi2 = solver.phi1, solver.phi2
        S_phys = (ops.nsf1 * phi1 + ops.nsf2 * phi2).sum() * solver.k0
        P = solver.hist.power[-1] * solver.P0
        return {n: P * (phi2[sel].mean() / S_phys) / self.g_ref[n] for n, sel in self.det.items()}

    def _fc_rate(self, ch, signal_w):
        n = ch["_cps_per_w"] * signal_w                       # true count rate
        tau = float(ch.get("dead_time_s", 0.0))
        return n / (1.0 + n * tau)                            # non-paralyzable dead time

    def update(self, solver):
        """Call after every solver step. Returns a trip message or None."""
        t = solver.t
        sig = self.detector_signals(solver)
        flow = solver.thermal.flow if solver.thermal is not None else 1.0
        dt = None if self._last is None else t - self._last[0]
        readings = []
        trip = None
        for ch in self.channels:
            s = sig[ch["detector"]]
            name = ch["name"]
            if ch["kind"] == "fission_chamber":
                m = self._fc_rate(ch, s)
                sd = np.sqrt(m / (2 * self.tau_meter)) if m > 0 else 0.0   # RC count-rate meter
                shown = max(m + sd * self.rng.standard_normal(), 0.0)
                hi = s > float(ch["full_scale_w"])
                status = "saturated" if hi else ("below range" if m < float(ch.get("min_cps", 0.1)) else "")
                rd = ChannelReading(name, ch["kind"], shown, m, not hi and status == "", status)
            else:
                lo = float(ch.get("on_scale_w", 0.0))
                status = "below range" if s < lo else ("off scale high" if s > float(ch.get("full_scale_w", 1e9)) else "")
                rd = ChannelReading(name, ch["kind"], s, s, status == "", status)
            if ch.get("period"):
                # period from the log of the signal, through a first-order meter lag;
                # fission-chamber period channels hand over to an ion chamber above switch_w
                used = s
                if dt is not None and dt > 0 and self._last[1].get(ch["detector"], 0) > 0 and used > 0:
                    w_raw = np.log(used / self._last[1][ch["detector"]]) / dt
                    a = 1.0 - np.exp(-dt / self.tau_period)
                    self._omega[name] += a * (w_raw - self._omega[name])
                active = s >= float(ch.get("period_on_scale_w", 0.0))
                w = self._omega[name]
                rd.period = 1.0 / w if abs(w) > 1e-12 else np.inf
                if active and 0 < rd.period < self.short_period and trip is None:
                    trip = f"short-period scram ({name}: period {rd.period:.1f} s < {self.short_period:g} s)"
            if ch.get("power"):
                limit = self.high_power_no_flow if flow < self.no_flow_below else self.high_power
                if s > limit and trip is None:
                    trip = (f"high-flux scram ({name}: {_fmt_w(s)} > {_fmt_w(limit)}"
                            f"{' without primary flow' if flow < self.no_flow_below else ''})")
            readings.append(rd)

        # counting for 1/M: integrate the (dead-time corrected) rate, then draw Poisson counts
        if self._count is not None and dt:
            c = self._count
            ch = c["channel"]
            m_now = self._fc_rate(ch, sig[ch["detector"]])
            c["expected"] += 0.5 * (c["m_last"] + m_now) * dt
            c["m_last"] = m_now
            c["elapsed"] += dt
            if c["elapsed"] >= c["seconds"] - 1e-9:
                counts = self.rng.poisson(c["expected"])
                self.one_over_m.add(solver.prog.position(t), counts, c["elapsed"])
                self._count = None
        self._last = (t, sig)
        self.readings = readings
        if trip and self.trip is None:
            self.trip = trip
        return trip

    # ------------------------------------------------------------------ 1/M counting
    def start_count(self, solver, seconds=30.0, channel=None):
        ch = next(c for c in self.channels if c["kind"] == "fission_chamber"
                  and (channel is None or c["name"] == channel))
        sig = self.detector_signals(solver)
        self._count = dict(channel=ch, seconds=float(seconds), elapsed=0.0, expected=0.0,
                           m_last=self._fc_rate(ch, sig[ch["detector"]]))

    @property
    def counting(self):
        """(elapsed, total) seconds of the active count, or None."""
        return None if self._count is None else (self._count["elapsed"], self._count["seconds"])


def run_with_trips(solver, inst, t_end):
    """Advance the solver to t_end with the instruments watching; a trip scrams the blades."""
    if inst._last is None:
        inst.update(solver)

    def watch(s):
        trip = inst.update(s)
        if trip and s.prog.scram_time is None:
            s.prog.scram(s.t)
            s.hist.events.append((s.t, trip))

    return solver.run(t_end, callback=watch)


def source_from_config(cfg, model):
    """Per-cell source density [n/(cm^3 s)] from the 'source' section (None if absent)."""
    sc = cfg.get("source")
    if not sc:
        return None
    mesh = model.problem.mesh
    xc, yc, zc = (mesh.centers(a) for a in "xyz")
    X, Y, Z = np.meshgrid(xc, yc, zc, indexing="ij")
    p, r = np.asarray(sc["position"], float), float(sc.get("radius", mesh.dx))
    dist = np.sqrt((X - p[0]) ** 2 + (Y - p[1]) ** 2 + (Z - p[2]) ** 2).ravel()
    sel = np.nonzero(dist <= r)[0]
    if sel.size == 0:
        sel = np.array([int(np.argmin(dist))])
    Q = np.zeros(mesh.n_cells)
    Q[sel] = float(sc["strength_n_s"]) / sel.size / mesh.cell_volume
    return Q


def _fmt_w(w):
    for scale, unit in ((1e6, "MW"), (1e3, "kW"), (1.0, "W"), (1e-3, "mW"), (1e-6, "uW")):
        if abs(w) >= scale:
            return f"{w / scale:.3g} {unit}"
    return f"{w:.2g} W"
