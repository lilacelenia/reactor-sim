"""
live_sim.py -- the quasi-static transient solver running live in a background thread.

No GUI code: the 3D viewer (core_viewer.py --live) drives it, and it can be tested headless.

Time compression
    The simulation clock is paced to   sim time = wall time x speed.
    Each step's size is chosen so the simulation lands just ahead of that target, then the
    thread waits for the wall clock. If the computer cannot keep up, the simulation runs as
    fast as it can and `achieved_speed` shows the real rate; the backlog is capped, so
    lowering the speed takes effect at once instead of racing to catch up.

Operator commands (thread-safe, from any thread)
    command_blades(target_cm), scram(), set_inlet_temperature(K), set_flow(relative),
    count(seconds) / clear_one_over_m()   (1/M counting on the source-range channel),
    set_speed(x), set_running(bool), reset()  (reset re-initializes and leaves it paused)
    They take effect at the current simulation time, at the next step boundary. Blades move
    toward the target at the drive speed; after a scram, blade commands are ignored until
    reset().

Startup mode (startup=True): the core starts subcritical with the blades in, driven by the
Pu-Be startup source, in absolute units; power follows from the source. The instruments
(instruments.py) give count rates, periods and power channels, and trip the reactor on
short period or high flux.

Reading the state (from any thread)
    snapshot()  latest LiveSnapshot (immutable copies) or None before initialization
    trends()    dict of arrays for trend plots (time, power, temperatures, ...)
    status      "initializing" | "paused" | "running" | "running (lagging)" | "error: ..."
"""
import queue
import threading
import time
import traceback
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from instruments import InstrumentSystem, source_from_config
from kinetics3d import BladeProgram, PlantProgram, TransientSolver, kinetics_from_config
from thermal import ChannelThermal


@dataclass(frozen=True)
class LiveSnapshot:
    seq: int
    t: float                    # simulation time [s]
    withdrawn: float            # actual blade position [cm]
    target: float               # commanded blade position [cm]
    power_w: float
    period: float               # [s], +-inf when steady
    rho: float                  # net reactivity [dk/k]
    rho_feedback: float         # feedback reactivity change since start [dk/k]
    element_power: np.ndarray   # per element / core-average (0 for dummies) -> power shape
    element_power_w: np.ndarray # per element [W]
    element_temp: np.ndarray    # hottest plate node per element [C] (NaN for dummies)
    power_density: np.ndarray   # 3D, relative to the fissile-cell average (NaN outside fuel)
    phi1: np.ndarray            # 3D fast flux, max = 1
    phi2: np.ndarray            # 3D thermal flux, same scale
    fuel_max_c: float
    outlet_c: float
    outlet_max_c: float
    inlet_c: float
    flow: float
    scrammed: bool
    trip_reason: str
    model: object               # CoreModel (for axial profiles)
    result: object              # .phi1/.phi2 as 3D arrays (for CoreModel.element_axial_profile)
    instruments: tuple = ()     # instruments.ChannelReading per channel
    counting: object = None     # (elapsed, total) seconds of an active 1/M count
    one_over_m: tuple = ()      # (blade, cps, sigma, 1/M, sigma_1/M) per count
    ecp: object = None          # estimated critical position [cm] from the 1/M points
    k_initial: object = None    # eigenvalue of the starting configuration (startup mode)


class LiveSimulation:
    def __init__(self, cfg, model, power_w=None, start_height=None, method="pcqs", feedback=True,
                 trip_power_w="auto", dt_max=5.0, dh_max=1.0, max_backlog_wall=2.0, tol=1e-6,
                 startup=False):
        tr = cfg.get("transient", {}) or {}
        bank = cfg["control"]["shim_bank"]
        self.cfg, self.model = cfg, model
        # startup: subcritical core driven by the startup source (absolute units)
        self.startup = bool(startup)
        self.source = source_from_config(cfg, model) if self.startup else None
        if self.startup and self.source is None:
            raise ValueError("startup mode needs a 'source' section in the configuration")
        self.P0 = float(power_w if power_w is not None else tr.get("initial_power_w", 1e6))
        default_h = (cfg.get("startup") or {}).get("start_height", 0.0) if self.startup else \
            tr.get("start_height", bank["withdrawn"])
        self.h0 = float(start_height if start_height is not None else default_h)
        self.has_instruments = "instrumentation" in cfg
        self.method = method
        self.feedback = bool(feedback) and "thermal_hydraulics" in cfg
        if trip_power_w == "auto":
            # trips come from the instrumentation model when the config has one; otherwise
            # a simple high-power trip at 120 % of rated power
            rated = (cfg.get("thermal_hydraulics") or {}).get("rated_power_w")
            trip_power_w = None if self.has_instruments else \
                (tr.get("trip_power_w") or (1.2 * float(rated) if rated else None))
        self.trip_power_w = None if trip_power_w is None else float(trip_power_w)
        self.dt_max, self.dh_max = dt_max, dh_max
        self.tol = tol      # 1e-6 is ~20 % faster than 1e-7 and changes power by < 1e-9
        self.max_backlog_wall = max_backlog_wall
        self.kin = kinetics_from_config(cfg)
        self.drive_speed = float(bank["drive_speed"])
        self.scram_speed = float(bank["scram_speed"])
        self.travel = float(bank["travel"])

        self._cmds = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._snap = None
        self._seq = 0
        self._trend = {k: [] for k in ("t", "power_w", "blade", "fuel_max_c", "outlet_c", "inlet_c",
                                      "rho", "rho_feedback", "ch1_cps")}
        self.speed = 1.0
        self.running = False
        self.status = "idle"
        self.error = None
        self.achieved_speed = 0.0
        self.solver = None
        self.inst = None
        self._thread = threading.Thread(target=self._run, name="live-sim", daemon=True)

    # ------------------------------------------------------------------ public API
    def start(self, running=False):
        self.running = running
        self._thread.start()

    def shutdown(self, timeout=30.0):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout)

    def set_running(self, flag):
        self._cmds.put(("run", bool(flag)))

    def set_speed(self, x):
        self._cmds.put(("speed", float(x)))

    def command_blades(self, target_cm):
        self._cmds.put(("blades", float(target_cm)))

    def scram(self):
        self._cmds.put(("scram", None))

    def set_inlet_temperature(self, T_kelvin):
        self._cmds.put(("inlet_temperature", float(T_kelvin)))

    def set_flow(self, relative):
        self._cmds.put(("flow", float(relative)))

    def reset(self):
        self._cmds.put(("reset", None))

    def count(self, seconds=30.0):
        """Count the source-range channel for `seconds` of simulated time -> one 1/M point."""
        self._cmds.put(("count", float(seconds)))

    def clear_one_over_m(self):
        self._cmds.put(("clear_1m", None))

    def snapshot(self):
        with self._lock:
            return self._snap

    def trends(self):
        with self._lock:
            return {k: np.array(v) for k, v in self._trend.items()}

    @property
    def initialized(self):
        return self._snap is not None

    # ------------------------------------------------------------------ worker thread
    def _run(self):
        try:
            self._init_solver()
            while not self._stop.is_set():
                if self._drain_commands():          # reset requested
                    self._init_solver()
                    continue
                if not self.running:
                    self.status = "paused"
                    time.sleep(0.05)
                    continue
                self._advance()
        except Exception as exc:  # noqa: BLE001 -- report to the GUI instead of dying silently
            self.error = traceback.format_exc()
            self.status = f"error: {exc}"

    def _init_solver(self):
        self.status = "initializing"
        with self._lock:
            self._snap = None
            for v in self._trend.values():
                v.clear()
        self.prog = BladeProgram(self.h0, self.drive_speed, self.scram_speed, self.travel)
        self.plant = PlantProgram()
        self.target = self.h0
        th = ChannelThermal(self.model, self.cfg) if self.feedback else None
        trip = None if (self.trip_power_w is None or self.startup) else self.trip_power_w / self.P0
        if self.has_instruments and self.inst is None:
            self.inst = InstrumentSystem(self.cfg, self.model, self.kin)
        self.solver = TransientSolver(self.model, self.prog, kin=self.kin, method=self.method,
                                      dt_max=self.dt_max, dh_max=self.dh_max, thermal=th,
                                      power_w=self.P0, plant=self.plant, trip_power=trip,
                                      tol=self.tol, source=self.source)
        if self.startup:
            self.P0 = self.solver.P0                 # set by the source, not chosen
        if self.inst is not None:
            self.inst.reset()
            self.inst.update(self.solver)
        self._step_wall = 0.5                       # running estimate of wall time per step
        self._speed_log = [(time.monotonic(), self.solver.t)]
        self._anchor()
        self._publish()

    def _anchor(self):
        self._wall_ref = time.monotonic()
        self._sim_ref = self.solver.t if self.solver else 0.0

    def _drain_commands(self):
        s = self.solver
        while True:
            try:
                kind, value = self._cmds.get_nowait()
            except queue.Empty:
                return False
            if kind == "reset":
                self.running = False             # a reset always leaves the simulation paused
                # drop anything queued behind the reset; it referred to the old run
                while True:
                    try:
                        self._cmds.get_nowait()
                    except queue.Empty:
                        break
                return True
            if kind == "run":
                self.running = value
                self._anchor()
            elif kind == "speed":
                self.speed = max(value, 1e-3)
                self._anchor()
            elif kind == "blades":
                if s.prog.scram_time is None:
                    self.target = float(np.clip(value, 0.0, self.travel))
                    s.prog.command(s.t, self.target)
            elif kind == "scram":
                s.prog.scram(s.t)
                s.hist.events.append((s.t, "manual scram"))
            elif kind in ("inlet_temperature", "flow"):
                s.plant.set(s.t, kind, value)
            elif kind == "count" and self.inst is not None:
                self.inst.start_count(s, value)
            elif kind == "clear_1m" and self.inst is not None:
                self.inst.one_over_m.clear()

    def _advance(self):
        s = self.solver
        now = time.monotonic()
        target = self._sim_ref + (now - self._wall_ref) * self.speed
        lag = target - s.t
        # behind by more than ~2 steps (or 0.5 s) of wall time -> the computer cannot keep up
        lagging = lag > self.speed * max(0.5, 2.0 * self._step_wall)
        if lag > self.speed * self.max_backlog_wall:
            # cap the backlog so the sim never "races" to catch up once it can
            self._wall_ref = now - self.max_backlog_wall
            self._sim_ref = s.t
            lag = self.speed * self.max_backlog_wall
        if lag <= 0.0:
            self.status = "running"
            time.sleep(min(0.05, -lag / self.speed + 1e-3))
            return
        # aim one step ahead of the wall clock; never beyond dt_max
        cap = min(self.dt_max, lag + self.speed * self._step_wall)
        dt = s._choose_dt(s.t + cap)
        self.status = "running (lagging)" if lagging else "running"
        t0 = time.monotonic()
        s.step(dt)
        if self.inst is not None:
            trip = self.inst.update(s)
            if trip and s.prog.scram_time is None:
                s.prog.scram(s.t)
                s.hist.events.append((s.t, trip))
        w = time.monotonic() - t0
        self._step_wall = 0.7 * self._step_wall + 0.3 * w
        self._speed_log.append((time.monotonic(), s.t))
        while len(self._speed_log) > 2 and self._speed_log[-1][0] - self._speed_log[1][0] > 5.0:
            self._speed_log.pop(0)
        (w0, t0s), (w1, t1s) = self._speed_log[0], self._speed_log[-1]
        self.achieved_speed = (t1s - t0s) / max(w1 - w0, 1e-9)
        self._publish()

    def _publish(self):
        s = self.solver
        m = self.model
        h = s.hist
        S = s.ops.fission(s.phi1, s.phi2)
        Pe = m.W.T @ S
        fueled = Pe > 0
        shape = np.where(fueled, Pe / Pe[fueled].mean(), 0.0)
        power_w = h.power[-1] * self.P0
        fissile = s.ops.nsf2 > 0
        pd = np.where(fissile, S / S[fissile].mean(), np.nan).reshape(m.problem.mesh.shape)
        norm = s.phi1.max()
        phi1 = (s.phi1 / norm).reshape(m.problem.mesh.shape)
        phi2 = (s.phi2 / norm).reshape(m.problem.mesh.shape)
        if len(h.t) >= 2 and h.power[-1] > 0 and h.power[-2] > 0 and h.power[-1] != h.power[-2]:
            period = (h.t[-1] - h.t[-2]) / np.log(h.power[-1] / h.power[-2])
        else:
            period = np.inf
        th = s.thermal
        elem_T = np.full(len(m.labels), np.nan)
        if th is not None:
            elem_T[th.elems] = th.Tf.reshape(th.ne, th.nl).max(axis=1) - 273.15
            fuel_max, out, out_max = h.fuel_max[-1], h.coolant_outlet[-1], h.coolant_outlet_max[-1]
            inlet, flow, rho_fb = th.T_in, th.flow, h.rho_feedback[-1] - h.rho_feedback[0]
        else:
            fuel_max = out = out_max = inlet = np.nan
            flow, rho_fb = 1.0, 0.0
        trip = next((txt for _, txt in reversed(h.events)), "") if s.prog.scram_time is not None else ""
        self._seq += 1
        snap = LiveSnapshot(
            seq=self._seq, t=s.t, withdrawn=s.prog.position(s.t), target=self.target,
            power_w=power_w, period=period, rho=s.rho, rho_feedback=rho_fb,
            element_power=shape, element_power_w=Pe * s.watt_per_fission, element_temp=elem_T,
            power_density=pd, phi1=phi1, phi2=phi2,
            fuel_max_c=fuel_max - 273.15, outlet_c=out - 273.15, outlet_max_c=out_max - 273.15,
            inlet_c=inlet - 273.15, flow=flow, scrammed=s.prog.scram_time is not None,
            trip_reason=trip, model=m, result=SimpleNamespace(phi1=phi1, phi2=phi2),
            instruments=tuple(self.inst.readings) if self.inst is not None else (),
            counting=self.inst.counting if self.inst is not None else None,
            one_over_m=tuple(self.inst.one_over_m.points) if self.inst is not None else (),
            ecp=self.inst.one_over_m.ecp() if self.inst is not None else None,
            k_initial=getattr(s, "k_initial", None))
        with self._lock:
            self._snap = snap
            tr = self._trend
            tr["t"].append(s.t)
            tr["power_w"].append(power_w)
            tr["blade"].append(snap.withdrawn)
            tr["fuel_max_c"].append(snap.fuel_max_c)
            tr["outlet_c"].append(snap.outlet_c)
            tr["inlet_c"].append(snap.inlet_c)
            tr["rho"].append(s.rho)
            tr["rho_feedback"].append(rho_fb)
            fc = [r.expected for r in snap.instruments if r.kind == "fission_chamber"]
            tr["ch1_cps"].append(fc[0] if fc else np.nan)
