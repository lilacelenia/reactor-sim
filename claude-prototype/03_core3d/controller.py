"""
controller.py -- automatic power controller acting on the regulating rod (no GUI code).

Sampled-data period-demand law (constants in the YAML section power_controller), evaluated
every Ts seconds from the moment AUTO is switched on:

    x_k  = ln(P_k / P_set)                       log power error (control channel)
    w_k  = (x_k - x_{k-1}) / Ts                  measured inverse period
    w*_k = clip(-x_k / T_r, -1/T_dn, +1/T_up)    demanded inverse period
    v_k  = clip(K (w*_k - w_k), -v_max, +v_max)  regulating rod speed [cm/s], + = withdraw

and the rod is driven at v_k for one sample period (piecewise-linear motion, so the kinetics
solver integrates it exactly). The rod position integrates v, so at a steady state
(w = 0, v = 0) the power equals the setpoint exactly -- no offset, with or without feedback.
The first sample after switching on only records x (bumpless transfer: v = 0).

The controller reads its own channel (the fission power), never the safety channels: the
MITR tech specs require the safety system to be separate from any automatic controller.
On a scram it drops to MANUAL and stops the rod (the regulating rod has no scram).

Commands are scheduled at simulation times (scenario files) or at the current time (live):
    ctl.schedule(t, "auto", (True, setpoint_w or None))   # None: hold the power at that moment
    ctl.schedule(t, "auto", (False, None))
    ctl.schedule(t, "setpoint", watts)
    ctl.schedule(t, "reg", target_cm)                     # manual rod target (leaves AUTO)
TransientSolver calls update() at the start (t = 0) and at the end of every step, and uses
breakpoints() so that steps end exactly on command and sample times.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ControllerParams:
    sample_period_s: float = 1.0
    approach_time_s: float = 30.0
    rod_gain_cm: float = 40.0
    min_period_up_s: float = 30.0
    min_period_down_s: float = 30.0
    v_max: float = 0.5                # cm/s, the regulating rod's drive speed


def controller_params(cfg):
    pc = cfg.get("power_controller") or {}
    rr = cfg["control"]["regulating_rod"]
    keys = ("sample_period_s", "approach_time_s", "rod_gain_cm", "min_period_up_s",
            "min_period_down_s")
    return ControllerParams(**{k: float(pc[k]) for k in keys if k in pc},
                            v_max=float(rr["drive_speed"]))


def regulating_rod_program(cfg, h0=None):
    """BladeProgram for the regulating rod (None if the configuration has none)."""
    from kinetics3d import BladeProgram
    rr = cfg["control"].get("regulating_rod")
    if not rr:
        return None
    h = float(rr["withdrawn"] if h0 is None else h0)
    # no scram capability: the scram speed is never used (scram() is never called on it)
    return BladeProgram(h, float(rr["drive_speed"]), float(rr["drive_speed"]), float(rr["travel"]))


def controller_from_config(cfg, reg_program):
    if reg_program is None or "power_controller" not in cfg:
        return None
    return PowerController(controller_params(cfg), reg_program)


class PowerController:
    def __init__(self, params, reg_program):
        self.par, self.reg = params, reg_program
        self.events = []                 # (time, seq, kind, value), applied in time order
        self._seq = 0
        self.reset()

    def reset(self):
        self.events.clear()
        self.auto = False
        self.setpoint_w = None
        self.t_on = None                 # sample grid: t_on + k Ts
        self.x_prev = None
        self.at_limit = False
        self._last_sample = None
        self.log = []                    # (t, P, x, w, w*, v, rod) per sample in AUTO
        self.messages = []               # (t, text): mode changes

    # ------------------------------------------------------------------ commands
    def schedule(self, t, kind, value):
        if kind not in ("auto", "setpoint", "reg"):
            raise ValueError(kind)
        self._seq += 1
        self.events.append((float(t), self._seq, kind, value))
        self.events.sort(key=lambda e: (e[0], e[1]))

    # ------------------------------------------------------------------ timing
    def _next_sample(self, t):
        Ts = self.par.sample_period_s
        k = np.floor((t - self.t_on) / Ts + 1e-9) + 1.0
        return self.t_on + k * Ts

    def _on_grid(self, t):
        Ts = self.par.sample_period_s
        k = np.round((t - self.t_on) / Ts)
        return abs(t - (self.t_on + k * Ts)) < 1e-9 * max(1.0, Ts)

    def breakpoints(self, t0, t1):
        out = [te for te, _, _, _ in self.events if t0 < te < t1]
        if self.auto:
            ts = self._next_sample(t0 - 1e-9)
            if t0 < ts < t1:
                out.append(ts)
        return out

    # ------------------------------------------------------------------ update
    def update(self, t, power_w, scrammed=False):
        """Apply commands due at t, then take a sample if t is on the AUTO sample grid."""
        sample_now = False
        while self.events and self.events[0][0] <= t + 1e-9:
            _, _, kind, value = self.events.pop(0)
            if kind == "auto":
                on, sp = value
                if on and not scrammed:
                    if not self.auto:
                        self.auto, self.t_on, self.x_prev = True, t, None
                        self._last_sample = None
                        self.messages.append((t, "automatic control on"))
                        sample_now = True
                        self.setpoint_w = float(power_w)      # default: hold this power
                    if sp is not None:
                        self.setpoint_w = float(sp)
                elif not on and self.auto:
                    self._to_manual(t, "automatic control off")
            elif kind == "setpoint":
                self.setpoint_w = float(value)
            elif kind == "reg":
                if self.auto:
                    self._to_manual(t, "manual rod command: automatic control off")
                self.reg.command(t, float(value))
        if scrammed and self.auto:
            self._to_manual(t, "scram: automatic control off")
        if self.auto and (sample_now or self._on_grid(t)) and t != self._last_sample:
            self._sample(t, power_w)        # (at most once per instant: live commands may call
                                            #  update again at a time already sampled)

    def _to_manual(self, t, why):
        self.auto = False
        h = self.reg.position(t)
        self.reg.command(t, h)              # stop where it is
        self.messages.append((t, why))

    def _sample(self, t, power_w):
        p = self.par
        self._last_sample = t
        x = float(np.log(max(power_w, 1e-300) / self.setpoint_w))
        if self.x_prev is None:
            w, v = 0.0, 0.0
            w_star = float(np.clip(-x / p.approach_time_s, -1 / p.min_period_down_s,
                                   1 / p.min_period_up_s))
        else:
            w = (x - self.x_prev) / p.sample_period_s
            w_star = float(np.clip(-x / p.approach_time_s, -1 / p.min_period_down_s,
                                   1 / p.min_period_up_s))
            v = float(np.clip(p.rod_gain_cm * (w_star - w), -p.v_max, p.v_max))
        self.x_prev = x
        h = self.reg.position(t)
        self.at_limit = (v > 0 and h + v * p.sample_period_s >= self.reg.travel) or \
                        (v < 0 and h + v * p.sample_period_s <= 0.0)
        if v == 0.0:
            self.reg.command(t, h)                     # hold
        else:
            # drive toward the end stop at |v|; the next sample re-commands. (Commanding the
            # point h + v Ts instead would put a round-off "arrival" kink just off the grid.)
            self.reg.command(t, self.reg.travel if v > 0 else 0.0, speed=abs(v))
        self.reg.compact(t)
        self.log.append((t, power_w, x, w, w_star, v, h))
