"""
verify_live.py -- headless checks of the live (background-thread) simulation (4 cm mesh).

[1] Pacing: at 5x, simulation time advances 5x as fast as the wall clock; steady at power.
[2] Same physics as offline: a live run of a scripted withdrawal matches TransientSolver.
[3] Operator commands: blades move at drive speed; inlet change; scram inserts the blades,
    power falls, and later blade commands are ignored.
[4] Time compression limits: an unreachable speed reports "lagging" with the true rate, and
    lowering the speed afterwards takes effect at once (no racing to catch up).
[5] Reset returns to the initial state.

    python3 verify_live.py        (~5 min)
"""
import time

import numpy as np

from core_builder import build, load_config
from kinetics3d import BladeProgram, PlantProgram, TransientSolver, kinetics_from_config
from live_sim import LiveSimulation
from thermal import ChannelThermal


def wait_until(cond, timeout, poll=0.05):
    t0 = time.monotonic()
    while not cond():
        if time.monotonic() - t0 > timeout:
            raise TimeoutError
        time.sleep(poll)


def main():
    live = None
    try:
        live = _main()
    finally:
        if live is not None:
            live.shutdown()


def _main():
    cfg = load_config("mitr2_like.yaml")
    cfg["mesh"]["cell_size"] = 4.0
    model = build(cfg)
    P0, h0 = 1e6, 25.9
    live = LiveSimulation(cfg, model, power_w=P0, start_height=h0)
    t_init = time.monotonic()
    live.start(running=False)
    wait_until(lambda: live.initialized, 300)
    print(f"initialized in {time.monotonic() - t_init:.0f} s; status '{live.status}'; trips from "
          f"the instrument model: {live.inst.short_period:g} s period, {live.inst.high_power / 1e6:g} MW")

    # ---- [2] scripted withdrawal, run as fast as possible, compared with offline
    live.solver.prog.command(5.0, h0 + 0.1)           # injected before running (test only)
    live.set_speed(1e4)
    live.set_running(True)
    wait_until(lambda: live.snapshot().t >= 60.0, 600)
    live.set_running(False)
    wait_until(lambda: live.status == "paused", 30)
    tr = live.trends()
    prog = BladeProgram(h0, live.drive_speed, live.scram_speed, live.travel)
    prog.command(5.0, h0 + 0.1)
    off = TransientSolver(model, prog, kin=kinetics_from_config(cfg), method="pcqs", dt_max=5.0,
                          dh_max=1.0, thermal=ChannelThermal(model, cfg), power_w=P0)
    off.run(60.0)
    P_live = np.interp(60.0, tr["t"], tr["power_w"])
    P_off = off.hist.power[-1] * P0
    print(f"[2] scripted +0.1 cm at 5 s, power at 60 s: live {P_live / 1e6:.5f} MW "
          f"({len(tr['t']) - 1} steps), offline {P_off / 1e6:.5f} MW ({len(off.hist.t) - 1} steps), "
          f"rel. diff {abs(P_live / P_off - 1):.1e}")
    assert abs(P_live / P_off - 1) < 2e-3

    # ---- [1] pacing at 5x
    live.set_speed(5.0)
    snap0 = live.snapshot()
    w0 = time.monotonic()
    live.set_running(True)
    time.sleep(12.0)
    snap1 = live.snapshot()
    rate = (snap1.t - snap0.t) / (time.monotonic() - w0)
    print(f"[1] requested 5x: simulation advanced {snap1.t - snap0.t:.1f} s in 12 s of wall time "
          f"-> {rate:.2f}x (achieved_speed reports {live.achieved_speed:.2f}x)")
    assert 4.0 < rate < 5.6

    # ---- [3] operator commands
    live.set_speed(20.0)
    s = live.snapshot()
    cmd = s.withdrawn + 0.2
    live.command_blades(cmd)
    # commands apply at the next step boundary; follow the blades until they arrive
    seen = [s]
    def arrived():
        x = live.snapshot()
        if x.seq != seen[-1].seq:
            seen.append(x)
        return np.isclose(x.target, cmd) and abs(x.withdrawn - cmd) < 1e-9
    wait_until(arrived, 120, poll=0.01)
    v = max(abs(b.withdrawn - a.withdrawn) / (b.t - a.t) for a, b in zip(seen[:-1], seen[1:]))
    s = live.snapshot()
    print(f"[3] blade command +0.2 cm: blades reached {s.withdrawn:.3f} cm; fastest observed blade "
          f"speed {v:.3f} cm/s (drive speed {live.drive_speed} cm/s)")
    assert v <= live.drive_speed * (1 + 1e-9)
    T_before = s.inlet_c
    live.set_inlet_temperature(273.15 + T_before - 3.0)
    wait_until(lambda: abs(live.snapshot().inlet_c - (T_before - 3.0)) < 1e-9, 60, poll=0.01)
    t_app = live.snapshot().t
    wait_until(lambda: live.snapshot().t > t_app + 2.0, 60)
    s2 = live.snapshot()
    print(f"    inlet {T_before:.1f} C -> {s2.inlet_c:.1f} C; core outlet {s.outlet_c:.2f} -> {s2.outlet_c:.2f} C")
    assert abs(s2.inlet_c - (T_before - 3.0)) < 1e-9 and s2.outlet_c < s.outlet_c
    P_pre = s2.power_w
    live.scram()
    wait_until(lambda: live.snapshot().scrammed and live.snapshot().withdrawn == 0.0, 60)
    t_sc = live.snapshot().t
    wait_until(lambda: live.snapshot().t > t_sc + 3.0, 60)
    s3 = live.snapshot()
    live.command_blades(40.0)
    wait_until(lambda: live.snapshot().t > s3.t + 3.0, 60)   # long enough to have applied it
    s4 = live.snapshot()
    print(f"    scram: blades {s3.withdrawn:.1f} cm, power {P_pre / 1e6:.3f} MW -> {s3.power_w / 1e6:.4f} MW "
          f"in {s3.t - s2.t:.1f} s ('{s3.trip_reason}'); blade command after scram -> target {s4.target:.2f} cm")
    assert s3.power_w < 0.1 * P_pre and s4.withdrawn == 0.0 and s4.target != 40.0

    # ---- [4] unreachable speed, then back to 1x
    live.set_speed(1e4)
    time.sleep(4.0)
    st, ach = live.status, live.achieved_speed
    print(f"[4] requested 10000x: status '{st}', achieved {ach:.0f}x")
    assert st == "running (lagging)" and ach < 1e4
    live.set_speed(1.0)
    time.sleep(0.5)
    a = live.snapshot()
    w0 = time.monotonic()
    time.sleep(6.0)
    b = live.snapshot()
    rate = (b.t - a.t) / (time.monotonic() - w0)
    print(f"    back to 1x: {rate:.2f}x over 6 s (no catch-up burst)")
    assert rate < 1.5

    # ---- [5] reset
    live.reset()
    wait_until(lambda: live.status == "initializing", 30)
    wait_until(lambda: live.initialized, 300)
    s = live.snapshot()
    print(f"[5] reset: t = {s.t:.1f} s, blades {s.withdrawn:.2f} cm, power {s.power_w / 1e6:.4f} MW, "
          f"scrammed {s.scrammed}")
    time.sleep(1.0)
    assert live.status == "paused" and live.snapshot().t == 0.0, "reset must leave it paused"
    assert s.t == 0.0 and s.withdrawn == h0 and abs(s.power_w / P0 - 1) < 1e-9 and not s.scrammed
    assert live.error is None
    print("\nLive simulation checks passed.")
    return live


if __name__ == "__main__":
    main()
