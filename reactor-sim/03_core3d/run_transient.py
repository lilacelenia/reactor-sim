"""
run_transient.py -- run the time-dependent scenario described in a core's YAML file.

    python3 run_transient.py mitr2_like.yaml                 # production mesh (slow: ~30 min)
    python3 run_transient.py mitr2_like.yaml --coarse        # 4 cm mesh, quick look (~3 min)
    python3 run_transient.py mitr2_like.yaml --no-feedback   # same scenario, feedback off

Writes <stem>_transient.png (power, reactivity, period, element power maps) and
<stem>_transient.npz (full time history, including element powers at every step).
"""
import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from core_builder import build, load_config
from kinetics3d import BladeProgram, PlantProgram, TransientSolver, kinetics_from_config
from thermal import ChannelThermal


def make_program(cfg):
    bank, tr = cfg["control"]["shim_bank"], cfg["transient"]
    h0 = float(tr["start_height"])
    prog = BladeProgram(h0, bank["drive_speed"], bank["scram_speed"], bank["travel"])
    plant = PlantProgram()
    for c in tr.get("commands", []):
        if c.get("scram"):
            prog.scram(c["time"])
        elif "move" in c:
            prog.command(c["time"], h0 + float(c["move"]))
        elif "target" in c:
            prog.command(c["time"], float(c["target"]))
        for kind in ("inlet_temperature", "flow"):
            if kind in c:
                plant.set(c["time"], kind, c[kind])
    return prog, plant


def inhour_period(rho, Lam, kin):
    """Stable period T solving rho = w*Lambda + sum beta_i w / (w + lam_i), w = 1/T."""
    from scipy.optimize import brentq
    f = lambda w: w * Lam + np.sum(kin.beta_i * w / (w + kin.lam_i)) - rho
    if rho > 0:
        return 1.0 / brentq(f, 1e-12, 1e4)
    if rho < 0:
        return 1.0 / brentq(f, -kin.lam_i.min() * (1 - 1e-12), -1e-12)
    return np.inf


def plot(cfg, model, hist, kin, P0, out_png, feedback):
    h = hist.arrays()
    t, P = h["t"], h["power"] * P0
    lat = model.lattice
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f"{cfg['name']} -- {cfg['transient']['method'].upper()} transient, "
                 f"{'with' if feedback else 'without'} thermal feedback", fontsize=13)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 0.85, 1.0])

    # power + blade position
    ax = fig.add_subplot(gs[0, :2])
    if feedback:            # linear MW scale shows the feedback level-off best
        ax.plot(t, P / 1e6, "C3", lw=2, label="fission power")
        ax.set(xlabel="time [s]", ylabel="power [MW]")
    else:
        ax.semilogy(t, P, "C3", lw=2, label="fission power")
        ax.set(xlabel="time [s]", ylabel="power [W]")
    ax.grid(alpha=0.3, which="both")
    ax2 = ax.twinx()
    ax2.plot(t, h["blade"], "C0", lw=1.2, label="shim bank height")
    ax2.set_ylabel("shim bank withdrawal [cm]", color="C0")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    for te, txt in hist.events:
        ax.axvline(te, color="k", ls=":", lw=1)

    # reactivity
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(t, h["rho"] / kin.beta, "C2", lw=2, label="net")
    if feedback:
        dfb = (h["rho_feedback"] - h["rho_feedback"][0]) / kin.beta
        ax.plot(t, dfb, "C3--", label="temperature feedback")
        ax.plot(t, h["rho"] / kin.beta - dfb, "C0:", label="blades")
        ax.legend(fontsize=8)
    ax.set_yscale("symlog", linthresh=0.5)          # shows cents and the scram's -$13 together
    ax.set(xlabel="time [s]", ylabel="reactivity [$]  (symlog)", title="Reactivity (adjoint-weighted)")
    ax.axhline(0, color="k", lw=0.6)
    ax.grid(alpha=0.3)

    # inverse period
    ax = fig.add_subplot(gs[1, 0] if feedback else gs[1, :2])
    tm = 0.5 * (t[1:] + t[:-1])
    inv_T = np.diff(np.log(h["power"])) / np.diff(t)
    ax.step(tm, inv_T, where="mid", color="C1")
    ax.axhline(0, color="k", lw=0.6)
    ax.set(xlabel="time [s]", ylabel="1 / period [1/s]",
           title="Inverse period, d ln P / dt (spikes clipped)")
    ax.set_ylim(-0.08, 0.08)
    ax.grid(alpha=0.3)

    if feedback:
        ax = fig.add_subplot(gs[1, 1])
        ax.plot(t, h["fuel_max"] - 273.15, "C3", label="hottest plate")
        ax.plot(t, h["fuel_avg"] - 273.15, "C1", label="average plate")
        ax.plot(t, h["coolant_outlet_max"] - 273.15, "C4", label="hottest channel outlet")
        ax.plot(t, h["coolant_outlet"] - 273.15, "C0", label="core outlet (mixed)")
        ax.plot(t, h["inlet"] - 273.15, "k:", label="inlet")
        ax.set(xlabel="time [s]", ylabel="temperature [C]", title="Channel temperatures")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # power distribution peaking vs time
    ax = fig.add_subplot(gs[1, 2])
    E = h["element_power"]
    fueled = E[0] > 0
    shape = E[:, fueled] / E[:, fueled].mean(axis=1, keepdims=True)
    ax.plot(t, shape.max(axis=1), label="peak element")
    ax.plot(t, shape.min(axis=1), label="lowest element")
    ax.set(xlabel="time [s]", ylabel="element power / core average", title="Power shape over time")
    ax.legend()
    ax.grid(alpha=0.3)

    # element maps at three moments
    moments = [("start", 0.0), ("peak power", t[int(np.argmax(P))]), ("end", t[-1])]
    polys = [e["verts"] for e in lat.elements]
    for col, (name, tm_) in enumerate(moments):
        i = int(np.argmin(np.abs(t - tm_)))
        rel = np.where(fueled, E[i] / E[i, fueled].mean(), np.nan)
        ax = fig.add_subplot(gs[2, col])
        pc = PolyCollection(polys, array=rel, cmap="inferno", edgecolors="k", linewidths=0.6)
        pc.set_clim(0.5, 1.6)
        pc.cmap.set_bad("#b0b8c0")
        ax.add_collection(pc)
        for e, v in zip(lat.elements, rel):
            if np.isfinite(v):
                ax.text(*e["center"], f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v < 1.0 else "black")
        ax.set(xlim=(-22, 22), ylim=(-22, 22), aspect="equal",
               title=f"{name}: t = {t[i]:.0f} s, P = {P[i]:.3g} W, blades {h['blade'][i]:.1f} cm")
        ax.title.set_fontsize(9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(pc, ax=ax, shrink=0.8, label="element / core average")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="mitr2_like.yaml")
    ap.add_argument("--coarse", action="store_true", help="4 cm mesh for a quick look")
    ap.add_argument("--method", choices=["pcqs", "adiabatic", "direct"])
    ap.add_argument("--no-feedback", action="store_true", help="switch thermal feedback off")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.coarse:
        cfg["mesh"]["cell_size"] = 4.0
    tr = cfg["transient"]
    if args.method:
        tr["method"] = args.method
    feedback = bool(tr.get("feedback", True)) and "thermal_hydraulics" in cfg and not args.no_feedback
    kin = kinetics_from_config(cfg)
    model = build(cfg)
    prog, plant = make_program(cfg)
    P0 = float(tr["initial_power_w"])
    trip = tr.get("trip_power_w")
    thermal = ChannelThermal(model, cfg) if feedback else None
    print(f"{cfg['name']}: {tr['method']} transient, mesh {model.problem.mesh.shape}, "
          f"feedback {'on' if feedback else 'off'}")
    t0 = time.time()
    solver = TransientSolver(model, prog, kin=kin, method=tr["method"], dt_max=tr["dt_max"],
                             dh_max=tr["dh_max"], trip_power=None if trip is None else trip / P0,
                             thermal=thermal, power_w=P0, plant=plant, verbose=True)
    hist = solver.run(float(tr["end_time"]))
    h = hist.arrays()

    if not feedback:
        # stable period during the hold, compared with the inhour equation
        hold = [c["time"] for c in tr["commands"]]
        if len(hold) >= 2:
            a, b = hold[0] + 0.6 * (hold[1] - hold[0]), hold[1]
            m = (h["t"] >= a) & (h["t"] <= b)
            if m.sum() >= 2:
                T_sim = (h["t"][m][-1] - h["t"][m][0]) / np.log(h["power"][m][-1] / h["power"][m][0])
                rho = h["rho"][m].mean()
                T_inh = inhour_period(rho, h["gen_time"][m].mean(), kin)
                print(f"\nDuring the hold: reactivity {rho * 1e5:.1f} pcm (${rho / kin.beta:.3f}), "
                      f"simulated period {T_sim:.1f} s, inhour-equation period {T_inh:.1f} s")
    else:
        i = int(np.argmax(h["power"]))
        print(f"\nAt peak power ({h['power'][i] * P0 / 1e6:.3f} MW, t = {h['t'][i]:.0f} s): hottest plate "
              f"{h['fuel_max'][i] - 273.15:.1f} C, core outlet {h['coolant_outlet'][i] - 273.15:.1f} C, "
              f"feedback {(h['rho_feedback'][i] - h['rho_feedback'][0]) * 1e5:+.1f} pcm since start")
    print(f"Peak power {h['power'].max() * P0:.4g} W; final {h['power'][-1] * P0:.4g} W; "
          f"{len(h['t']) - 1} steps in {time.time() - t0:.0f} s")
    for te, txt in hist.events:
        print(f"Event at {te:.2f} s: {txt}")

    stem = Path(args.config).stem + ("_coarse" if args.coarse else "") + ("" if feedback else "_nofb")
    np.savez_compressed(f"{stem}_transient.npz", **h, initial_power_w=P0, labels=np.array(model.labels))
    plot(cfg, model, hist, kin, P0, f"{stem}_transient.png", feedback)
    print(f"Saved {stem}_transient.png and {stem}_transient.npz")


if __name__ == "__main__":
    main()
