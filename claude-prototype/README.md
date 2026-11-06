# Reactor simulation project

A step-by-step build of a nuclear reactor simulator in Python, from a point-kinetics model
up to a 3D, time-dependent, thermally coupled core based on an MITR-II-like configuration.

> **Illustrative, not licensing-grade.** The MITR-II geometry follows published data
> (27 rhomboid positions in rings of 3/9/15, 57 cm fueled length, six shim blades with
> 53 cm travel, D2O reflector, 6 MW, 1800 gpm). Cross sections, feedback coefficients and
> thermal parameters are textbook-style **estimates**, marked "estimate" in
> `03_core3d/mitr2_like.yaml`. Results show realistic behavior, not MITR's actual numbers.

## Install

```
python3 -m pip install -r requirements.txt
```

`pyside6`, `pyvista` and `pyvistaqt` are only needed for the two GUIs.

## Contents

### 01_point_kinetics_and_1d — first steps
| File | What it does |
|---|---|
| `point_kinetics.py` | 6-group point kinetics with a lumped Doppler feedback model; $0.10 step insertion |
| `diffusion_1d.py` | 1D two-group diffusion k-eigenvalue solver, checked against the analytic slab |

### 02_control_panel_gui — 2D desktop simulator
| File | What it does |
|---|---|
| `reactor_model.py` | Point kinetics + lumped fuel/coolant temperatures, rods, trips (no GUI code) |
| `reactor_gui.py` | PySide6 + pyqtgraph control panel: rod slider, coolant controls, SCRAM, live plots |

Run with `python3 reactor_gui.py` from inside this folder.

### 03_core3d — 3D core simulator (MITR-II-like)
Run everything from inside `03_core3d/`.

| File | What it does |
|---|---|
| `mitr2_like.yaml` | **The reactor description**: mesh, geometry, materials, blades, kinetics data, thermal-hydraulics, feedback and the transient scenario. Edit this rather than the code. |
| `diffusion3d.py` | Generic 3D two-group diffusion solver: k-eigenvalue and fixed-source |
| `core_builder.py` | Turns the YAML into a mesh problem: hexagonal rhombic lattice, blades, reflector, cheap blade re-mixing |
| `run_core.py` | Static analysis: blade worth curve, critical height, element power map |
| `viewer_core.py` | Geometry and background solver for the 3D viewer (no graphics) |
| `core_viewer.py` | PyVista 3D viewer. Static mode: blade slider with k-eigenvalue re-solves. `--live` mode: real-time transient with time compression (see below) |
| `live_sim.py` | Runs the quasi-static solver with feedback in a background thread, paced to wall time x speed, taking operator commands (no GUI code) |
| `instruments.py` | Nuclear instrumentation on the MITR pattern: source-range fission chambers (dead time, counting statistics), period and power channels, scrams, 1/M counting and ECP extrapolation; Pu-Be source placement |
| `kinetics3d.py` | Time-dependent 3D kinetics: direct implicit, predictor-corrector quasi-static (PCQS), adiabatic |
| `thermal.py` | Per-channel thermal-hydraulics (24 channels x axial nodes) and temperature feedback |
| `run_transient.py` | Runs the scenario in the YAML and plots power, reactivity, temperatures and power maps |
| `verify_solver.py` | Static solver checks: analytic bare cube, 2nd-order convergence, symmetry, albedo BCs |
| `verify_kinetics.py` | Kinetics checks: exact separable solution, extrapolated reference on the MITR core |
| `verify_feedback.py` | Feedback checks: sensitivities, energy conservation, equilibrium vs static search |
| `verify_live.py` | Live-mode checks: pacing, commands, scram lockout, backlog cap, reset, match with offline runs |
| `gui_smoke_test.py` | Runs all of `core_viewer.py`'s logic with stand-in Qt/PyVista classes (for machines without a display or without those packages) |
| `verify_startup.py` | Startup checks: exact source-driven kinetics, source steady state, instruments and trips, scripted approach to criticality |
| `results/` | Figures and data from the runs below |

#### Typical order
```
python3 verify_solver.py                          # ~1.5 min
python3 run_core.py mitr2_like.yaml               # ~1.5 min -> critical height, power map
python3 core_viewer.py mitr2_like.yaml            # interactive 3D viewer (static)
python3 core_viewer.py mitr2_like.yaml --live     # live transient with feedback
python3 core_viewer.py mitr2_like.yaml --startup  # start up from the Pu-Be source
python3 run_transient.py mitr2_like.yaml --coarse # ~3 min transient with feedback (4 cm mesh)
python3 run_transient.py mitr2_like.yaml          # ~30 min on the 2 cm mesh
python3 verify_kinetics.py cube steady            # ~1 min
python3 verify_feedback.py                        # ~5 min
python3 verify_live.py                            # ~5 min
python3 verify_startup.py                         # ~25 min (the approach to criticality is ~15)
```
Full reference comparisons take longer and can run in parallel:
`python3 verify_kinetics.py run <name>` / `compare`, and likewise for `verify_feedback.py`.

#### Live mode (`core_viewer.py --live`)
The core runs in real time (or faster) while you operate it:
- **Run / Pause, Reset, speed** (0.5x to 1000x time compression). If the computer cannot keep
  up, the status says *lagging* and shows the speed actually achieved; lowering the speed
  takes effect at once.
- **Shim bank slider** sets the *commanded* position (0.01 cm resolution, plus +-0.05 cm
  buttons); blades drive there at 0.2 cm/s. **SCRAM** drops them in under a second.
- **Coolant inlet temperature and flow** can be changed while running.
- Readouts: power, period, reactivity, feedback, hottest plate, core outlet, peak element;
  trend plots of power, blade position and temperatures; 3D colors by element power or
  hottest plate temperature, or flux / power-density slices.
- Automatic high-power trip at 120 % of rated power (7.2 MW); press Reset afterwards.
- Options: `--mesh 4` (default; with feedback it reached ~3-5x real time on the test machine),
  `--mesh 2` (finer, well below real time), `--power 100` (start at 100 W, i.e. a startup
  from low power), `--no-feedback`.

Commands take effect at the next solver step (a fraction of a second of wall time).

#### Startup from the source (`core_viewer.py --startup`)
Starts shut down: blades fully in, Pu-Be source in, k = 0.95 on the 4 cm mesh, about
0.4 mW and a few counts per second on the source-range channels. Approach criticality the
way operators do:
1. Press **Count** (30 or 100 s) for a reference count with the blades in.
2. Withdraw in steps (big ones far from critical), let the count rate settle (minutes near
   critical), count again. Each count adds a point to the **1/M plot** and updates the
   **estimated critical position (ECP)** from a fit through the last three points.
3. Near critical use the 0.05 cm buttons. MITR practice is 0.1 inch (0.254 cm) steps, never
   more than 0.5 inch above the ECP.
4. Go a step past critical: the count rate keeps rising on a stable period. The fission
   chambers saturate around 5 kW; the ion-chamber channels take over above ~500 W.

Large steps near critical trip the 10 s period scram, just as on the real reactor. In the
scripted test, a 0.8 cm step did exactly that, and so did 0.5 cm in one move at critical.

Instrumentation and source in `mitr2_like.yaml`. Published (MITR): Pu-Be source; channels 1-2
source-range fission chambers switched to ion chambers ~1 kW and ineffective above ~5 kW;
channel 3 ion chamber on scale ~500 W; channels 1-3 period scram at 10 s; channels 4-6 high
flux scram at 6.5 MW (80 kW without primary flow); 0.1 inch startup increments. Estimates:
source strength (1e7 n/s) and position, detector positions (the real ports are 4-5 ft away,
outside the model, so detectors sample the domain edge), fission-chamber sensitivity (set
for a few cps at shutdown), dead time, meter time constants (3 s period, 2 s count rate),
trip logic (any one channel trips). MITR has been upgrading to four wide-range fission
chamber channels with 2-out-of-4 logic; that is not modeled.

## Verification status

| Check | Result |
|---|---|
| Bare cube k vs analytic | 2nd-order convergence, 8e-6 error at 64^3 |
| Reflected slab vs independent 1D eigen solve | agrees to 3e-10 |
| Subcritical multiplication | M = 1/(1-k) to 1e-4 |
| Kinetics vs exact separable solution | direct: 1st order; PCQS with 0.5 s steps: 3e-5 |
| PCQS vs extrapolated reference (MITR core, scram) | 4e-6 before, 5e-4 during, 6e-5 after the blade drop |
| Inhour equation vs simulated period | 18.2 s vs 18.2 s |
| Feedback sensitivities vs exact reactivity | 6e-5 |
| Thermal energy conservation | 4e-14 |
| Coupled transient vs extrapolated reference | PCQS (1 s steps): 7e-4 in power, 0.08 K in temperature |
| Transient end state vs static feedback equilibrium | 4.0033 vs 4.0038 MW |
| Live run vs offline run of the same scenario | 1.8e-5 |
| Source-driven state vs exact separable solution | 4e-13 |
| Source kinetics, -$8.1 to +$0.6 (power x3949) vs exact | PCQS 1 s steps: 6e-5; direct: 1st order |
| MITR source state vs independent fixed-source iteration | 3e-9 |
| Count-rate meter noise vs counting theory | 0.914 vs 0.911 cps |
| Period meters vs true stable period | 242.2 s vs 242.2 s |
| Trips | period (continuous withdrawal), 6.51 MW, 98 kW without flow |
| 1/M approach ECP vs static critical height | 20.47 vs 20.55 cm (1/3 of a 0.1 inch step) |
| Live pacing at 5x requested | 4.8x achieved; 1.0x after dropping back from an unreachable speed (no catch-up burst) |

The GUIs were written without a display, so their windows have never been opened. Their
physics and solver logic are tested headlessly, and `core_viewer.py` (both modes) was also
exercised end to end with stand-in Qt/PyVista classes: every button, slider, field, pick,
scram and reset path runs against the real engine, with the matplotlib plots rendered.

## Known limitations
- Diffusion theory is crude for the thin, near-black shim blades; blade constants are
  "equivalent" values, and rod cusping is only partly corrected (critical height +-1 cm).
- Elements are homogenized, so plate-edge hot spots are not resolved.
- No boiling, decay heat, natural circulation or D2O photoneutrons. Flow and inlet changes
  are instantaneous steps.
- The adiabatic method is inaccurate after a scram (15 %); use `pcqs`.

## Next steps
- Decay heat, pump coastdown and natural circulation for loss-of-flow events.
- The regulating rod (fine control) and an automatic power controller.
- MITR's wide-range Campbelling channels with 2-out-of-4 trip logic.
