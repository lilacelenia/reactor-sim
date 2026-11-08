# Reactor simulation project

A step-by-step build of a nuclear reactor simulator in Python, from a point-kinetics model
up to a 3D, time-dependent, thermally coupled core based on an MITR-II-like configuration.

> **Illustrative, not licensing-grade.** The MITR-II geometry follows published data
> (27 rhomboid positions in rings of 3/9/15, 57 cm fueled length, six shim blades with
> 53 cm travel, a cadmium regulating rod at a core corner, D2O reflector, 6 MW, 2000 gpm
> normal flow from two pumps, four natural convection valves). Decay heat uses the
> published ANS-5.1-1979 standard; the shim blades' absorber and the fuel's nu-Sigma_f are
> fitted to the published HEU bank worth (9927 pcm) and critical bank height (7.36 in); the
> regulating rod's absorber is calibrated to its published 124 pcm worth. Cross sections,
> feedback coefficients, thermal and pump parameters are textbook-style **estimates**, marked
> "estimate" in `03_core3d/mitr2_like.yaml`. Results show realistic behavior, not MITR's
> actual numbers.

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
| `core_builder.py` | Turns the YAML into a mesh problem: hexagonal rhombic lattice, blades, regulating rod, reflector, cheap absorber re-mixing (`problem_at(bank, reg_rod)`) |
| `run_core.py` | Static analysis: blade worth curve, critical height, element power map |
| `viewer_core.py` | Geometry and background solver for the 3D viewer (no graphics) |
| `core_viewer.py` | PyVista 3D viewer. Static mode: blade slider with k-eigenvalue re-solves. `--live` mode: real-time transient with time compression (see below) |
| `live_sim.py` | Runs the quasi-static solver with feedback in a background thread, paced to wall time x speed, taking operator commands (no GUI code) |
| `controller.py` | Automatic power controller on the regulating rod: sampled-data period-demand law, AUTO/manual, setpoint, drops to manual on a scram (no GUI code) |
| `instruments.py` | Nuclear instrumentation on the MITR pattern: source-range fission chambers (dead time, counting statistics), period and power channels, scrams, 1/M counting and ECP extrapolation; Pu-Be source placement |
| `kinetics3d.py` | Time-dependent 3D kinetics: direct implicit, predictor-corrector quasi-static (PCQS), adiabatic; shim bank and regulating rod programs; controller sampled on exact step boundaries |
| `thermal.py` | Per-channel thermal-hydraulics (24 channels x axial nodes), temperature feedback, decay heat (23 ANS-5.1 groups per node), primary loop (pump coastdown, natural circulation through the natural convection valves) |
| `run_transient.py` | Runs the scenario in the YAML and plots power, reactivity, temperatures and power maps |
| `verify_solver.py` | Static solver checks: analytic bare cube, 2nd-order convergence, symmetry, albedo BCs, fixed source, multigrid preconditioner, Anderson acceleration |
| `verify_kinetics.py` | Kinetics checks: exact separable solution, extrapolated reference on the MITR core |
| `verify_feedback.py` | Feedback checks: sensitivities, energy conservation, equilibrium vs static search; loss of flow: decay heat vs the ANS closed form, coastdown vs exact solutions, natural circulation vs an algebraic steady state, loss-of-flow accident vs extrapolated reference |
| `verify_live.py` | Live-mode checks: pacing, commands, scram lockout, backlog cap, reset, match with offline runs |
| `gui_smoke_test.py` | Runs all of `core_viewer.py`'s logic with stand-in Qt/PyVista classes (for machines without a display or without those packages) |
| `verify_all.py` | The whole verification suite, independent scripts and reference runs as parallel processes (one per performance core); logs in `results/logs/` |
| `multigrid.py` | Smoothed-aggregation algebraic multigrid preconditioner (2x2x2 aggregates on the structured mesh, Galerkin coarse grids, Jacobi-smoothed V-cycle) for CG on fine meshes |
| `parallel.py` | Process pool for independent runs: counts Apple Silicon performance cores, one BLAS thread per process |
| `verify_startup.py` | Startup checks: exact source-driven kinetics, source steady state, instruments and trips, scripted approach to criticality |
| `verify_blades.py` | Shim blade checks: exact blade areas on every mesh, individual blade positions, fit to the published HEU bank worth and critical height, and comparison with published values that were not fitted (excess reactivity, single-blade worths, shutdown margin); `calibrate` refits blades and fuel |
| `verify_control.py` | Regulating rod and controller checks: worth vs published value and tech-spec limits, inhour, closed loop vs static searches, PCQS vs extrapolated direct reference; `calibrate` refits the rod absorber |
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
python3 verify_feedback.py lof                    # ~5 min (decay heat, coastdown, natural circulation)
python3 verify_live.py                            # ~5 min
python3 verify_startup.py                         # ~25 min (the approach to criticality is ~15)
python3 verify_blades.py                          # ~1 min (shim blades vs published worths)
python3 verify_control.py                         # ~20 min (regulating rod and controller)
```
Full reference comparisons take longer. `python3 verify_kinetics.py run` (no names = all
runs) runs them in parallel processes, then `compare`; likewise `verify_feedback.py` and
`verify_control.py`. Or run everything at once:
```
python3 verify_all.py             # whole suite incl. reference runs; --quick skips those
python3 verify_all.py --jobs 4    # limit processes (default: performance cores; or REACTOR_JOBS=4)
```

#### Performance notes (Apple Silicon)
- The solvers are single-threaded sparse code (scipy CSR products), so speed comes from
  fewer iterations and from running independent cases in parallel, not from BLAS or the GPU.
  Apple GPUs have no 64-bit floats, which the 1e-9 eigenvalue tolerances need.
- Use a native arm64 Python (`python3 -c "import platform; print(platform.machine())"` should
  print `arm64`; an x86_64 Python under Rosetta is much slower).
- Parallel runs use the performance cores only (`sysctl hw.perflevel0.physicalcpu`): an
  efficiency core would take several times longer on its share. Each process is limited to
  one BLAS thread (Accelerate/OpenBLAS) so processes do not compete.
- The group solves use warm-started Jacobi-preconditioned CG with a tolerance that tightens as
  the outer iteration converges, and a lean CG loop instead of scipy's (whose operator
  wrapping cost more than the matrix products). Direct LU is used only for tiny test meshes:
  for these 3D meshes its fill makes it slower (4 cm) or runs out of memory (2 cm).
- Measured on the 2-core test machine, same results to 1e-11 in k: 2 cm eigenvalue solve
  24 -> 7 s (warm-started 19 -> 7 s); 4 cm solve 6.3 -> 0.6 s; transient start-up
  (steady state + adjoint) 25 -> 1.2 s; `verify_blades.py` 11 -> 1.1 min; whole suite with
  reference runs ~4 h sequential -> 76 min with `verify_all.py` on 2 cores. With more
  performance cores the suite is bounded by its longest tasks (startup ~25 min, the finest
  loss-of-flow reference ~15 min).
- The PCQS amplitude/thermal ODE (Radau) runs at the solver's tolerance (`tol`, 1e-7 by
  default; it was a fixed 1e-9, which took 2.7x the steps without changing any verified
  result), and its right-hand side fills one array instead of concatenating pieces, with one
  water-density evaluation per call. The ODE share of a 4 cm transient with feedback fell
  from 22 s to 10.5 s (whole run 65 -> 53 s); PCQS reference runs are 1.5-1.8x faster,
  `verify_startup.py` 26 -> 19 min, and live mode keeps up at up to ~6x real time on the test
  machine (was ~3x). Every verified error is unchanged.
- Tried and dropped: integrating knot to knot of the reactivity table and carrying the ODE step
  size between macro steps (no gain); natural-order LU inside Radau (faster on isolated
  matrices, slower in the runs).
- Multigrid (`multigrid.py`, no extra dependency): CG preconditioned by one smoothed-aggregation
  V-cycle needs 12-17 iterations on every mesh instead of Jacobi's 80-230 (doubling per mesh
  halving). A V-cycle costs ~6-7 fine-grid matrix products, so the net gain depends on mesh
  size: on the 2 cm mesh the eigenvalue solve is 7.8 -> 5.4 s and a transient 1.46x faster
  (10 s with a scram: 491 -> 337 s, power equal to 6e-10); on the 4 cm mesh it breaks even, so
  it is used only from 50,000 cells (`diffusion3d.MG_LIMIT`). The 4 cm live mesh is unaffected.
  pyamg (native Apple Silicon wheels) offers the same method with more tuning options.
- Anderson acceleration (depth 5) of the outer (fission-source) iteration in the eigenvalue,
  adjoint and fixed-source solvers: ~4x fewer outer iterations (MITR 2 cm: 101 -> 23). With
  multigrid, the 2 cm eigenvalue solve went 24 s (start of this work) -> 1.8 s, the 2 cm adjoint
  5.9 -> 2.0 s, the 4 cm eigenvalue solve 0.52 -> 0.19 s. Safeguards restart the history if the
  residual grows or the source turns negative. `solve_k(..., accelerate=False)` gives the
  plain power iteration. PCQS time steps are unaffected (their shape solve is already GMRES).
- Not yet done: the time-step matrices are rebuilt every step; the shape solves (CG inside
  GMRES) are now most of a transient.

#### Live mode (`core_viewer.py --live`)
The core runs in real time (or faster) while you operate it:
- **Run / Pause, Reset, speed** (0.5x to 1000x time compression). If the computer cannot keep
  up, the status says *lagging* and shows the speed actually achieved; lowering the speed
  takes effect at once.
- **Shim bank slider** sets the *commanded* position (0.01 cm resolution, plus +-0.05 cm
  buttons); blades drive there at 0.2 cm/s. **SCRAM** drops them in under a second.
- **Regulating rod** (no scram): +-0.5 cm buttons move it by hand at 0.5 cm/s. **Automatic
  power control** puts it on the controller, which holds the power of that moment; type a
  new setpoint (kW) to change power. The rod is worth only ~124 pcm (~3 pcm/s at most), so
  this holds power and makes small changes; big changes are for the shim bank. A manual rod
  command or a scram switches AUTO off. If the rod reaches an end of travel the panel says so:
  move the shim bank to bring it back. The 3D view draws it in orange.
- **Coolant**: inlet temperature, pump speed and **primary pumps running** (2, 1, 0) can be
  changed while running. Tripping pumps starts a coastdown; at power the low-flow scram trips
  the reactor, and with no pumps the core is cooled by natural circulation through the
  natural convection valves. The *Primary loop* readout shows flow, pumps, valves and decay heat.
- Readouts: power, period, reactivity, feedback, hottest plate, core outlet, peak element;
  trend plots of power, blade position and temperatures; 3D colors by element power or
  hottest plate temperature, or flux / power-density slices.
- Automatic high-power trip at 120 % of rated power (7.2 MW); press Reset afterwards.
- Options: `--mesh 4` (default; with feedback it keeps up to ~6x real time on the test
  machine; AUTO samples every second, which costs a little),
  `--mesh 2` (finer, well below real time), `--power 100` (start at 100 W, i.e. a startup
  from low power), `--no-feedback`.

Commands take effect at the next solver step (a fraction of a second of wall time).

#### Startup from the source (`core_viewer.py --startup`)
Starts shut down: blades fully in, Pu-Be source in, k = 0.977 on the 4 cm mesh, about
2 mW and 10-20 counts per second on the source-range channels. Approach criticality the
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
flux scram at 6.5 MW (80 kW without primary flow); low primary flow scram at 1800 gpm (of
2000) with two pumps; 0.1 inch startup increments. Estimates:
source strength (1e7 n/s) and position, detector positions (the real ports are 4-5 ft away,
outside the model, so detectors sample the domain edge), fission-chamber sensitivity (set
for a few cps at shutdown), dead time, meter time constants (3 s period, 2 s count rate),
trip logic (any one channel trips; the low-flow scram is armed only above the 80 kW no-flow
limit, a simplification of the tech-spec pump modes). MITR has been upgrading to four
wide-range fission chamber channels with 2-out-of-4 logic; that is not modeled.

#### Loss of flow: decay heat, pump coastdown, natural circulation
Published and used as such: ANS-5.1-1979 U-235 decay heat (23 groups); two primary pumps;
upward core flow; four natural convection valves (ball valves that open when forced flow
stops); the ~76 cm flow guide above the core (chimney); tech-spec flow limits; and the
MULCH-II natural-convection result (1.35 kg/s at 100 kW, 60 C inlet, valves only) that the
valve resistance is calibrated to. Estimates: pump head, core pressure drop, loop inertias,
the coastdown time constant and friction torque (pumps stop ~18 s after a trip), the
pump-loop drop to the heat exchangers, and the check-valve factor. No MITR coastdown data
was found. The plant starts in decay-heat equilibrium (long operation). Commands in the
YAML scenario: `{time: t, pumps: 0}` trips both pumps.

#### Shim blades: fitted to published HEU worths
Published (HEU SAR Table 4-4 and MCNP values quoted in ANL/RERTR/TM-10-40, Tables 1, 4, 12,
13): bank integral worth 9927 pcm (12.63 beta); HEU core 1 initial critical bank height
7.36 in (18.69 cm); typical excess reactivity 6-7 beta; typical bank worth 10061 pcm of which
7388 pcm lies between 5 in and fully out; blade #5 integral worth 1776 pcm; one blade 5 in to
out 1258 pcm; shutdown margin 3.35-3.56 % dk/k with one blade out (HEU core 2), tech-spec
minimum 1 %. No OpenMC blade worths for MITR were found; these MCNP/SAR values stand in.

`python3 verify_blades.py calibrate` fits two constants to the first two values on the 2 cm
mesh (regulating rod at 15 cm): the blade's thermal absorption Sa2 (= 1.237 /cm; Sa1 kept at
0.03 Sa2, an estimate) and a common scale on the fuel's nu-Sigma_f (x 0.9912). The fuel had to
be part of the fit: the old fuel gave ~8800 pcm of excess reactivity, which the old blades
(Sa2 = 10 /cm, worth 14,000-19,000 pcm) were compensating; blades worth 9927 pcm could not
have shut that core down. Everything else in the list above is compared, not fitted.

The fit also exposed a rasterization bug: blades were sampled on the material subsampling
grid, which is coarser than the 0.6 cm blade, so on the 4 cm mesh the two blades parallel to
the y axis were missing entirely (worth 0). Blades now use a 0.05 cm sub-grid, like the
regulating rod, and their cross-section areas are exact on every mesh.

Individual blades can be moved: `model.problem_at([h0, ..., h5], reg)` (heights in the order
of `shim_bank.sides`); a single number still moves the whole bank.

#### Regulating rod and automatic power control
Published (MITR) and used as such: one regulating rod, cadmium on aluminum, with no scram
capability, used to hold power by hand or by an automatic controller (tech specs
ML102320068, 1.3.31); it stands at a corner of the core hexagon between two shim blades;
integral worth 124 pcm (158 mbeta, HEU SAR value quoted in the ANL HEU-to-LEU conversion
analysis); its worth grows as the shim bank is withdrawn (shadowing; ANL benchmark of the
LEU core: roughly doubles over the bank's range); tech-spec limits for a device on an
automatic controller: < 0.5 % dk/k available positive reactivity, insertion < 5e-4 dk/k/s,
and the controller must be separate from the safety system.
Estimates: which corner, the rod's radius (0.9 cm) and radial position, travel (53 cm),
drive speed (0.5 cm/s), the nominal position (15 cm, just below the 18.7 cm bank tip; above
the tip the blades shadow it so strongly that it is worth almost nothing) and the absorber
constants (`reg_rod` Sa2 = 17.3 /cm is fitted so the integral worth is 124 pcm on the 2 cm mesh
at the critical bank height: `python3 verify_control.py calibrate`; cadmium's real thermal
absorption is ~100 /cm, so a near-black "equivalent" value is plausible).

The controller (`controller.py`, constants in `power_controller`) is not MITR's: no
description of MITR's controller was found. It is a sampled-data period-demand law: every
second it measures the log power error x = ln(P/P_set) and the inverse period, demands an
inverse period -x/30 s (limited to +-1/30 s), and drives the rod at a speed proportional to
the difference (limited to the drive speed). The rod position integrates, so the power
settles exactly on the setpoint, with or without feedback. It reads the fission power, not
the trip channels (separate from the safety system, as the tech specs require), and drops
to manual on a scram. Gains were tuned on point kinetics: ~5 % overshoot of a +10 %
setpoint step and ~4 min to settle within 0.1 %. Scenario commands:
`{time: t, auto: true, setpoint_w: P}`, `{time: t, auto: false}`, `{time: t, reg_rod: cm}`.

## Verification status

| Check | Result |
|---|---|
| Bare cube k vs analytic | 2nd-order convergence, 8e-6 error at 64^3 |
| Reflected slab vs independent 1D eigen solve | agrees to 3e-10 |
| Subcritical multiplication | M = 1/(1-k) to 1e-4 |
| Kinetics vs exact separable solution | direct: 1st order; PCQS with 0.5 s steps: 3e-5 |
| PCQS vs extrapolated reference (MITR core, scram) | 0.25 s steps: 1e-6 before, 1e-3 during (reference uncertainty 7e-4), 1e-5 after the blade drop |
| Inhour equation vs simulated period | 18.2 s vs 18.2 s |
| Feedback sensitivities vs exact reactivity | 6e-5 |
| Thermal energy conservation | 2e-14 |
| Coupled transient vs extrapolated reference | PCQS (1 s steps): 3e-4 in power, 0.014 K in temperature |
| Transient end state vs static feedback equilibrium | 3.2723 vs 3.2739 MW (decay heat off: it needs days to settle) |
| Live run vs offline run of the same scenario | 7.4e-6 |
| Source-driven state vs exact separable solution | 4e-13 |
| Source kinetics, -$8.1 to +$0.6 (power x3949) vs exact | PCQS 1 s steps: 6e-5; direct: 1st order |
| MITR source state vs independent fixed-source iteration | 3e-9 |
| Count-rate meter noise vs counting theory | 0.908 vs 0.905 cps |
| Period meters vs true stable period | 417.2 s vs 417.2 s |
| Trips | period (continuous withdrawal); 6.51 MW; low primary flow 0.78 s after both pumps trip at 100 kW; 80.4 kW in natural circulation |
| 1/M approach ECP vs static critical height (4 cm mesh) | 16.26 vs 16.12 cm (0.55 of a 0.1 inch step) |
| Decay-heat groups vs ANS-5.1 closed form (30 days at power, then 1 s ... 1 day) | 2e-16 |
| ANS-5.1 decay heat vs published MITR analyses | 4.40 % vs 4.39 % 16 s after shutdown (2003 analysis); 2.40 % vs 2.53 % 300 s after 4200 MWd (LEU PSAR, ORIGEN-S) |
| Pump coastdown in the solver vs exact solutions | pump speed 1e-15; core flow vs exact Riccati solution 8e-8 |
| Natural circulation vs independent algebraic steady state | flow 9e-10, temperatures 4e-8 K |
| Natural circulation vs published MULCH-II point (calibration) | 1.37 vs 1.35 kg/s at 100 kW, 60 C inlet |
| Loss of flow at 6 MW (pump trip, scram) vs extrapolated reference | PCQS 0.2 s: power 1.4e-4 before the scram, 1.3e-4 after the blade drop; flow 6e-7; hottest plate 0.02 K; NCVs open at 21.8 s in every run |
| Energy conservation with decay heat and changing flow | 1e-15 (PCQS), 1e-13 (direct) |
| Live pacing at 5x requested | 4.7x achieved; 0.97x after dropping back from an unreachable speed (no catch-up burst) |
| Multigrid preconditioner: symmetry, CG vs direct solve, mesh independence, MITR 2 cm k | 1e-17; 3e-14; CG iterations 14 -> 15 from 32^3 to 64^3 (Jacobi 40 -> 82); k equal to 6e-11 |
| Anderson-accelerated vs plain power iteration (MITR 4 cm, tight tolerance) | k equal to 8e-13, thermal flux to 1e-9; 37 vs 118 outer iterations |
| Blade areas on the mesh vs thickness x width | 2e-16 (2 cm), 8e-4 (4 cm); before the fix two blades were missing on the 4 cm mesh |
| Six blades moved individually vs the bank | identical cross sections (0 difference) |
| Shim bank fit (targets): worth, critical height vs published HEU | 9924 vs 9927 pcm (3e-4); 18.70 vs 18.69 cm |
| Not fitted: excess reactivity vs published | 6723 pcm = 8.55 beta vs 6-7 beta |
| Not fitted: share of bank worth from 5 in to out | 0.84 vs 0.73 |
| Not fitted: single-blade worth vs blade #5 | 1151-2224 pcm (mean 1740) vs 1776 pcm; 5 in -> out mean 1417 vs 1258 pcm |
| Not fitted: shutdown margin, one blade out | 0.76-1.99 % vs 3.35-3.56 % (tech-spec minimum 1 %) |
| Regulating rod integral worth vs published HEU value (calibration target) | 124.00 vs 124 pcm (2 cm mesh, bank at its critical 18.70 cm); 91 pcm on the 4 cm live mesh |
| Regulating rod vs tech-spec limits for automatic control | 55.5 pcm available positive reactivity (limit 500); at most 3.4 pcm/s (limit 50) |
| Shadowing: rod worth vs bank height vs published trend | 124 -> 290 pcm from the critical bank to fully out (x2.3); published (LEU analysis): 146 -> 301 pcm (x2.1) |
| Rod withdrawal 8 -> 18 cm: transient vs static reactivity (1 - k0/k1); period vs inhour | 2.0e-6; 170.33 vs 170.33 s (2.4e-6) |
| AUTO at zero power: setpoint +10 %, then a +9 pcm bank disturbance | power on the setpoint to 1e-8; final rod vs static search for k = k0: 4e-7 cm (8e-6 pcm); 7 % overshoot |
| AUTO at 2 MW with feedback: setpoint +10 % | power on the setpoint to 5e-8; final rod vs static feedback-equilibrium search: 4e-5 cm (4e-4 pcm) |
| AUTO closed loop (1 MW, feedback, setpoint +5 %) vs extrapolated direct reference | PCQS 1 s steps: power 2.2e-5, rod 5e-4 cm, plate 7e-4 K; 0.5 s: 6.3e-6 |
| Live AUTO vs offline run of the same commands | 2e-10; a manual rod command or a scram switches AUTO off at once, rod stays put |

The GUIs were written without a display, so their windows have never been opened. Their
physics and solver logic are tested headlessly, and `core_viewer.py` (both modes) was also
exercised end to end with stand-in Qt/PyVista classes: every button, slider, field, pick,
scram and reset path runs against the real engine, with the matplotlib plots rendered.

## Known limitations
- Diffusion theory is crude for the thin, near-black shim blades; blade constants are
  "equivalent" values fitted to two published numbers, and rod cusping is only partly
  corrected (critical height +-1 cm).
- The bank's worth is right in total but sits too high: 84 % of it lies above 5 in
  (published 73 %), so the excess reactivity is 8.55 beta (published 6-7) and the shutdown
  margin with one blade out is 0.8-2.0 % (published 3.4-3.6 %; below the 1 % tech-spec
  minimum for two blades). Refits with 10x more fast absorption or without the cusping
  correction changed the excess by < 1 %, so this is the axial flux shape (axial reflectors,
  fuel), not the blades.
- Single-blade worths vary 1151-2224 pcm around the core (published shutdown margins vary
  only 3.35-3.56 %): the estimated dummy-element layout (A-2, A-3, B-5) tilts the flux.
- The 4 cm live mesh differs from the 2 cm mesh: critical bank 16.1 vs 18.7 cm.
- Elements are homogenized, so plate-edge hot spots are not resolved.
- No boiling or D2O photoneutrons. Inlet temperature changes are instantaneous steps, and
  the inlet temperature stays fixed after a loss of flow (pool heat-up not modeled).
- Loop hydraulics are lumped: all channels get the same flow (no redistribution toward hot
  channels in natural circulation), friction is quadratic (no laminar term), and the
  anti-siphon valves are not modeled. Pump and loop parameters are estimates; the core flow
  takes ~60 s to reach its natural-circulation minimum, while the published MITR analyses
  show the second temperature peak ~16 s after loss of flow.
- During the 0.44 s blade drop the direct reference is not yet converged (~1 %), so PCQS
  accuracy there is only known to ~2 %.
- The adiabatic method is inaccurate after a scram (6-15 %); use `pcqs`.
- The direct method switches the natural convection valves at a step boundary when the
  switching function crosses zero there, so the switch can be up to one step late
  (first order, like the method). Before this was fixed it could miss the opening entirely:
  with the refitted blades the direct LOF runs never opened the valves while PCQS did.
- The regulating rod is a thin cadmium rod smeared over 2-4 cm cells with fitted
  "equivalent" constants: its integral worth is right by construction on the 2 cm mesh
  (91 pcm on the 4 cm live mesh), but its worth curve and shadowing are diffusion-theory
  shapes. Shadowing: worth x2.3 from the critical bank to fully out (published LEU analysis:
  146 -> 301 pcm, x2.1), but x4.6 from 12 cm to fully out.
- The controller sees the true fission power (no detector noise, signal lag or calibration
  drift), and nothing moves the shim bank automatically when the rod runs out of travel.

## Next steps
- MITR's wide-range Campbelling channels with 2-out-of-4 trip logic.
- Bank worth shape / excess reactivity: needs the axial geometry (grid plates, axial
  reflectors) or fuel data, not blade constants.
- Map this program's blade sides and positions to the MITR numbering, so per-blade worths
  can be compared one by one.
