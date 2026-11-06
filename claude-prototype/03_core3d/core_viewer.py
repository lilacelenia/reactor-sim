"""
core_viewer.py -- interactive 3D viewer for a YAML-configured core.

    python3 -m pip install numpy scipy matplotlib pyyaml pyside6 pyvista pyvistaqt
    python3 core_viewer.py mitr2_like.yaml            # static explorer (k-eigenvalue)
    python3 core_viewer.py mitr2_like.yaml --live     # live transient with feedback
    python3 core_viewer.py mitr2_like.yaml --startup  # start up from the Pu-Be source

Static mode
  * Shim bank slider: blades move immediately; the core is re-solved in the background
    (a few seconds; results are cached, so revisiting a height is instant).
  * "Find critical": secant search for the bank height where k = 1.

Live mode (quasi-static kinetics + channel thermal feedback in a background thread)
  * Run / Pause, speed (time compression), Reset. If the computer cannot keep up with the
    requested speed, the status says "lagging" and shows the speed actually achieved.
  * The shim bank slider is the *commanded* position: blades drive there at the drive speed
    (0.2 cm/s). Fine-step buttons move the target by 0.05 cm. SCRAM drops the blades; after
    a scram (manual or high-power trip) press Reset to start again.
  * Coolant inlet temperature, pump speed and the number of primary pumps running can be
    changed while running. Tripping pumps starts a coastdown; with none running the core
    cools by natural circulation through the natural convection valves (decay heat).
  * Regulating rod (no scram): +-0.5 cm buttons move it by hand. "Automatic power control"
    puts it on the controller, which holds the power present at that moment; type another
    setpoint to change power (slowly: the rod is worth ~124 pcm, ~2 pcm/s at most). A manual
    rod command or a scram switches automatic control off.
  * Instruments: source-range fission chambers (count rate), period channels, linear power
    channels, with the MITR scrams (10 s period, 6.5 MW, 80 kW without flow, low flow).
  * Options: --mesh 4 (default for live; ~3-5x real time) or --mesh 2 (fine, slower than
    real time), --power W (initial power, default from the YAML), --no-feedback,
    --startup (start shut down: blades in, Pu-Be source in, a few counts per second).

Startup (--startup): approach criticality the way operators do. Press "Count" to take a
source-range count at the current blade height; each count adds a point to the 1/M plot and
updates the estimated critical position (ECP). Withdraw in steps, wait for the count rate to
settle, count again; near critical use the 0.05 cm buttons. Too large a step near critical
trips the 10 s period scram, as on the real reactor.

Both modes
  * Display: element power, hottest plate temperature (live), or slices of power density /
    thermal / fast flux. Right-click an element (or pick it in the list) for its axial shape.
  * Mouse: left-drag rotate, scroll zoom, shift-drag pan, "r" resets the camera.
"""
import argparse
import os
import sys
import time

os.environ.setdefault("QT_API", "pyside6")

import numpy as np
import pyvista as pv
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QMainWindow, QPushButton, QSlider, QVBoxLayout, QWidget,
)
from pyvistaqt import QtInteractor

from core_builder import build, load_config
from instruments import _fmt_w
from viewer_core import SimController, blade_boxes, element_prisms

SLICES = {
    "Power density (slices)": ("power_density", "Relative power density", "inferno"),
    "Thermal flux (slices)": ("phi2", "Thermal flux (rel.)", "viridis"),
    "Fast flux (slices)": ("phi1", "Fast flux (rel.)", "plasma"),
}
FIELDS = {"Element power": None, **SLICES}
LIVE_FIELDS = {"Element power": None, "Hottest plate temperature": "temp", **SLICES}
BLADE_COLOR = "#3a7bd5"
REG_COLOR = "#e67e22"
REG_STEP = 0.5              # cm per regulating rod button click
PANEL_WIDTH = 350


class CoreViewer(QMainWindow):
    """Static explorer: k-eigenvalue solves at the slider's blade position."""
    fields = FIELDS

    def __init__(self, cfg_path, cell_size=None):
        super().__init__()
        self._setup_scene(cfg_path, cell_size)
        self.setWindowTitle(f"{self.cfg['name']} -- 3D viewer")
        self.ctrl = SimController(self.cfg)
        self._build_ui()
        self._draw_static()
        self._draw_blades(self.bank["withdrawn"])
        self._draw_reg(self.base.reg_default)
        self._draw_elements()

        self.debounce = QTimer(self, singleShot=True, interval=250)
        self.debounce.timeout.connect(self._request_current)
        self.poller = QTimer(self, interval=100)
        self.poller.timeout.connect(self._poll)
        self.poller.start()
        self._request_current()

    # ------------------------------------------------------------------ shared setup
    def _setup_scene(self, cfg_path, cell_size):
        self.cfg = load_config(cfg_path)
        if cell_size:
            self.cfg["mesh"]["cell_size"] = float(cell_size)
        self.bank = self.cfg["control"]["shim_bank"]
        self.travel = float(self.bank["travel"])
        self.z0 = float(self.cfg["core"]["fuel_bottom"])
        self.H = float(self.cfg["core"]["fuel_height"])
        self.base = build(self.cfg)
        self.lattice = self.base.lattice
        self.labels = self.base.labels
        m = self.base.problem.mesh
        self.mesh = m
        self.z_top = m.z0 + m.nz * m.dz
        self.grid = pv.ImageData(dimensions=(m.nx + 1, m.ny + 1, m.nz + 1),
                                 spacing=(m.dx, m.dy, m.dz), origin=(m.x0, m.y0, m.z0))
        self.prism_pts, self.prism_faces, self.face_elem = element_prisms(self.lattice, self.z0, self.H)
        self.snap = None
        self.selected = None

    # ------------------------------------------------------------------ layout
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        left = QVBoxLayout()
        root.addLayout(left, 1)
        self.plotter = QtInteractor(central)
        self.plotter.set_background("#1e2228", top="#3b4250")
        left.addWidget(self.plotter.interactor, 1)
        self._build_trends(left)

        panel = QVBoxLayout()
        root.addLayout(panel, 0)
        self._build_controls(panel)
        self._build_display(panel)
        self._build_axial(panel)
        panel.addStretch()
        for w in (panel.itemAt(i).widget() for i in range(panel.count())):
            if w is not None:
                w.setFixedWidth(PANEL_WIDTH)
        self.resize(1450, 920)

        try:   # right-click picking (method name varies across PyVista versions)
            pick = getattr(self.plotter, "enable_surface_point_picking", None) or \
                self.plotter.enable_surface_picking
            pick(callback=self._on_pick, show_message=False, show_point=False)
        except Exception as exc:  # noqa: BLE001 -- picking is optional
            print("3D picking unavailable, use the element list instead:", exc)

    def _build_trends(self, layout):
        pass                                     # live mode adds trend plots here

    def _build_controls(self, panel):
        g = QGroupBox("Shim bank")
        gl = QVBoxLayout(g)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, int(round(self.travel * 10)))
        self.slider.setValue(int(round(self.bank["withdrawn"] * 10)))
        self.slider.valueChanged.connect(self._on_slider)
        self.slider_lbl = QLabel()
        self.crit_btn = crit_btn = QPushButton("Find critical height")
        crit_btn.clicked.connect(self._find_critical)
        gl.addWidget(QLabel(f"Withdrawal (0 = fully inserted, {self.travel:.0f} cm = out)"))
        gl.addWidget(self.slider)
        gl.addWidget(self.slider_lbl)
        gl.addWidget(crit_btn)
        panel.addWidget(g)
        self.read = self._readout_group(panel, "Core state",
                                        ("k-eff", "Reactivity", "Peak element", "Solved at", "Status"))
        self._update_slider_label()

    def _readout_group(self, panel, title, names):
        g = QGroupBox(title)
        fl = QFormLayout(g)
        out = {}
        for name in names:
            lbl = QLabel("-")
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            fl.addRow(name, lbl)
            out[name] = lbl
        panel.addWidget(g)
        return out

    def _build_display(self, panel):
        g = QGroupBox("Display")
        fl = QFormLayout(g)
        self.field_box = QComboBox()
        self.field_box.addItems(list(self.fields))
        self.field_box.currentTextChanged.connect(lambda _: self._redraw_state())
        self.zslice = QSlider(Qt.Horizontal)
        self.zslice.setRange(0, 100)
        self.zslice.setValue(30)
        self.zslice.valueChanged.connect(lambda _: self._redraw_state())
        fl.addRow("Field", self.field_box)
        fl.addRow("Horizontal slice", self.zslice)
        self.checks = {}
        for name, default in (("Reflector", True), ("Blades", True), ("Labels", False)):
            cb = QCheckBox(name)
            cb.setChecked(default)
            cb.toggled.connect(self._apply_visibility)
            fl.addRow(cb)
            self.checks[name] = cb
        panel.addWidget(g)

    def _build_axial(self, panel):
        g = QGroupBox("Element axial power (right-click an element)")
        vl = QVBoxLayout(g)
        self.elem_box = QComboBox()
        self.elem_box.addItems(sorted(self.labels, key=lambda s: (s[0], int(s[2:]))))
        self.elem_box.currentTextChanged.connect(self._select_element)
        self.fig = Figure(figsize=(3.2, 2.8), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.fig)
        vl.addWidget(self.elem_box)
        vl.addWidget(self.canvas)
        panel.addWidget(g)

    # ------------------------------------------------------------------ drawing
    def _draw_static(self):
        sur = self.cfg["surroundings"]
        zc, h = (self.mesh.z0 + self.z_top) / 2, self.z_top - self.mesh.z0
        tank = pv.Cylinder(center=(0, 0, zc), direction=(0, 0, 1), radius=sur["tank_radius"],
                           height=h, resolution=96, capping=False)
        d2o = pv.Cylinder(center=(0, 0, zc), direction=(0, 0, 1),
                          radius=sur["reflector"]["outer_radius"], height=h, resolution=96, capping=False)
        self.refl_actors = [
            self.plotter.add_mesh(tank, color="#9ecae1", opacity=0.10, name="tank"),
            self.plotter.add_mesh(d2o, color="#6baed6", opacity=0.07, name="d2o"),
            self.plotter.add_mesh(d2o.extract_feature_edges(), color="#6baed6", name="d2o_edges"),
        ]
        self.plotter.add_axes()
        self.plotter.view_isometric()
        self.plotter.camera.zoom(1.6)

    def _draw_blades(self, h):
        pts, faces = blade_boxes(self.cfg, self.lattice, h, self.z_top)
        self.blade_actor = self.plotter.add_mesh(pv.PolyData(pts, faces), color=BLADE_COLOR,
                                                 opacity=0.85, name="blades", reset_camera=False)
        self.blade_actor.SetVisibility(self.checks["Blades"].isChecked())
        self._drawn_blade = h

    def _draw_reg(self, r):
        """Regulating rod: cylinder from its tip (fuel bottom + r) up to the domain top."""
        if r is None or not np.isfinite(r):
            return
        from core_builder import regulating_rod_xy
        (cx, cy), rad = regulating_rod_xy(self.cfg)
        z_tip = self.z0 + float(r)
        cyl = pv.Cylinder(center=(cx, cy, 0.5 * (z_tip + self.z_top)), direction=(0, 0, 1),
                          radius=rad, height=self.z_top - z_tip, resolution=24)
        self.reg_actor = self.plotter.add_mesh(cyl, color=REG_COLOR, opacity=0.9, name="reg_rod",
                                               reset_camera=False)
        self.reg_actor.SetVisibility(self.checks["Blades"].isChecked())
        self._drawn_reg = float(r)

    def _element_values(self):
        """(per-element values, colorbar title, colormap) for the current element field."""
        spec = self.fields[self.field_box.currentText()]
        if spec == "temp":
            V = np.asarray(self.snap.element_temp, float)
            return V, "Hottest plate [C]", "coolwarm"
        P = np.asarray(self.snap.element_power, float)
        return np.where(P > 0, P, np.nan), "Element power", "inferno"

    def _draw_elements(self):
        mesh = pv.PolyData(self.prism_pts, self.prism_faces)
        spec = self.fields[self.field_box.currentText()]
        slices_mode = isinstance(spec, tuple)
        self._remove_bars(only="Element power")
        self._remove_bars(only="Hottest plate [C]")
        V = None
        if self.snap is not None and not slices_mode:
            V, title, cmap = self._element_values()
            if not np.isfinite(V).any():
                V = None
        if V is None:
            # outline only (no data yet, or slices are the focus)
            style = dict(style="wireframe", color="white", opacity=0.35) if slices_mode else \
                dict(color="#8a8f98", show_edges=True)
            self.plotter.add_mesh(mesh, name="elements", reset_camera=False, **style)
            return
        fin = V[np.isfinite(V)]
        lo, hi = float(fin.min()), float(fin.max())
        if hi - lo < 1e-9:
            hi = lo + 1e-6
        mesh.cell_data[title] = V[self.face_elem]
        self.plotter.add_mesh(mesh, scalars=title, cmap=cmap, nan_color="#b0b8c0", clim=(lo, hi),
                              show_edges=True, edge_color="black", name="elements", reset_camera=False,
                              scalar_bar_args=dict(title=title, vertical=True))

    def _draw_slices(self):
        for name in ("slice_z", "slice_y"):
            self.plotter.remove_actor(name)
        for s in SLICES.values():
            self._remove_bars(only=s[1])
        spec = self.fields[self.field_box.currentText()]
        if not isinstance(spec, tuple) or self.snap is None:
            return
        attr, title, cmap = spec
        arr = getattr(self.snap, attr)
        self.grid.cell_data["f"] = np.ravel(arr, order="F")      # VTK: x varies fastest
        finite = arr[np.isfinite(arr)]
        clim = (float(finite.min()), float(finite.max()))
        zlo, zhi = self.mesh.z0 + 0.5, self.z_top - 0.5
        z = zlo + (zhi - zlo) * self.zslice.value() / 100
        common = dict(scalars="f", cmap=cmap, clim=clim, nan_opacity=0.0, reset_camera=False)
        self.plotter.add_mesh(self.grid.slice(normal="z", origin=(0, 0, z)), name="slice_z",
                              scalar_bar_args=dict(title=title, vertical=True), **common)
        self.plotter.add_mesh(self.grid.slice(normal="y", origin=(0, 0.01, 0)), name="slice_y",
                              show_scalar_bar=False, **common)

    def _remove_bars(self, keep=None, only=None):
        try:
            titles = list(self.plotter.scalar_bars.keys())
        except Exception:  # noqa: BLE001
            titles = []
        for t in titles:
            if t != keep and (only is None or t == only):
                try:
                    self.plotter.remove_scalar_bar(t)
                except Exception:  # noqa: BLE001
                    pass

    def _draw_labels(self):
        self.plotter.remove_actor("labels")
        pts = np.array([[*e["center"], self.z0 + self.H + 1.5] for e in self.lattice.elements])
        if self.snap is None:
            text = self.labels
        elif self.fields[self.field_box.currentText()] == "temp":
            text = [f"{l}\n{T:.0f} C" if np.isfinite(T) else f"{l}\nAl"
                    for l, T in zip(self.labels, self.snap.element_temp)]
        else:
            text = [f"{l}\n{p:.2f}" if p > 0 else f"{l}\nAl"
                    for l, p in zip(self.labels, self.snap.element_power)]
        self.label_actor = self.plotter.add_point_labels(
            pts, text, name="labels", font_size=10, point_size=1, shape_opacity=0.35,
            always_visible=True, reset_camera=False)
        self.label_actor.SetVisibility(self.checks["Labels"].isChecked())

    def _draw_highlight(self):
        self.plotter.remove_actor("highlight")
        if self.selected is None:
            return
        sel = np.nonzero(self.face_elem == self.selected)[0]
        faces = self.prism_faces.reshape(-1, 5)[sel].ravel()
        mesh = pv.PolyData(self.prism_pts, faces)
        self.plotter.add_mesh(mesh, style="wireframe", color="cyan", line_width=4,
                              name="highlight", reset_camera=False, pickable=False)

    def _redraw_state(self):
        self._draw_elements()
        self._draw_slices()
        self._draw_labels()
        self._draw_highlight()
        self._apply_visibility()
        self.plotter.render()

    def _apply_visibility(self, *_):
        for a in self.refl_actors:
            a.SetVisibility(self.checks["Reflector"].isChecked())
        if getattr(self, "reg_actor", None) is not None:
            self.reg_actor.SetVisibility(self.checks["Blades"].isChecked())
        self.blade_actor.SetVisibility(self.checks["Blades"].isChecked())
        if hasattr(self, "label_actor"):
            self.label_actor.SetVisibility(self.checks["Labels"].isChecked())
        self.plotter.render()

    def _plot_axial(self):
        self.ax.clear()
        if self.snap is None or self.selected is None:
            self.canvas.draw_idle()
            return
        i = self.selected
        label = self.labels[i]
        if self.snap.element_power[i] <= 0:
            self.ax.text(0.5, 0.5, f"{label}: no fuel", ha="center", transform=self.ax.transAxes)
        else:
            prof = self.snap.model.element_axial_profile(self.snap.result, i)
            z = self.mesh.centers("z")
            m = prof > 0
            # scale so the curve's mean equals the element's relative power
            self.ax.plot(prof[m] / prof[m].mean() * self.snap.element_power[i], z[m], "o-", ms=2.5)
            self.ax.axhline(self.z0 + self.snap.withdrawn, color=BLADE_COLOR, ls="--", lw=1,
                            label="blade tips")
            self.ax.legend(fontsize=7)
        self.ax.set_title(f"{label}  (element power {self.snap.element_power[i]:.2f})", fontsize=9)
        self.ax.set_xlabel("relative power density", fontsize=8)
        self.ax.set_ylabel("height [cm]", fontsize=8)
        self.ax.tick_params(labelsize=7)
        self.ax.grid(alpha=0.3)
        self.canvas.draw_idle()

    def _select_hottest_if_unset(self):
        if self.selected is None and self.snap is not None:
            hot = int(np.argmax(self.snap.element_power))
            self.elem_box.blockSignals(True)
            self.elem_box.setCurrentText(self.labels[hot])
            self.elem_box.blockSignals(False)
            self.selected = hot

    # ------------------------------------------------------------------ static-mode logic
    def _show_snapshot(self, snap):
        self.snap = snap
        P = snap.element_power
        hot = int(np.argmax(P))
        self.read["k-eff"].setText(f"{snap.k:.5f}")
        self.read["Reactivity"].setText(f"{snap.rho_pcm:+.0f} pcm  ({snap.rho_pcm / 1000:+.2f} % dk/k)")
        self.read["Peak element"].setText(f"{self.labels[hot]}  ({P.max():.3f} x avg)")
        self.read["Solved at"].setText(f"{snap.withdrawn:.2f} cm  ({snap.solve_seconds:.1f} s)")
        self._select_hottest_if_unset()
        if abs(snap.withdrawn - self.slider.value() / 10) > 0.05:
            # e.g. after a critical search: move the slider without triggering a new solve
            self.slider.blockSignals(True)
            self.slider.setValue(int(round(snap.withdrawn * 10)))
            self.slider.blockSignals(False)
            self._update_slider_label()
            self._draw_blades(snap.withdrawn)
        self._redraw_state()
        self._plot_axial()
        self._update_status()

    def _update_status(self):
        h = self.slider.value() / 10
        if self.ctrl.busy:
            what = self.ctrl.busy_with
            txt = "searching for critical..." if isinstance(what, str) else f"solving {what:.1f} cm..."
        elif self.snap is not None and abs(self.snap.withdrawn - h) > 0.05:
            txt = "queued"
        else:
            txt = self.ctrl.message or "up to date"
        self.read["Status"].setText(txt)

    def _update_slider_label(self):
        self.slider_lbl.setText(f"Commanded position: {self.slider.value() / 10:.1f} cm")

    def _on_slider(self, v):
        self._update_slider_label()
        self._draw_blades(v / 10)           # instant visual feedback
        self.plotter.render()
        self.debounce.start()               # solve once the slider settles

    def _request_current(self):
        snap = self.ctrl.request(self.slider.value() / 10)
        if snap is not None:
            self._show_snapshot(snap)
        self._update_status()

    def _poll(self):
        try:
            snap = self.ctrl.poll()
        except Exception as exc:  # noqa: BLE001 -- show solver errors instead of crashing
            self.read["Status"].setText(f"solver error: {exc}")
            return
        if snap is not None:
            self._show_snapshot(snap)
        self._update_status()

    def _find_critical(self, *_):
        if self.ctrl.find_critical(self.slider.value() / 10):
            self._update_status()

    # ------------------------------------------------------------------ shared events
    def _select_element(self, label):
        if label in self.labels:
            self.selected = self.labels.index(label)
            self._draw_highlight()
            self.plotter.render()
            self._plot_axial()

    def _on_pick(self, point, *args):
        if point is None:
            return
        idx = int(self.lattice.locate(np.array([point[0]]), np.array([point[1]]))[0])
        if idx >= 0:
            self.elem_box.setCurrentText(self.labels[idx])

    def closeEvent(self, event):
        self.poller.stop()
        self.ctrl.shutdown()
        self.plotter.close()
        super().closeEvent(event)


# ============================================================================== live mode
SPEEDS = ["0.5x", "1x", "2x", "5x", "10x", "30x", "100x", "1000x"]
SLIDER_SCALE = 100          # slider units per cm (0.01 cm resolution)
FINE_STEP = 0.05            # cm per fine-step click


class LiveViewer(CoreViewer):
    """Live transient: LiveSimulation (quasi-static kinetics + thermal feedback) runs in a
    background thread; this window sends operator commands and draws its snapshots."""
    fields = LIVE_FIELDS

    def __init__(self, cfg_path, cell_size=4.0, power_w=None, feedback=True, method="pcqs",
                 startup=False):
        QMainWindow.__init__(self)
        from live_sim import LiveSimulation          # imported here: static mode doesn't need it
        self._setup_scene(cfg_path, cell_size)
        self.live = LiveSimulation(self.cfg, self.base, power_w=power_w, feedback=feedback,
                                   method=method, startup=startup)
        mode = "startup from source" if startup else "live transient"
        self.setWindowTitle(f"{self.cfg['name']} -- {mode} "
                            f"({self.mesh.dx:g} cm mesh, feedback {'on' if self.live.feedback else 'off'})")
        self._last_seq = -1
        self._last_3d = 0.0
        self._build_ui()
        self._draw_static()
        self._draw_blades(self.live.h0)
        self._draw_reg(self.live.r0)
        self._draw_elements()
        self.live.start(running=False)

        self.blade_debounce = QTimer(self, singleShot=True, interval=200)
        self.blade_debounce.timeout.connect(self._send_blade_target)
        self.timer = QTimer(self, interval=150)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        self.trend_timer = QTimer(self, interval=1000)
        self.trend_timer.timeout.connect(self._plot_trends)
        self.trend_timer.start()

    # ------------------------------------------------------------------ layout
    def _build_trends(self, layout):
        self.tfig = Figure(figsize=(12, 2.4), tight_layout=True)
        self.tax_p = self.tfig.add_subplot(1, 3, 1)
        self.tax_t = self.tfig.add_subplot(1, 3, 2)
        self.tax_m = self.tfig.add_subplot(1, 3, 3)
        self.tcanvas = FigureCanvasQTAgg(self.tfig)
        self.tcanvas.setFixedHeight(240)
        layout.addWidget(self.tcanvas, 0)

    def _build_controls(self, panel):
        live = self.live
        g = QGroupBox("Simulation")
        gl = QVBoxLayout(g)
        self.status_lbl = QLabel("initializing...")
        self.status_lbl.setWordWrap(True)
        row = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.run_btn.setCheckable(True)
        self.run_btn.toggled.connect(self._on_run)
        self.reset_btn = reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._on_reset)
        self.speed_box = QComboBox()
        self.speed_box.addItems(SPEEDS)
        self.speed_box.setCurrentText("1x")
        self.speed_box.currentTextChanged.connect(lambda s: live.set_speed(float(s[:-1])))
        row.addWidget(self.run_btn)
        row.addWidget(reset_btn)
        row.addWidget(QLabel("Speed"))
        row.addWidget(self.speed_box)
        gl.addWidget(self.status_lbl)
        gl.addLayout(row)
        panel.addWidget(g)

        g = QGroupBox("Shim bank")
        gl = QVBoxLayout(g)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, int(round(self.travel * SLIDER_SCALE)))
        self.slider.setValue(int(round(live.h0 * SLIDER_SCALE)))
        self.slider.valueChanged.connect(self._on_slider)
        self.slider_lbl = QLabel()
        row = QHBoxLayout()
        for txt, d in ((f"-{FINE_STEP} cm", -FINE_STEP), (f"+{FINE_STEP} cm", FINE_STEP)):
            b = QPushButton(txt)
            b.clicked.connect(lambda *_, d=d: self._nudge(d))
            row.addWidget(b)
        self.scram_btn = scram_btn = QPushButton("SCRAM")
        scram_btn.setMinimumHeight(48)
        scram_btn.setStyleSheet("QPushButton { background: #c0392b; color: white; "
                                "font-size: 18px; font-weight: bold; }")
        scram_btn.clicked.connect(lambda *_: live.scram())
        gl.addWidget(QLabel(f"Commanded withdrawal; blades drive at {live.drive_speed:g} cm/s"))
        gl.addWidget(self.slider)
        gl.addLayout(row)
        gl.addWidget(self.slider_lbl)
        gl.addWidget(scram_btn)
        panel.addWidget(g)

        self.auto_chk = self.sp_spin = None
        if live.has_reg:
            g = QGroupBox("Regulating rod (no scram)")
            gl = QVBoxLayout(g)
            self.reg_lbl = QLabel("-")
            row = QHBoxLayout()
            self.reg_btns = []
            for txt, d in ((f"-{REG_STEP} cm", -REG_STEP), (f"+{REG_STEP} cm", REG_STEP)):
                b = QPushButton(txt)
                b.clicked.connect(lambda *_, d=d: self._nudge_reg(d))
                row.addWidget(b)
                self.reg_btns.append(b)
            gl.addWidget(self.reg_lbl)
            gl.addLayout(row)
            if live.has_controller:
                self.auto_chk = QCheckBox("Automatic power control")
                self.auto_chk.toggled.connect(self._on_auto)
                self.sp_spin = QDoubleSpinBox()
                self.sp_spin.setRange(0.001, 7000.0)
                self.sp_spin.setDecimals(3)
                self.sp_spin.setSuffix(" kW")
                self.sp_spin.setValue(min(max(live.P0 / 1e3, 0.001), 7000.0))
                self.sp_spin.editingFinished.connect(self._on_setpoint)
                row = QHBoxLayout()
                row.addWidget(self.auto_chk)
                row.addWidget(QLabel("Setpoint"))
                row.addWidget(self.sp_spin)
                gl.addLayout(row)
                self.ctl_lbl = QLabel("manual")
                self.ctl_lbl.setWordWrap(True)
                gl.addWidget(self.ctl_lbl)
            panel.addWidget(g)
        self._shown_sp = None

        if live.has_instruments:
            g = QGroupBox("Nuclear instruments")
            fl = QFormLayout(g)
            self.inst_lbl = {}
            for ch in self.cfg["instrumentation"]["channels"]:
                lbl = QLabel("-")
                fl.addRow(ch["name"], lbl)
                self.inst_lbl[ch["name"]] = lbl
            tr = self.cfg["instrumentation"]["trips"]
            low = (f", flow < {float(tr['low_flow_relative']):.0%} above "
                   f"{_fmt_w(float(tr['high_power_no_flow_w']))}") if tr.get("low_flow_relative") else ""
            fl.addRow(QLabel(f"Scrams: period < {tr['short_period_s']:g} s, power > "
                             f"{_fmt_w(float(tr['high_power_w']))} "
                             f"({_fmt_w(float(tr['high_power_no_flow_w']))} without flow){low}"))
            self.count_lbl = QLabel("1/M: no counts yet")
            self.count_lbl.setWordWrap(True)
            row = QHBoxLayout()
            for secs in (30, 100):
                b = QPushButton(f"Count {secs} s")
                b.clicked.connect(lambda *_, x=secs: live.count(x))
                row.addWidget(b)
            clr = QPushButton("Clear 1/M")
            clr.clicked.connect(lambda *_: live.clear_one_over_m())
            row.addWidget(clr)
            fl.addRow(row)
            fl.addRow(self.count_lbl)
            panel.addWidget(g)

        g = QGroupBox("Coolant")
        fl = QFormLayout(g)
        T_in = float(self.cfg["thermal_hydraulics"]["inlet_temperature"]) - 273.15 \
            if "thermal_hydraulics" in self.cfg else 40.0
        self._T_in0 = T_in
        self.inlet_spin = QDoubleSpinBox()
        self.inlet_spin.setRange(10.0, 70.0)
        self.inlet_spin.setSingleStep(0.5)
        self.inlet_spin.setSuffix(" C")
        self.inlet_spin.setValue(T_in)
        self.inlet_spin.editingFinished.connect(
            lambda *_: live.set_inlet_temperature(self.inlet_spin.value() + 273.15))
        self.flow_spin = QDoubleSpinBox()
        self.flow_spin.setRange(0.3, 1.2)
        self.flow_spin.setSingleStep(0.05)
        self.flow_spin.setValue(1.0)
        self.flow_spin.editingFinished.connect(lambda *_: live.set_flow(self.flow_spin.value()))
        fl.addRow("Inlet temperature", self.inlet_spin)
        n_pumps = int((self.cfg.get("primary_loop") or {}).get("pumps", 0))
        self._n_pumps = n_pumps
        self.pump_box = None
        if n_pumps:
            fl.addRow("Pump speed (x nominal)", self.flow_spin)
            self.pump_box = QComboBox()
            self.pump_box.addItems([str(n) for n in range(n_pumps, -1, -1)])
            self.pump_box.setCurrentText(str(n_pumps))
            self.pump_box.currentTextChanged.connect(lambda s, *_: live.set_pumps(int(s)))
            fl.addRow("Primary pumps running", self.pump_box)
        else:
            fl.addRow("Flow (x nominal)", self.flow_spin)
        g.setEnabled(live.feedback)
        panel.addWidget(g)

        self.read = self._readout_group(panel, "Plant status", (
            "Sim time", "Power", "Period", "Reactivity", "Feedback", "Hottest plate",
            "Core outlet", "Primary loop", "Peak element"))
        self._update_slider_label()

    # ------------------------------------------------------------------ commands
    def _update_slider_label(self):
        s = self.snap
        actual = f"{s.withdrawn:.2f}" if s is not None else "-"
        self.slider_lbl.setText(f"Target {self.slider.value() / SLIDER_SCALE:.2f} cm   actual {actual} cm")

    def _on_slider(self, v):
        self._update_slider_label()
        self.blade_debounce.start()

    def _send_blade_target(self):
        self.live.command_blades(self.slider.value() / SLIDER_SCALE)

    def _nudge(self, d):
        self.slider.setValue(self.slider.value() + int(round(d * SLIDER_SCALE)))

    def _nudge_reg(self, d):
        s = self.snap
        base = s.reg_target if s is not None and np.isfinite(s.reg_target) else self.live.r0
        self.live.command_reg_rod(base + d)

    def _on_auto(self, checked, *_):
        self.live.set_auto(checked)          # on: holds the power at this moment

    def _on_setpoint(self, *_):
        self.live.set_setpoint(self.sp_spin.value() * 1e3)

    def _on_run(self, checked):
        self.live.set_running(checked)
        self.run_btn.setText("Pause" if checked else "Run")

    def _on_reset(self, *_):
        self.live.reset()
        self.run_btn.blockSignals(True)
        self.run_btn.setChecked(False)
        self.run_btn.setText("Run")
        self.run_btn.blockSignals(False)
        self.live.set_running(False)
        for w, v in ((self.slider, int(round(self.live.h0 * SLIDER_SCALE))),
                     (self.inlet_spin, self._T_in0), (self.flow_spin, 1.0)):
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)
        if self.pump_box is not None:
            self.pump_box.blockSignals(True)
            self.pump_box.setCurrentText(str(self._n_pumps))
            self.pump_box.blockSignals(False)
        if self.auto_chk is not None:
            self.auto_chk.blockSignals(True)
            self.auto_chk.setChecked(False)
            self.auto_chk.blockSignals(False)
            self._shown_sp = None
        self.slider.setEnabled(True)
        self.snap = None
        self._last_seq = -1

    # ------------------------------------------------------------------ updates
    def _tick(self):
        live = self.live
        status = live.status
        if live.error:
            status = live.status
            print(live.error, file=sys.stderr)
        elif status.startswith("running"):
            status += f" -- {live.achieved_speed:.1f}x achieved"
        snap = live.snapshot()
        if snap is not None and snap.scrammed:
            status = f"SCRAMMED ({snap.trip_reason}) -- press Reset to restart. " + status
        self.status_lbl.setText(status)
        self.status_lbl.setStyleSheet("color: #c0392b; font-weight: bold;"
                                      if (snap is not None and snap.scrammed) or live.error else "")
        if snap is None or snap.seq == self._last_seq:
            return
        self._last_seq = snap.seq
        self.snap = snap
        self.slider.setEnabled(not snap.scrammed)
        self._update_readouts(snap)
        self._update_instruments(snap)
        self._update_slider_label()
        self._update_reg(snap)
        if abs(snap.withdrawn - getattr(self, "_drawn_blade", -1.0)) > 0.005:
            self._draw_blades(snap.withdrawn)
            self.plotter.render()
        if np.isfinite(snap.reg_rod) and abs(snap.reg_rod - getattr(self, "_drawn_reg", -1.0)) > 0.005:
            self._draw_reg(snap.reg_rod)
            self.plotter.render()
        if time.monotonic() - self._last_3d > 0.4:      # throttle the heavier 3D redraws
            self._last_3d = time.monotonic()
            self._select_hottest_if_unset()
            self._redraw_state()
            self._plot_axial()

    def _update_readouts(self, s):
        beta = self.live.kin.beta
        r = self.read
        r["Sim time"].setText(f"{s.t:9.1f} s")
        r["Power"].setText(_fmt_power(s.power_w))
        r["Period"].setText("steady" if not np.isfinite(s.period) or abs(s.period) > 1e4
                            else f"{s.period:+.1f} s")
        r["Reactivity"].setText(f"{s.rho * 1e5:+.1f} pcm  (${s.rho / beta:+.3f})")
        if self.live.feedback:
            r["Feedback"].setText(f"{s.rho_feedback * 1e5:+.1f} pcm since start")
            r["Hottest plate"].setText(f"{s.fuel_max_c:.1f} C")
            r["Core outlet"].setText(f"{s.outlet_c:.1f} C (inlet {s.inlet_c:.1f} C, flow {s.flow:.2f})")
            loop = f"{s.core_flow_kg_s:.1f} kg/s ({s.flow:.0%})"
            if self._n_pumps:
                loop += (f", pumps {s.pumps_running}/{self._n_pumps} at {s.pump_speed:.0%}, "
                         f"NCVs {'open' if s.ncv_open else 'shut'}")
            loop += f", decay heat {_fmt_power(s.decay_heat_w)}"
            r["Primary loop"].setText(loop)
        else:
            for k in ("Feedback", "Hottest plate", "Core outlet", "Primary loop"):
                r[k].setText("feedback off")
        hot = int(np.argmax(s.element_power))
        r["Peak element"].setText(f"{self.labels[hot]}  ({s.element_power[hot]:.2f} x avg, "
                                  f"{_fmt_power(s.element_power_w[hot])})")

    def _update_reg(self, s):
        if not self.live.has_reg:
            return
        mode = "AUTO" if s.auto else "manual"
        self.reg_lbl.setText(f"actual {s.reg_rod:.2f} cm, target {s.reg_target:.2f} cm  ({mode}; "
                             f"{self.live.reg_speed:g} cm/s)")
        for b in self.reg_btns:
            b.setEnabled(True)               # a manual command also leaves AUTO
        if self.auto_chk is None:
            return
        if self.auto_chk.isChecked() != s.auto:          # e.g. a scram switched AUTO off
            self.auto_chk.blockSignals(True)
            self.auto_chk.setChecked(s.auto)
            self.auto_chk.blockSignals(False)
        self.auto_chk.setEnabled(not s.scrammed)
        if np.isfinite(s.setpoint_w) and s.setpoint_w != self._shown_sp:
            self._shown_sp = s.setpoint_w             # show the setpoint AUTO actually holds
            self.sp_spin.blockSignals(True)
            self.sp_spin.setValue(s.setpoint_w / 1e3)
            self.sp_spin.blockSignals(False)
        txt = (f"holding {_fmt_power(s.setpoint_w)}" if s.auto else "manual") + \
            (f" -- {s.control_message}" if s.control_message else "")
        if s.reg_at_limit:
            txt += " -- ROD AT END OF TRAVEL: move the shim bank"
        self.ctl_lbl.setText(txt)

    def _update_instruments(self, s):
        if not self.live.has_instruments:
            return
        for rd in s.instruments:
            txt = rd.text()
            if np.isfinite(rd.period) and abs(rd.period) < 1e4:
                txt += f"   period {rd.period:+.0f} s"
            self.inst_lbl[rd.name].setText(txt)
        msg = []
        if s.counting is not None:
            msg.append(f"counting... {s.counting[0]:.0f} / {s.counting[1]:.0f} s")
        if s.one_over_m:
            last = s.one_over_m[-1]
            msg.append(f"last count {last[1]:.1f} +- {last[2]:.1f} cps at {last[0]:.2f} cm, "
                       f"1/M = {last[3]:.3f}")
        if s.ecp is not None:
            msg.append(f"ECP = {s.ecp:.2f} cm")
        if s.k_initial is not None and not s.one_over_m:
            msg.append(f"start: k = {s.k_initial:.4f} (blades {self.live.h0:.1f} cm)")
        self.count_lbl.setText("   ".join(msg) if msg else "1/M: no counts yet")

    def _plot_trends(self):
        tr = self.live.trends()
        ap, at = self.tax_p, self.tax_t
        ap.clear()
        at.clear()
        if len(tr["t"]) >= 1:
            t = tr["t"]
            ap.semilogy(t, np.maximum(tr["power_w"], 1e-3), "C3")
            ap.set_xlabel("time [s]", fontsize=8)
            ap.set_ylabel("power [W]", fontsize=8)
            ap2 = ap.twinx()
            ap2.plot(t, tr["blade"], "C0", lw=1)
            if self.live.has_reg:
                ap2.plot(t, tr["reg_rod"], color=REG_COLOR, lw=1)
            ap2.set_ylabel("blades, reg rod [cm]", fontsize=8, color="C0")
            ap2.tick_params(labelsize=7)
            if self.live.feedback:
                at.plot(t, tr["fuel_max_c"], "C3", label="hottest plate")
                at.plot(t, tr["outlet_c"], "C0", label="core outlet")
                at.plot(t, tr["inlet_c"], "k:", label="inlet")
                at.legend(fontsize=7, loc="upper left")
            at.set_xlabel("time [s]", fontsize=8)
            at.set_ylabel("temperature [C]", fontsize=8)
        am = self.tax_m
        am.clear()
        s = self.snap
        if s is not None and s.one_over_m:
            pts = np.array([(p[0], p[3], p[4]) for p in s.one_over_m])
            am.errorbar(pts[:, 0], pts[:, 1], yerr=pts[:, 2], fmt="o-", ms=3, capsize=2)
            if s.ecp is not None:
                am.axvline(s.ecp, color="C3", ls="--", lw=1, label=f"ECP {s.ecp:.2f} cm")
                am.legend(fontsize=7)
            am.axvline(s.withdrawn, color="C0", ls=":", lw=1)
            am.set_ylim(0, max(1.05, pts[:, 1].max() * 1.05))
        am.set_xlabel("blades [cm]", fontsize=8)
        am.set_ylabel("1/M", fontsize=8)
        am.set_title("Inverse multiplication", fontsize=8)
        for a in (ap, at, am):
            a.tick_params(labelsize=7)
            a.grid(alpha=0.3, which="both")
        self.tcanvas.draw_idle()

    def closeEvent(self, event):
        self.timer.stop()
        self.trend_timer.stop()
        self.live.shutdown(timeout=5.0)
        self.plotter.close()
        QMainWindow.closeEvent(self, event)


def _fmt_power(w):
    return _fmt_w(w)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="3D core viewer (static explorer or live transient)")
    ap.add_argument("config", nargs="?", default="mitr2_like.yaml")
    ap.add_argument("--live", action="store_true", help="live transient with time compression")
    ap.add_argument("--mesh", type=float, help="cell size in cm (live default 4; static: from YAML)")
    ap.add_argument("--power", type=float, help="live: initial power in W (default from the YAML)")
    ap.add_argument("--no-feedback", action="store_true", help="live: switch thermal feedback off")
    ap.add_argument("--startup", action="store_true",
                    help="live: start shut down with the startup source (implies --live)")
    args = ap.parse_args()
    app = QApplication(sys.argv[:1])
    if args.live or args.startup:
        win = LiveViewer(args.config, cell_size=args.mesh or 4.0, power_w=args.power,
                         feedback=not args.no_feedback, startup=args.startup)
    else:
        win = CoreViewer(args.config, cell_size=args.mesh)
    win.show()
    sys.exit(app.exec())
