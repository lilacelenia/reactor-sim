"""
reactor_gui.py -- desktop control panel for the simulation in reactor_model.py.

Run:   python3 -m pip install pyside6 pyqtgraph numpy
       python3 reactor_gui.py

Architecture (the pattern for any GUI simulation):
  * Model  (reactor_model.Reactor): owns the state and the physics. Knows nothing about Qt.
  * View   (this window): widgets that show the state and send operator inputs to the model.
  * Loop   (a QTimer): ~20 times per second, advance the model by (frame time x speed),
           then refresh the readouts and plots. The GUI never blocks waiting on physics.
"""
import sys
from collections import deque

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QPushButton, QSlider, QVBoxLayout, QWidget,
)

from reactor_model import BETA, Reactor

FRAME_MS = 50          # GUI refresh interval (20 frames/s)
HISTORY_S = 300        # seconds of history shown in the plots


class ReactorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reactor Simulator")
        self.reactor = Reactor()
        self.speed = 1.0
        self.running = True
        self._init_history()
        self._build_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(FRAME_MS)

    # ------------------------------------------------------------------ UI layout
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.addLayout(self._build_controls(), 0)
        root.addLayout(self._build_plots(), 1)
        self.resize(1200, 700)

    def _build_controls(self):
        col = QVBoxLayout()

        # --- Control rods ---
        rods = QGroupBox("Control rods")
        rl = QVBoxLayout(rods)
        self.rod_slider = QSlider(Qt.Horizontal)
        self.rod_slider.setRange(0, 1000)                      # 0.1 % resolution
        self.rod_slider.setValue(int(self.reactor.ROD_CRIT * 10))
        self.rod_slider.valueChanged.connect(
            lambda v: self.reactor.set_rod_target(v / 10.0))
        self.rod_target_lbl = QLabel()
        self.rod_pos_lbl = QLabel()
        rl.addWidget(QLabel("Target position (% withdrawn)"))
        rl.addWidget(self.rod_slider)
        rl.addWidget(self.rod_target_lbl)
        rl.addWidget(self.rod_pos_lbl)
        col.addWidget(rods)

        # --- Coolant ---
        cool = QGroupBox("Coolant")
        cf = QFormLayout(cool)
        self.inlet_spin = QDoubleSpinBox()
        self.inlet_spin.setRange(500, 600)
        self.inlet_spin.setSingleStep(1.0)
        self.inlet_spin.setValue(self.reactor.T_inlet)
        self.inlet_spin.setSuffix(" K")
        self.inlet_spin.valueChanged.connect(lambda v: setattr(self.reactor, "T_inlet", v))
        self.flow_spin = QDoubleSpinBox()
        self.flow_spin.setRange(0.2, 1.2)
        self.flow_spin.setSingleStep(0.05)
        self.flow_spin.setValue(self.reactor.flow)
        self.flow_spin.valueChanged.connect(lambda v: setattr(self.reactor, "flow", v))
        cf.addRow("Inlet temperature", self.inlet_spin)
        cf.addRow("Relative flow", self.flow_spin)
        col.addWidget(cool)

        # --- Readouts ---
        read = QGroupBox("Plant status")
        grid = QGridLayout(read)
        self.readouts = {}
        names = ["Power", "Fuel temp", "Coolant temp", "Reactivity", "Period", "Sim time"]
        for i, name in enumerate(names):
            grid.addWidget(QLabel(name), i, 0)
            lbl = QLabel("-")
            lbl.setStyleSheet("font-family: monospace; font-size: 14px;")
            grid.addWidget(lbl, i, 1)
            self.readouts[name] = lbl
        self.status_lbl = QLabel()
        self.status_lbl.setAlignment(Qt.AlignCenter)
        grid.addWidget(self.status_lbl, len(names), 0, 1, 2)
        col.addWidget(read)

        # --- Simulation controls ---
        sim = QGroupBox("Simulation")
        sl = QGridLayout(sim)
        self.speed_box = QComboBox()
        self.speed_box.addItems(["1x", "2x", "5x", "10x", "30x"])
        self.speed_box.currentTextChanged.connect(lambda s: setattr(self, "speed", float(s[:-1])))
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.toggle_pause)
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self.reset)
        sl.addWidget(QLabel("Speed"), 0, 0)
        sl.addWidget(self.speed_box, 0, 1)
        sl.addWidget(self.pause_btn, 1, 0)
        sl.addWidget(reset_btn, 1, 1)
        col.addWidget(sim)

        # --- SCRAM ---
        scram_btn = QPushButton("SCRAM")
        scram_btn.setMinimumHeight(60)
        scram_btn.setStyleSheet(
            "QPushButton { background: #c0392b; color: white; font-size: 20px; font-weight: bold; }")
        scram_btn.clicked.connect(lambda: self.reactor.scram())
        col.addWidget(scram_btn)
        col.addStretch()
        return col

    def _build_plots(self):
        pg.setConfigOptions(antialias=True)
        layout = QVBoxLayout()
        self.power_plot = pg.PlotWidget(title="Reactor power")
        self.power_plot.setLabel("left", "Power", units="%")
        self.power_plot.setLabel("bottom", "Time", units="s")
        self.power_plot.showGrid(x=True, y=True, alpha=0.3)
        self.power_plot.addLine(y=self.reactor.HIGH_POWER_TRIP * 100,
                                pen=pg.mkPen("r", style=Qt.DashLine))   # trip setpoint
        self.power_curve = self.power_plot.plot(pen=pg.mkPen("#3498db", width=2))

        self.temp_plot = pg.PlotWidget(title="Temperatures")
        self.temp_plot.setLabel("left", "Temperature", units="K")
        self.temp_plot.setLabel("bottom", "Time", units="s")
        self.temp_plot.showGrid(x=True, y=True, alpha=0.3)
        self.temp_plot.addLegend()
        self.fuel_curve = self.temp_plot.plot(pen=pg.mkPen("#e67e22", width=2), name="Fuel")
        self.mod_curve = self.temp_plot.plot(pen=pg.mkPen("#16a085", width=2), name="Coolant")

        self.rho_plot = pg.PlotWidget(title="Reactivity")
        self.rho_plot.setLabel("left", "Reactivity", units="$")
        self.rho_plot.setLabel("bottom", "Time", units="s")
        self.rho_plot.showGrid(x=True, y=True, alpha=0.3)
        self.rho_curve = self.rho_plot.plot(pen=pg.mkPen("#8e44ad", width=2))

        layout.addWidget(self.power_plot, 2)
        layout.addWidget(self.temp_plot, 2)
        layout.addWidget(self.rho_plot, 1)
        return layout

    # ------------------------------------------------------------------ simulation loop
    def _init_history(self):
        maxlen = int(HISTORY_S * 1000 / FRAME_MS) * 2
        self.hist = {k: deque(maxlen=maxlen) for k in ("t", "power", "fuel", "mod", "rho")}

    def tick(self):
        if self.running:
            self.reactor.step(FRAME_MS / 1000.0 * self.speed)
            self._record()
        self._refresh()

    def _record(self):
        r = self.reactor
        self.hist["t"].append(r.t)
        self.hist["power"].append(r.n * 100)
        self.hist["fuel"].append(r.T_fuel)
        self.hist["mod"].append(r.T_mod)
        self.hist["rho"].append(r.reactivity() / BETA)
        while self.hist["t"] and self.hist["t"][0] < r.t - HISTORY_S:   # rolling window
            for d in self.hist.values():
                d.popleft()

    def _refresh(self):
        r = self.reactor
        t = np.fromiter(self.hist["t"], float)
        self.power_curve.setData(t, np.fromiter(self.hist["power"], float))
        self.fuel_curve.setData(t, np.fromiter(self.hist["fuel"], float))
        self.mod_curve.setData(t, np.fromiter(self.hist["mod"], float))
        self.rho_curve.setData(t, np.fromiter(self.hist["rho"], float))

        period = r.period()
        self.readouts["Power"].setText(f"{r.power_mw:8.1f} MW  ({r.n*100:6.2f} %)")
        self.readouts["Fuel temp"].setText(f"{r.T_fuel:8.1f} K")
        self.readouts["Coolant temp"].setText(f"{r.T_mod:8.1f} K")
        self.readouts["Reactivity"].setText(f"{r.reactivity()/BETA:+8.4f} $")
        self.readouts["Period"].setText("     inf" if not np.isfinite(period) or abs(period) > 1e4
                                        else f"{period:8.1f} s")
        self.readouts["Sim time"].setText(f"{r.t:8.1f} s")
        self.rod_target_lbl.setText(f"Target:   {r.rod_target:5.1f} %")
        self.rod_pos_lbl.setText(f"Actual:   {r.rod:5.1f} %")

        if r.scrammed:
            self.status_lbl.setText(f"TRIPPED: {r.trip_reason}")
            self.status_lbl.setStyleSheet("background:#c0392b; color:white; font-weight:bold; padding:6px;")
        else:
            self.status_lbl.setText("Normal operation")
            self.status_lbl.setStyleSheet("background:#27ae60; color:white; font-weight:bold; padding:6px;")

    # ------------------------------------------------------------------ buttons
    def toggle_pause(self):
        self.running = not self.running
        self.pause_btn.setText("Resume" if not self.running else "Pause")

    def reset(self):
        self.reactor.reset()
        self._init_history()
        for widget, value in ((self.rod_slider, int(self.reactor.ROD_CRIT * 10)),
                              (self.inlet_spin, self.reactor.T_inlet),
                              (self.flow_spin, self.reactor.flow)):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ReactorWindow()
    win.show()
    sys.exit(app.exec())
