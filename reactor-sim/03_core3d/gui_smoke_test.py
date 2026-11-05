"""
gui_smoke_test.py -- exercise core_viewer.py's logic without a display.

Qt and PyVista are replaced by small stand-in classes (sliders, combo boxes, buttons,
signals); matplotlib renders for real. Every button, slider, display field, pick, scram,
reset and count path runs against the real simulation engine in static, live and startup
modes. Useful where PySide6/PyVista can't be installed (e.g. a headless cloud sandbox).

    python3 gui_smoke_test.py        (~5 min, 4 cm mesh)
"""
import os
import sys
import time
import types
from unittest.mock import MagicMock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)


class Signal:
    def __init__(self):
        self.slots = []

    def connect(self, f):
        self.slots.append(f)

    def emit(self, *a):
        for f in self.slots:
            f(*a)


class W:
    """Generic widget: unknown methods are no-ops."""
    def __init__(self, *a, **k):
        self._blocked = False
        self._enabled = True
        self._text = ""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return MagicMock()

    def closeEvent(self, event):
        pass

    def blockSignals(self, b):
        self._blocked = b

    def setEnabled(self, b):
        self._enabled = b

    def isEnabled(self):
        return self._enabled

    def setText(self, t):
        self._text = t

    def text(self):
        return self._text

    def _emit(self, sig, *a):
        if not self._blocked:
            sig.emit(*a)


class QSlider(W):
    def __init__(self, *a, **k):
        super().__init__()
        self.valueChanged = Signal()
        self._v, self._lo, self._hi = 0, 0, 99

    def setRange(self, lo, hi):
        self._lo, self._hi = lo, hi

    def value(self):
        return self._v

    def setValue(self, v):
        v = int(min(max(v, self._lo), self._hi))
        if v != self._v:
            self._v = v
            self._emit(self.valueChanged, v)


class QComboBox(W):
    def __init__(self, *a, **k):
        super().__init__()
        self.currentTextChanged = Signal()
        self._items, self._cur = [], ""

    def addItems(self, items):
        self._items += list(items)
        if not self._cur and self._items:
            self._cur = self._items[0]

    def currentText(self):
        return self._cur

    def setCurrentText(self, t):
        assert t in self._items, t
        if t != self._cur:
            self._cur = t
            self._emit(self.currentTextChanged, t)


class QCheckBox(W):
    def __init__(self, *a, **k):
        super().__init__()
        self.toggled = Signal()
        self._c = False

    def isChecked(self):
        return self._c

    def setChecked(self, c):
        if c != self._c:
            self._c = c
            self._emit(self.toggled, c)


class QPushButton(QCheckBox):
    def __init__(self, *a, **k):
        super().__init__()
        self.clicked = Signal()


class QDoubleSpinBox(W):
    def __init__(self, *a, **k):
        super().__init__()
        self.editingFinished = Signal()
        self._v = 0.0

    def value(self):
        return self._v

    def setValue(self, v):
        self._v = float(v)


class QLabel(W):
    def __init__(self, text="", *a, **k):
        super().__init__()
        self._text = text


class QTimer(W):
    def __init__(self, *a, **k):
        super().__init__()
        self.timeout = Signal()


class Canvas(W):
    def __init__(self, fig):
        super().__init__()
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        self._agg = FigureCanvasAgg(fig)
        self.draws = 0

    def draw_idle(self):
        self._agg.draw()            # real rendering: catches plotting errors
        self.draws += 1


qtcore = types.ModuleType("PySide6.QtCore")
qtcore.Qt = MagicMock()
qtcore.QTimer = QTimer
qtw = types.ModuleType("PySide6.QtWidgets")
for n in ("QApplication", "QFormLayout", "QGroupBox", "QHBoxLayout", "QMainWindow", "QVBoxLayout", "QWidget"):
    setattr(qtw, n, type(n, (W,), {}))
qtw.QCheckBox, qtw.QComboBox, qtw.QDoubleSpinBox = QCheckBox, QComboBox, QDoubleSpinBox
qtw.QLabel, qtw.QPushButton, qtw.QSlider = QLabel, QPushButton, QSlider
pyside = types.ModuleType("PySide6")
sys.modules.update({"PySide6": pyside, "PySide6.QtCore": qtcore, "PySide6.QtWidgets": qtw,
                    "pyvista": MagicMock(), "pyvistaqt": types.SimpleNamespace(QtInteractor=lambda *a: MagicMock())})
qtagg = types.ModuleType("matplotlib.backends.backend_qtagg")
qtagg.FigureCanvasQTAgg = Canvas
sys.modules["matplotlib.backends.backend_qtagg"] = qtagg

import numpy as np  # noqa: E402
import core_viewer  # noqa: E402


def pump(win, seconds, trends_every=1.0):
    t0 = last = time.monotonic()
    while time.monotonic() - t0 < seconds:
        win._tick()
        if time.monotonic() - last > trends_every:
            win._plot_trends()
            last = time.monotonic()
        time.sleep(0.15)


def wait(cond, timeout, win=None):
    t0 = time.monotonic()
    while not cond():
        if win is not None:
            win._tick()
        if time.monotonic() - t0 > timeout:
            raise TimeoutError
        time.sleep(0.1)


# ------------------------------------------------------------------ live mode
win = core_viewer.LiveViewer("mitr2_like.yaml", cell_size=4.0, power_w=1e6)
wait(lambda: win.live.initialized, 300, win)
win._tick()
print("live: initialized;", win.status_lbl.text(), "| power", win.read["Power"].text())
assert win.snap is not None and win.read["Power"].text() == "1 MW"
win.run_btn.setChecked(True)
win.speed_box.setCurrentText("10x")
win._nudge(+0.05)
win._nudge(+0.05)
win._send_blade_target()                 # what the debounce timer would do
pump(win, 10)
s = win.snap
print(f"live: t={s.t:.1f}s  {win.read['Power'].text()}  period {win.read['Period'].text()}  "
      f"blades {s.withdrawn:.2f}/{s.target:.2f}  status '{win.status_lbl.text()}'")
assert abs(s.target - (win.live.h0 + 0.1)) < 1e-9 and s.power_w > 1e6 and win.tcanvas.draws > 0
for f in core_viewer.LIVE_FIELDS:
    win.field_box.setCurrentText(f)
win.checks["Labels"].setChecked(True)
win.checks["Reflector"].setChecked(False)
win._on_pick((5.0, 3.0, 20.0))
win.inlet_spin.setValue(37.0)
win.inlet_spin.editingFinished.emit()
pump(win, 4)
print(f"live: selected {win.labels[win.selected]}, inlet now {win.snap.inlet_c:.1f} C, "
      f"axial canvas draws {win.canvas.draws}")
assert abs(win.snap.inlet_c - 37.0) < 1e-9 and win.canvas.draws > 0
win.scram_btn.clicked.emit(False)            # Qt passes 'checked'
wait(lambda: win.snap is not None and win.snap.scrammed, 60, win)
pump(win, 2)
print("live: after scram:", win.status_lbl.text()[:60], "| slider enabled:", win.slider.isEnabled())
assert "SCRAMMED" in win.status_lbl.text() and not win.slider.isEnabled()
win.reset_btn.clicked.emit(False)
wait(lambda: win.live.initialized and win.live.snapshot().t == 0.0, 300, win)
pump(win, 2)
assert win.live.snapshot().t == 0.0 and win.live.status == "paused", "reset must pause"
print("live: after reset:", win.read["Sim time"].text().strip(), win.read["Power"].text(),
      "| slider", win.slider.value() / core_viewer.SLIDER_SCALE, "cm | enabled", win.slider.isEnabled())
assert win.slider.isEnabled() and win.live.error is None
win.closeEvent(MagicMock())

# ------------------------------------------------------------------ live, no feedback
win = core_viewer.LiveViewer("mitr2_like.yaml", cell_size=4.0, power_w=100.0, feedback=False)
wait(lambda: win.live.initialized, 300, win)
win.run_btn.setChecked(True)
pump(win, 3)
print("live (no feedback):", win.read["Power"].text(), win.read["Hottest plate"].text())
win.field_box.setCurrentText("Hottest plate temperature")     # all NaN -> outline, no crash
win._redraw_state()
win.closeEvent(MagicMock())

# ------------------------------------------------------------------ startup mode
win = core_viewer.LiveViewer("mitr2_like.yaml", cell_size=4.0, startup=True)
wait(lambda: win.live.initialized, 300, win)
pump(win, 1)
print("startup: power", win.read["Power"].text(), "|", win.count_lbl.text())
print("         " + " | ".join(f"{k}: {v.text()}" for k, v in win.inst_lbl.items()))
assert "cps" in win.inst_lbl["Ch 1"].text() and win.live.P0 < 1.0
win.speed_box.setCurrentText("100x")
win.run_btn.setChecked(True)
win.live.count(30)
wait(lambda: win.snap is not None and len(win.snap.one_over_m) == 1, 120, win)
win.slider.setValue(int(round(8.0 * core_viewer.SLIDER_SCALE)))
win._send_blade_target()
wait(lambda: win.snap.withdrawn >= 8.0 - 1e-9, 300, win)
pump(win, 3)
win.live.count(30)
wait(lambda: win.snap is not None and len(win.snap.one_over_m) == 2, 300, win)
pump(win, 1.5)
print("startup: after two counts:", win.count_lbl.text())
assert "1/M" in win.count_lbl.text() and win.tcanvas.draws > 0
win.closeEvent(MagicMock())

# ------------------------------------------------------------------ static mode
win = core_viewer.CoreViewer("mitr2_like.yaml", cell_size=4.0)
wait(lambda: (win._poll(), win.snap is not None)[1], 300)
print("static: k =", win.read["k-eff"].text(), "| status", win.read["Status"].text())
win.slider.setValue(300)
win._request_current()
wait(lambda: (win._poll(), abs(win.snap.withdrawn - 30.0) < 1e-6)[1], 300)
win.crit_btn.clicked.emit(False)
wait(lambda: (win._poll(), not win.ctrl.busy and abs(win.snap.k - 1) < 1e-4)[1], 600)
print("static: critical search ->", win.read["Solved at"].text(), "k =", win.read["k-eff"].text())
for f in core_viewer.FIELDS:
    win.field_box.setCurrentText(f)
win.closeEvent(MagicMock())
print("\nGUI logic smoke test passed.")
