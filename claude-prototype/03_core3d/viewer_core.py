"""
viewer_core.py -- everything the 3D viewer needs that is NOT drawing:

  * plain-numpy geometry for the scene (element prisms, blade slabs)
  * SimController: runs diffusion solves in a background thread, caches results by blade
    height, warm-starts from the nearest cached solution, and can search for criticality.

Keeping this separate from the GUI means it can be tested without a display, and the GUI
(core_viewer.py) stays a thin layer that only draws what this module computes.
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from core_builder import build
from diffusion3d import solve_k


# ----------------------------------------------------------------------------- geometry
def element_prisms(lattice, z0, height, shrink=0.94):
    """Points and VTK-style face array for every element as a vertical rhombic prism.

    Returns (points (N,3), faces (flat int array: [4,i,j,k,l, 4,...]), face_elem (M,))
    where face_elem gives the element index of each face (6 faces per element)."""
    pts, faces, face_elem = [], [], []
    for idx, e in enumerate(lattice.elements):
        c = e["center"]
        v = c + shrink * (e["verts"] - c)                  # shrink to show gaps
        base = len(pts)
        for z in (z0, z0 + height):
            pts.extend([[x, y, z] for x, y in v])
        b, t = [base + i for i in range(4)], [base + 4 + i for i in range(4)]
        quads = [b[::-1], t] + [[b[i], b[(i + 1) % 4], t[(i + 1) % 4], t[i]] for i in range(4)]
        for q in quads:
            faces.extend([4, *q])
            face_elem.append(idx)
    return np.array(pts, float), np.array(faces, np.int64), np.array(face_elem)


def blade_boxes(cfg, lattice, withdrawn, z_top):
    """Points and face array for the shim blades: one slab per hexagon side,
    from the blade tip (fuel bottom + withdrawn) up to z_top."""
    bank = cfg["control"]["shim_bank"]
    z_tip = cfg["core"]["fuel_bottom"] + float(np.clip(withdrawn, 0, bank["travel"]))
    t0 = lattice.flat_distance + bank["gap"]
    t1 = t0 + bank["thickness"]
    half = lattice.hex_side / 2
    pts, faces = [], []
    for side in bank["sides"]:
        a = np.radians(60.0 * side)
        n, tg = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
        corners = [t0 * n - half * tg, t1 * n - half * tg, t1 * n + half * tg, t0 * n + half * tg]
        base = len(pts)
        for z in (z_tip, z_top):
            pts.extend([[x, y, z] for x, y in corners])
        b, t = [base + i for i in range(4)], [base + 4 + i for i in range(4)]
        for q in [b[::-1], t] + [[b[i], b[(i + 1) % 4], t[(i + 1) % 4], t[i]] for i in range(4)]:
            faces.extend([4, *q])
    return np.array(pts, float), np.array(faces, np.int64)


# ----------------------------------------------------------------------------- solving
@dataclass
class Snapshot:
    """One solved state, ready to draw."""
    withdrawn: float
    k: float
    element_power: np.ndarray        # per element, fueled mean = 1 (0 for dummies)
    power_density: np.ndarray        # per cell, NaN outside fuel
    phi1: np.ndarray                 # fast flux, normalized so max(phi1) = 1
    phi2: np.ndarray                 # thermal flux, same normalization
    model: object                    # CoreModel (for axial profiles)
    result: object
    solve_seconds: float

    @property
    def rho_pcm(self):
        return (self.k - 1.0) / self.k * 1e5


class SimController:
    """Background solver with a cache. The GUI calls request(h) as often as it likes; only
    the most recent request is solved once the worker is free, so dragging a slider never
    queues up a backlog of stale solves."""

    def __init__(self, cfg, cache_step=0.1):
        self.cfg = cfg
        self.step = cache_step
        self.cache = {}
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._lock = threading.Lock()
        self._future = None
        self._pending = None
        self.latest = None               # most recently completed Snapshot
        self.busy_with = None            # height currently being solved
        self.message = ""

    def key(self, h):
        return round(round(float(h) / self.step) * self.step, 6)

    # ---- public API (call from the GUI thread) ----
    def request(self, h):
        """Ask for the state at blade height h. Returns a cached Snapshot immediately if
        available, else None (and starts/queues a background solve)."""
        k = self.key(h)
        if k in self.cache:
            self.latest = self.cache[k]
            return self.latest
        with self._lock:
            if self._future is None or self._future.done():
                self._start(k)
            else:
                self._pending = k
        return None

    def poll(self):
        """Call periodically (e.g. from a QTimer). Returns a new Snapshot when one finishes."""
        with self._lock:
            f = self._future
            if f is None or not f.done():
                return None
            self._future = None
            self.busy_with = None
            snap = f.result()            # re-raises solver errors in the GUI thread
            if self._pending is not None:
                nxt, self._pending = self._pending, None
                if nxt not in self.cache:
                    self._start(nxt)
        self.latest = snap
        return snap

    def find_critical(self, h_start, tol=2e-5):
        """Background secant search on k(h) = 1. Result arrives through poll()."""
        with self._lock:
            if self._future is not None and not self._future.done():
                return False
            self._pending = None
            self.busy_with = "critical search"
            self._future = self._pool.submit(self._critical_search, float(h_start), tol)
        return True

    @property
    def busy(self):
        return self._future is not None and not self._future.done()

    # ---- worker side ----
    def _start(self, k):
        self.busy_with = k
        self._future = self._pool.submit(self._solve_cached, k)

    def _nearest_guess(self, h):
        if not self.cache:
            return None
        k = min(self.cache, key=lambda c: abs(c - h))
        return self.cache[k].result

    def _solve(self, h, tol_k=1e-6, tol_src=1e-5):
        t = time.time()
        model = build(self.cfg, blade_withdrawn=h)
        r = solve_k(model.problem, tol_k=tol_k, tol_src=tol_src, guess=self._nearest_guess(h))
        P = model.element_powers(r)
        fuel = (model.problem.xs["nSf2"] > 0)
        pd = np.where(fuel, r.power, np.nan)
        norm = r.phi1.max()
        return Snapshot(h, r.k, P, pd, r.phi1 / norm, r.phi2 / norm, model, r, time.time() - t)

    def _solve_cached(self, k):
        if k not in self.cache:
            self.cache[k] = self._solve(k)
        return self.cache[k]

    def _critical_search(self, h_start, tol):
        travel = float(self.cfg["control"]["shim_bank"]["travel"])
        a = self._solve_cached(self.key(np.clip(h_start, 0, travel)))
        b = self._solve_cached(self.key(np.clip(h_start + (3.0 if a.k < 1 else -3.0), 0, travel)))
        snap = b
        for _ in range(10):
            if abs(snap.k - 1) < tol or a.k == b.k:
                break
            h = b.withdrawn + (1 - b.k) * (b.withdrawn - a.withdrawn) / (b.k - a.k)
            h = float(np.clip(h, 0, travel))
            # exact (uncached) solve: the critical height needn't sit on the cache grid
            snap = self._solve(h, tol_k=1e-7, tol_src=1e-6)
            a, b = b, snap
            if h in (0.0, travel) and abs(snap.k - 1) > tol:
                self.message = "bank cannot reach criticality"
                break
        self.cache[self.key(snap.withdrawn)] = snap
        return snap

    def shutdown(self):
        self._pool.shutdown(wait=False, cancel_futures=True)
