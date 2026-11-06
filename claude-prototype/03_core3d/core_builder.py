"""
core_builder.py -- turns a YAML reactor configuration into a diffusion3d.Problem.

The geometry is described analytically (hexagonal lattice of rhombic elements, cylinders,
blade slabs) and rasterized onto the Cartesian mesh: each cell is subsampled, the material
at every sample point is looked up, and cross sections are volume-weighted (Sigmas
arithmetically, D through the transport cross section 1/D). Cells that straddle an
interface therefore get a properly mixed material instead of a jagged staircase.

It also keeps a cell->element weight matrix so per-element powers can be tallied.
"""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import yaml

from diffusion3d import FACES, XS_NAMES, Mesh, Problem

RING_LETTERS = "ABCDEFGH"


# ----------------------------------------------------------------------------- lattice
class HexRhombusLattice:
    """Hexagon of side n*s split into three sectors (spanned by unit vectors at 90, 210 and
    330 degrees); each sector is an n x n grid of 60/120-degree rhombi. Ring index of
    rhombus (i, j) is max(i, j), giving 3, 9, 15, ... elements per ring (n=3 -> 27, MITR)."""

    def __init__(self, side, n_rings=3):
        self.s, self.n = side, n_rings
        ang = np.radians([90.0, 210.0, 330.0])
        self.e = np.stack([np.cos(ang), np.sin(ang)], axis=1)            # unit vectors
        # inverse basis for each sector k: p = u*e_k + v*e_(k+1)
        self.inv = [np.linalg.inv(np.column_stack([self.e[k], self.e[(k + 1) % 3]]))
                    for k in range(3)]
        elems = []
        for k in range(3):
            a, b = self.e[k] * side, self.e[(k + 1) % 3] * side
            for i in range(n_rings):
                for j in range(n_rings):
                    o = i * a + j * b
                    verts = np.array([o, o + a, o + a + b, o + b])
                    elems.append(dict(sector=k, i=i, j=j, ring=max(i, j),
                                      center=verts.mean(0), verts=verts))
        # label within each ring by angle, counter-clockwise from +x
        for ring in range(n_rings):
            members = [e for e in elems if e["ring"] == ring]
            members.sort(key=lambda e: np.arctan2(e["center"][1], e["center"][0]) % (2 * np.pi))
            for rank, e in enumerate(members, 1):
                e["label"] = f"{RING_LETTERS[ring]}-{rank}"
        self.elements = elems
        self.index = {(e["sector"], e["i"], e["j"]): idx for idx, e in enumerate(elems)}
        self.flat_distance = n_rings * side * np.cos(np.radians(30))
        self.hex_side = n_rings * side

    def locate(self, x, y):
        """Element index for each point, -1 if outside the hexagon."""
        out = np.full(x.shape, -1, dtype=np.int32)
        pts = np.stack([x, y])
        lim = self.n * self.s
        for k in range(3):
            u, v = np.tensordot(self.inv[k], pts, axes=1)
            inside = (u >= 0) & (u < lim) & (v >= 0) & (v < lim) & (out < 0)
            i = np.floor(u[inside] / self.s).astype(int)
            j = np.floor(v[inside] / self.s).astype(int)
            out[inside] = k * self.n**2 + i * self.n + j      # matches construction order
        return out


# ----------------------------------------------------------------------------- model
@dataclass
class CoreModel:
    config: dict
    problem: Problem
    lattice: HexRhombusLattice
    element_materials: list            # material name per element
    W: sp.csr_matrix                   # (n_cells x n_elements) share of cell fission -> element
    vol_frac: dict                     # material name -> volume fraction array
    mix: dict = None                   # cached data for re-mixing blades (see xs_at)

    def xs_at(self, withdrawn):
        """Cross sections with the shim bank at `withdrawn` cm. Cheap (no re-rasterizing):
        only the blade volume fractions change, so this is safe to call every time step."""
        m = self.mix
        frac = m["frac_base"].copy()
        bf = _blade_fraction(m["blade_xy"], m["z_lo"], self.problem.mesh.dz, m["z_fuel_lo"],
                             m["travel"], m["cusp_q"], withdrawn)
        frac[..., m["blade_id"]] += bf
        frac[..., m["fill_id"]] -= bf
        if frac.min() < -1e-12:
            raise ValueError("blade overlaps something other than tank water")
        return _mix_xs(frac, m["mat_vals"])

    def problem_at(self, withdrawn):
        return Problem(self.problem.mesh, self.xs_at(withdrawn), self.problem.bc)

    @property
    def labels(self):
        return [e["label"] for e in self.lattice.elements]

    def element_powers(self, result):
        """Relative power per element (mean over fueled elements = 1)."""
        xs = self.problem.xs
        cell_fission = (xs["nSf1"] * result.phi1 + xs["nSf2"] * result.phi2).ravel()
        p = self.W.T @ cell_fission
        fueled = p > 0
        return p / p[fueled].mean()

    def element_axial_profile(self, result, elem_index):
        xs = self.problem.xs
        cell_fission = (xs["nSf1"] * result.phi1 + xs["nSf2"] * result.phi2).ravel()
        contrib = (self.W[:, elem_index].toarray().ravel() * cell_fission).reshape(self.problem.mesh.shape)
        return contrib.sum(axis=(0, 1))


def _blade_fraction(blade_xy, z_lo, dz, z_fuel_lo, travel, cusp_q, withdrawn):
    """Blade volume fraction per cell. Blade tips are placed analytically in z so k varies
    smoothly with bank position (sampling the tip would make k a staircase)."""
    z_tip = z_fuel_lo + float(np.clip(withdrawn, 0.0, travel))
    z_frac = np.clip((z_lo + dz - np.maximum(z_lo, z_tip)) / dz, 0.0, 1.0)
    # Rod-cusping correction: in a partially rodded cell the flux in the rodded part is
    # depressed, so plain volume weighting overstates the absorber. Weight the rodded
    # fraction by an assumed rodded/unrodded flux ratio (flux-volume weighting).
    z_frac = cusp_q * z_frac / (cusp_q * z_frac + (1.0 - z_frac))
    return blade_xy[:, :, None] * z_frac[None, None, :]


def _mix_xs(frac, mat_vals):
    xs = {}
    for name in XS_NAMES:
        vals = mat_vals[name]
        if name in ("D1", "D2"):
            xs[name] = 1.0 / (frac @ (1.0 / vals))         # average the transport xs
        else:
            xs[name] = frac @ vals
    return xs


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build(config, blade_withdrawn=None):
    """Rasterize the configuration. blade_withdrawn overrides the shim bank position [cm]."""
    mc = config["mesh"]
    h = float(mc["cell_size"])
    hw = float(mc["half_width"])
    nxy = int(round(2 * hw / h))
    nz = int(round((mc["z_max"] - mc["z_min"]) / h))
    mesh = Mesh(nxy, nxy, nz, h, h, h, -hw, -hw, float(mc["z_min"]))
    ns = int(mc.get("subsamples", 3))

    mats = list(config["materials"])
    mid = {name: i for i, name in enumerate(mats)}

    core = config["core"]
    lat = HexRhombusLattice(float(core["element_side"]), 3)
    labels = {e["label"]: i for i, e in enumerate(lat.elements)}
    elem_mat = [core["default_element"]] * len(lat.elements)
    for label, m in (core.get("positions") or {}).items():
        if label not in labels:
            raise ValueError(f"unknown core position {label}; valid: {sorted(labels)}")
        elem_mat[labels[label]] = m
    elem_mat_id = np.array([mid[m] for m in elem_mat])

    # ---- sample points: ns^3 per cell
    off = (np.arange(ns) + 0.5) / ns
    xs_ = (mesh.x0 + h * (np.arange(nxy)[:, None] + off[None, :])).ravel()     # nxy*ns
    zs_ = (mesh.z0 + h * (np.arange(nz)[:, None] + off[None, :])).ravel()      # nz*ns

    # radial (x,y) lookups done once on the 2D sample grid
    X, Y = np.meshgrid(xs_, xs_, indexing="ij")
    R = np.hypot(X, Y)
    elem2d = lat.locate(X, Y)
    sur = config["surroundings"]
    radial = np.full(X.shape, mid[sur["beyond_reflector"]], dtype=np.int32)
    radial[R < sur["reflector"]["outer_radius"]] = mid[sur["reflector"]["material"]]
    radial[R < sur["tank_radius"]] = mid[sur["core_fill"]]

    # shim blades: slabs just outside each hexagon flat
    bank = config["control"]["shim_bank"]
    withdrawn = bank["withdrawn"] if blade_withdrawn is None else blade_withdrawn
    withdrawn = float(np.clip(withdrawn, 0.0, bank["travel"]))
    blade2d = np.zeros(X.shape, bool)
    for side in bank["sides"]:
        ang = np.radians(60.0 * side)
        t = X * np.cos(ang) + Y * np.sin(ang)
        tang = -X * np.sin(ang) + Y * np.cos(ang)
        t0 = lat.flat_distance + bank["gap"]
        blade2d |= (t >= t0) & (t < t0 + bank["thickness"]) & (np.abs(tang) <= lat.hex_side / 2)

    z_fuel_lo = float(core["fuel_bottom"])
    z_fuel_hi = z_fuel_lo + float(core["fuel_height"])

    # ---- accumulate material counts layer by layer (keeps memory small)
    nmat = len(mats)
    n_el = len(lat.elements)
    counts = np.zeros((nxy, nxy, nz, nmat))
    W_rows, W_cols, W_vals = [], [], []
    cell_xy = (np.arange(nxy * ns) // ns)
    CI, CJ = np.meshgrid(cell_xy, cell_xy, indexing="ij")
    flat_xy = (CI * nxy + CJ).ravel()           # 2D cell index per sample
    in_hex = elem2d >= 0

    for kz in range(nz):
        layer_counts = np.zeros((nxy * nxy, nmat))
        layer_fuel = np.zeros((nxy * nxy, n_el))
        for z in zs_[kz * ns:(kz + 1) * ns]:
            m = radial.copy()
            if z_fuel_lo <= z < z_fuel_hi:
                m[in_hex] = elem_mat_id[elem2d[in_hex]]
            else:
                m[in_hex] = mid[sur["below_core"] if z < z_fuel_lo else sur["above_core"]]
            np.add.at(layer_counts, (flat_xy, m.ravel()), 1.0)
            if z_fuel_lo <= z < z_fuel_hi:
                fissile_pts = in_hex & (np.isin(m, [mid[n] for n in mats
                                                    if config["materials"][n].get("nSf2", 0) > 0]))
                np.add.at(layer_fuel, (flat_xy[fissile_pts.ravel()], elem2d[fissile_pts]), 1.0)
        counts[:, :, kz, :] = layer_counts.reshape(nxy, nxy, nmat)
        # cell -> element share of that cell's fission
        tot = layer_fuel.sum(axis=1)
        nz_cells = np.nonzero(tot)[0]
        for c in nz_cells:
            for e in np.nonzero(layer_fuel[c])[0]:
                ci, cj = divmod(c, nxy)
                W_rows.append((ci * nxy + cj) * nz + kz)
                W_cols.append(e)
                W_vals.append(layer_fuel[c, e] / tot[c])

    frac_base = counts / ns**3                    # everything except the blades
    blade_xy = blade2d.reshape(nxy, ns, nxy, ns).mean(axis=(1, 3))          # xy volume fraction
    mat_vals = {name: np.array([config["materials"][m].get(name, 0.0) for m in mats], float)
                for name in XS_NAMES}
    mix = dict(frac_base=frac_base, blade_xy=blade_xy, z_lo=mesh.z0 + h * np.arange(nz),
               z_fuel_lo=z_fuel_lo, travel=float(bank["travel"]),
               cusp_q=float(bank.get("cusping_flux_ratio", 1.0)),
               blade_id=mid[bank["material"]], fill_id=mid[sur["core_fill"]], mat_vals=mat_vals)
    frac = frac_base.copy()
    bf = _blade_fraction(blade_xy, mix["z_lo"], h, z_fuel_lo, mix["travel"], mix["cusp_q"], withdrawn)
    frac[..., mix["blade_id"]] += bf
    frac[..., mix["fill_id"]] -= bf
    if frac.min() < -1e-12:
        raise ValueError("blade overlaps something other than tank water")
    xs = _mix_xs(frac, mat_vals)

    b = config["boundary"]
    bc = {f: (b["type"], b["gamma"]) if b["type"] == "albedo" else (b["type"],) for f in FACES}
    W = sp.csr_matrix((W_vals, (W_rows, W_cols)), shape=(mesh.n_cells, n_el))
    vol_frac = {m: frac[..., i] for i, m in enumerate(mats)}
    return CoreModel(config, Problem(mesh, xs, bc), lat, elem_mat, W, vol_frac, mix)
