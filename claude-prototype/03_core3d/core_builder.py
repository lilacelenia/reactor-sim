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

    def __init__(self, side, n_rings=3, inner_rotated=False, numbering=None):
        """inner_rotated: tile the innermost ring (the hexagon of side s at the centre) with
        its three rhombi turned 60 degrees from the sector tiling, as in MITR (Fig. 2-7 of
        ANL/RERTR/TM-10-41: A1 points at the regulating-rod corner, while the B/C sectors
        split there). numbering: None = per ring counter-clockwise from +x (this program's
        convention); a number = per ring clockwise starting at that angle [deg], which is
        the MITR convention when the angle is the regulating-rod corner."""
        self.s, self.n = side, n_rings
        self.inner_rotated = bool(inner_rotated)
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
                    if self.inner_rotated and i == 0 and j == 0:
                        verts = verts @ _rot(60.0).T
                    elems.append(dict(sector=k, i=i, j=j, ring=max(i, j),
                                      center=verts.mean(0), verts=verts))
        # label within each ring by angle
        if numbering is None:                               # counter-clockwise from +x
            def key(e):
                return np.arctan2(e["center"][1], e["center"][0]) % (2 * np.pi)
        else:                                               # clockwise from `numbering`
            a0 = np.radians(float(numbering))

            def key(e):
                a = (a0 - np.arctan2(e["center"][1], e["center"][0])) % (2 * np.pi)
                return 0.0 if a > 2 * np.pi - 1e-9 else a  # an element on the start ray is 1st
        for ring in range(n_rings):
            members = [e for e in elems if e["ring"] == ring]
            members.sort(key=key)
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
        if self.inner_rotated:
            # the centre hexagon is the same region; re-split it with the rotated tiling:
            # p lies in rotated rhombus k exactly when R(-60) p lies in the unrotated one
            centre = np.isin(out, [k * self.n**2 for k in range(3)])
            if centre.any():
                q = _rot(-60.0) @ np.stack([x[centre], y[centre]])
                for k in range(3):
                    u, v = self.inv[k] @ q
                    hit = (u >= 0) & (u < self.s) & (v >= 0) & (v < self.s)
                    sub = out[centre]
                    sub[hit] = k * self.n**2
                    out[centre] = sub
        return out


def _rot(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


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

    def xs_at(self, withdrawn, reg=None):
        """Cross sections with the shim bank at `withdrawn` cm and the regulating rod (if the
        configuration has one) at `reg` cm (None = its configured position). `withdrawn` may
        also be a sequence with one height per blade (in the order of shim_bank.sides) to
        move blades individually. Cheap (no re-rasterizing): only the absorber volume
        fractions change, so this is safe to call every time step."""
        return _mix_xs(_absorber_fractions(self.mix, self.problem.mesh.dz, withdrawn, reg),
                       self.mix["mat_vals"])

    def problem_at(self, withdrawn, reg=None):
        return Problem(self.problem.mesh, self.xs_at(withdrawn, reg), self.problem.bc)

    @property
    def has_reg_rod(self):
        return self.mix.get("reg") is not None

    @property
    def reg_default(self):
        """Configured regulating rod position [cm] (None without a regulating rod)."""
        r = self.mix.get("reg")
        return None if r is None else r["default"]

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


def blade_tip_zero(config):
    """z [cm] of the shim-blade tips when fully inserted (withdrawn = 0): the fuel bottom
    plus shim_bank.inserted_tip_above_fuel_bottom (0 if absent)."""
    return (float(config["core"]["fuel_bottom"])
            + float(config["control"]["shim_bank"].get("inserted_tip_above_fuel_bottom", 0.0)))


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


def _absorber_fractions(mix, dz, withdrawn, reg=None):
    """Material volume fractions with the shim bank and the regulating rod inserted."""
    m = mix
    frac = m["frac_base"].copy()
    if np.ndim(withdrawn) == 0:                       # whole bank at one height
        bf = _blade_fraction(m["blade_xy"], m["z_lo"], dz, m["z_blade0"], m["travel"],
                             m["cusp_q"], withdrawn)
    else:                                             # one height per blade
        hs = np.asarray(withdrawn, float)
        if hs.shape != (len(m["blade_xy_sides"]),):
            raise ValueError(f"need one height per blade ({len(m['blade_xy_sides'])})")
        bf = sum(_blade_fraction(bxy, m["z_lo"], dz, m["z_blade0"], m["travel"], m["cusp_q"], h)
                 for bxy, h in zip(m["blade_xy_sides"], hs))
    frac[..., m["blade_id"]] += bf
    frac[..., m["fill_id"]] -= bf
    r = m.get("reg")
    if r is not None:
        rf = _blade_fraction(r["xy"], m["z_lo"], dz, m["z_fuel_lo"], r["travel"], m["cusp_q"],
                             r["default"] if reg is None else reg)
        frac[..., r["mat_id"]] += rf
        frac[..., m["fill_id"]] -= rf
    if frac.min() < -1e-12:
        raise ValueError("an absorber overlaps something other than tank water")
    return frac


def regulating_rod_xy(config):
    """(x, y) centre [cm] and radius of the regulating rod, or None. It stands just outside
    the core hexagon on the bisector of a corner (`corner_angle`, degrees from +x)."""
    r = config["control"].get("regulating_rod")
    if not r:
        return None
    ang = np.radians(float(r["corner_angle"]))
    d = float(r["radial_position"])
    return np.array([d * np.cos(ang), d * np.sin(ang)]), float(r["radius"])


def _blade_mask(X, Y, lat, bank):
    m = np.zeros(X.shape, bool)
    for side in bank["sides"]:
        ang = np.radians(60.0 * side)
        t = X * np.cos(ang) + Y * np.sin(ang)
        tang = -X * np.sin(ang) + Y * np.cos(ang)
        t0 = lat.flat_distance + bank["gap"]
        m |= (t >= t0) & (t < t0 + bank["thickness"]) & (np.abs(tang) <= lat.hex_side / 2)
    return m


def _reg_rod_xy_fraction(config, mesh, lat, nsub=20):
    """Volume fraction of each (x, y) cell covered by the regulating rod (fine subsampling:
    the rod is thin compared with a cell). Checks it touches only tank water."""
    geo = regulating_rod_xy(config)
    c, rad = geo
    tank = float(config["surroundings"]["tank_radius"])
    if np.hypot(*c) + rad > tank:
        raise ValueError("regulating rod must lie inside the tank (core_fill water)")
    off = (np.arange(nsub) + 0.5) / nsub
    xs = (mesh.x0 + mesh.dx * (np.arange(mesh.nx)[:, None] + off[None, :])).ravel()
    ys = (mesh.y0 + mesh.dy * (np.arange(mesh.ny)[:, None] + off[None, :])).ravel()
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    rod = np.hypot(X - c[0], Y - c[1]) < rad
    if np.any(rod & (lat.locate(X, Y) >= 0)):
        raise ValueError("regulating rod overlaps the core hexagon")
    if np.any(rod & _blade_mask(X, Y, lat, config["control"]["shim_bank"])):
        raise ValueError("regulating rod overlaps a shim blade")
    return rod.reshape(mesh.nx, nsub, mesh.ny, nsub).mean(axis=(1, 3))


def _blade_xy_fraction(mesh, lat, bank, nsub=40):
    """Volume fraction of each (x, y) cell covered by each shim blade, one array per side.
    Fine subsampling (2 cm / 40 = 0.05 cm): the blades are thinner than the material
    subsampling pitch, which on some meshes missed whole blades."""
    off = (np.arange(nsub) + 0.5) / nsub
    xs = (mesh.x0 + mesh.dx * (np.arange(mesh.nx)[:, None] + off[None, :])).ravel()
    ys = (mesh.y0 + mesh.dy * (np.arange(mesh.ny)[:, None] + off[None, :])).ravel()
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return [_blade_mask(X, Y, lat, dict(bank, sides=[sd])).reshape(mesh.nx, nsub, mesh.ny, nsub)
            .mean(axis=(1, 3)) for sd in bank["sides"]]


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


def make_lattice(config):
    """Lattice per core.inner_ring_rotated and core.position_numbering ('program': counter-
    clockwise from +x; 'mitr': clockwise from the regulating-rod corner, as the facility)."""
    core = config["core"]
    numbering = None
    if core.get("position_numbering", "program") == "mitr":
        numbering = float(config["control"]["regulating_rod"]["corner_angle"])
    return HexRhombusLattice(float(core["element_side"]), 3,
                             inner_rotated=bool(core.get("inner_ring_rotated", False)),
                             numbering=numbering)


def build(config, blade_withdrawn=None, reg_withdrawn=None):
    """Rasterize the configuration. blade_withdrawn overrides the shim bank position [cm],
    reg_withdrawn the regulating rod's [cm] (both become the model's defaults)."""
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
    lat = make_lattice(config)
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
    withdrawn = np.clip(np.asarray(withdrawn, float), 0.0, bank["travel"])
    if withdrawn.ndim == 0:
        withdrawn = float(withdrawn)

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
    blade_xy_sides = _blade_xy_fraction(mesh, lat, bank)                    # xy volume fractions
    blade_xy = sum(blade_xy_sides)
    if blade_xy.max() > 1 + 1e-12 or np.any(_blade_mask(X, Y, lat, bank) & (elem2d >= 0)):
        raise ValueError("shim blades overlap each other or the core hexagon")
    mat_vals = {name: np.array([config["materials"][m].get(name, 0.0) for m in mats], float)
                for name in XS_NAMES}
    mix = dict(frac_base=frac_base, blade_xy=blade_xy, blade_xy_sides=blade_xy_sides, z_lo=mesh.z0 + h * np.arange(nz),
               z_fuel_lo=z_fuel_lo, z_blade0=blade_tip_zero(config), travel=float(bank["travel"]),
               cusp_q=float(bank.get("cusping_flux_ratio", 1.0)),
               blade_id=mid[bank["material"]], fill_id=mid[sur["core_fill"]], mat_vals=mat_vals,
               reg=None)
    rr = config["control"].get("regulating_rod")
    if rr:
        travel_r = float(rr["travel"])
        h_r = float(rr["withdrawn"]) if reg_withdrawn is None else float(reg_withdrawn)
        mix["reg"] = dict(xy=_reg_rod_xy_fraction(config, mesh, lat), travel=travel_r,
                          default=float(np.clip(h_r, 0.0, travel_r)), mat_id=mid[rr["material"]])
    frac = _absorber_fractions(mix, h, withdrawn)
    xs = _mix_xs(frac, mat_vals)

    b = config["boundary"]
    bc = {f: (b["type"], b["gamma"]) if b["type"] == "albedo" else (b["type"],) for f in FACES}
    W = sp.csr_matrix((W_vals, (W_rows, W_cols)), shape=(mesh.n_cells, n_el))
    vol_frac = {m: frac[..., i] for i, m in enumerate(mats)}
    return CoreModel(config, Problem(mesh, xs, bc), lat, elem_mat, W, vol_frac, mix)
