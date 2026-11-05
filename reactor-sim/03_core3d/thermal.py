"""
thermal.py -- per-channel thermal-hydraulics and temperature feedback on cross sections.

Every fuel element is one coolant channel, split axially into the mesh layers that contain
fuel. Each (element, layer) node has two temperatures:

    T_f  plate (fuel) temperature:  C_f dT_f/dt = q - hA (T_f - T_c)
    T_c  coolant temperature:       C_c dT_c/dt = hA (T_f - T_c) + m cp (T_c,below - T_c)

Coolant flows upward (inlet at the bottom); T_c of a node is its mixed outlet temperature
(donor-cell / upwind scheme), so at steady state T_c rises by q / (m cp) per node.
q is the fission power deposited in that node, from the 3D flux.

Feedback: the cross sections of the feedback materials (the homogenized fuel, which includes
the channel water) change linearly with the node temperatures:
    Sigma_x(T) = Sigma_x0 * (1 + c_x * (T_c - T_ref) + f_x * (T_f - T_ref))
with c_x, f_x the fractional derivatives in the config (D is changed the same way, then mixed
harmonically like the other D's). Cells partly filled with fuel get the change in proportion
to their fuel volume fraction, and a cell shared by two elements gets the fission-weighted
mix of their temperatures.

The module also gives the exact first-order sensitivity of the adjoint-weighted reactivity to
every node temperature, d rho / d T_k, used to put the feedback inside the fast amplitude
integration of the quasi-static method.
"""
import numpy as np
import scipy.sparse as sp

from diffusion3d import FACES, XS_NAMES

CP_WATER = 4180.0          # J/(kg K)


class ChannelThermal:
    def __init__(self, model, cfg):
        th = cfg["thermal_hydraulics"]
        fb = cfg.get("feedback") or {}
        mesh = model.problem.mesh
        self.mesh = mesh
        z0 = float(cfg["core"]["fuel_bottom"])
        H = float(cfg["core"]["fuel_height"])

        # ---- nodes: fueled elements x mesh layers overlapping the fuel
        zl = mesh.z0 + mesh.dz * np.arange(mesh.nz)
        overlap = np.clip(np.minimum(zl + mesh.dz, z0 + H) - np.maximum(zl, z0), 0.0, None)
        self.layers = np.nonzero(overlap > 1e-9)[0]
        hfrac = overlap[self.layers] / H                    # share of the fueled length per layer
        W = model.W.tocoo()
        elem_tot = np.bincount(W.col, weights=W.data, minlength=model.W.shape[1])
        self.elems = np.nonzero(elem_tot > 0)[0]
        ne, nl = len(self.elems), len(self.layers)
        self.ne, self.nl, self.n = ne, nl, ne * nl
        ie_of = np.full(model.W.shape[1], -1)
        ie_of[self.elems] = np.arange(ne)
        il_of = np.full(mesh.nz, -1)
        il_of[self.layers] = np.arange(nl)
        kz = W.row % mesh.nz
        ok = (ie_of[W.col] >= 0) & (il_of[kz] >= 0)
        node = ie_of[W.col[ok]] * nl + il_of[kz[ok]]
        # M: cell -> node weights (rows sum to 1 for every fuel-bearing cell)
        self.M = sp.csr_matrix((W.data[ok], (W.row[ok], node)), shape=(mesh.n_cells, self.n))
        self.MT = self.M.T.tocsr()
        self.labels = [model.labels[e] for e in self.elems]

        # ---- per-node thermal parameters
        pe = th["per_element"]
        hf = np.tile(hfrac, ne)
        self.Cf = float(pe["plate_heat_capacity_j_k"]) * hf
        self.Cc = float(pe["coolant_heat_capacity_j_k"]) * hf
        self.hA = float(pe["hA_w_k"]) * hf
        self.mcp_nominal = (float(th["total_flow_kg_s"]) * float(th["fuel_flow_fraction"])
                            / ne * CP_WATER)                 # W/K per element
        self.flow = 1.0                                      # relative flow
        self.T_in = float(th["inlet_temperature"])
        self.below = np.arange(self.n) - 1                   # node feeding each node
        self.below[np.arange(ne) * nl] = -1                  # bottom nodes: fed by the inlet

        # ---- feedback data
        self.T_ref = float(fb.get("reference_temperature", self.T_in))
        frac_base = model.mix["frac_base"]
        mats = list(cfg["materials"])
        self.fb = []
        for mname, d in (fb.get("materials") or {}).items():
            f = frac_base[..., mats.index(mname)].ravel()
            vals = {x: float(cfg["materials"][mname].get(x, 0.0)) for x in XS_NAMES}
            cc = {x: float(v) for x, v in (d.get("coolant_temperature") or {}).items()}
            ff = {x: float(v) for x, v in (d.get("fuel_temperature") or {}).items()}
            for x in list(cc) + list(ff):
                if x not in XS_NAMES:
                    raise ValueError(f"unknown cross section '{x}' in feedback for {mname}")
            self.fb.append((mname, f, vals, cc, ff))

        self.Tf = np.full(self.n, self.T_in)
        self.Tc = np.full(self.n, self.T_in)

    # ------------------------------------------------------------------ conveniences
    @property
    def mcp(self):
        return self.mcp_nominal * self.flow

    def node_power(self, fission_per_cell, scale):
        """Node powers [W] from a per-cell fission-rate vector; `scale` converts to watts."""
        return scale * (self.MT @ fission_per_cell)

    def steady_state(self, q, T_in=None, flow=None):
        """Exact steady temperatures for node powers q [W]."""
        T_in = self.T_in if T_in is None else T_in
        mcp = self.mcp_nominal * (self.flow if flow is None else flow)
        qe = q.reshape(self.ne, self.nl)
        Tc = T_in + np.cumsum(qe, axis=1) / mcp
        Tf = Tc + qe / self.hA.reshape(self.ne, self.nl)
        return Tf.ravel(), Tc.ravel()

    def stored_energy(self, Tf=None, Tc=None):
        Tf = self.Tf if Tf is None else Tf
        Tc = self.Tc if Tc is None else Tc
        return np.sum(self.Cf * Tf + self.Cc * Tc)

    def outflow_power(self, Tc=None):
        """Heat carried out of the core by the coolant [W] (relative to the inlet)."""
        Tc = self.Tc if Tc is None else Tc
        return self.mcp * np.sum(Tc.reshape(self.ne, self.nl)[:, -1] - self.T_in)

    def step_be(self, dt, q):
        """Backward-Euler update of all node temperatures with end-of-step powers q [W].
        Exactly energy-conserving: q dt = d(stored) + outflow dt, step by step."""
        ne, nl = self.ne, self.nl
        Tf0, Tc0 = self.Tf.reshape(ne, nl), self.Tc.reshape(ne, nl)
        Cf, Cc, hA = (a.reshape(ne, nl) for a in (self.Cf, self.Cc, self.hA))
        qe = q.reshape(ne, nl)
        Tf, Tc = np.empty_like(Tf0), np.empty_like(Tc0)
        T_below = np.full(ne, self.T_in)
        mcp = self.mcp
        for j in range(nl):
            a11, a12 = Cf[:, j] / dt + hA[:, j], -hA[:, j]
            a21, a22 = -hA[:, j], Cc[:, j] / dt + hA[:, j] + mcp
            b1 = Cf[:, j] / dt * Tf0[:, j] + qe[:, j]
            b2 = Cc[:, j] / dt * Tc0[:, j] + mcp * T_below
            det = a11 * a22 - a12 * a21
            Tf[:, j] = (b1 * a22 - a12 * b2) / det
            Tc[:, j] = (a11 * b2 - a21 * b1) / det
            T_below = Tc[:, j]
        self.Tf, self.Tc = Tf.ravel(), Tc.ravel()

    # ------------------------------------------------------------------ feedback on xs
    def apply(self, xs, Tf=None, Tc=None):
        """Cross sections with temperature feedback (returns new arrays; xs is not modified)."""
        Tf = self.Tf if Tf is None else Tf
        Tc = self.Tc if Tc is None else Tc
        shape = self.mesh.shape
        dTf = self.M @ (Tf - self.T_ref)
        dTc = self.M @ (Tc - self.T_ref)
        out = {k: np.asarray(v, float).ravel().copy() for k, v in xs.items()}
        for _, f, vals, cc, ff in self.fb:
            for x in set(cc) | set(ff):
                r = cc.get(x, 0.0) * dTc + ff.get(x, 0.0) * dTf
                if x in ("D1", "D2"):
                    D0 = vals[x]
                    out[x] = 1.0 / (1.0 / out[x] + f * (1.0 / (D0 * (1 + r)) - 1.0 / D0))
                else:
                    out[x] = out[x] + f * vals[x] * r
        return {k: v.reshape(shape) for k, v in out.items()}

    def rho_sensitivity(self, xs, k0, w1, w2, psi1, psi2, bc, Tf=None, Tc=None):
        """Exact first-order d(rho)/d(T_f,k) and d(rho)/d(T_c,k) for every node, for
        rho = 1 - <w, L psi> / <w, F psi> evaluated with cross sections xs (which must
        already include the feedback at Tf, Tc)."""
        Tf = self.Tf if Tf is None else Tf
        Tc = self.Tc if Tc is None else Tc
        x = {k: np.asarray(v, float).ravel() for k, v in xs.items()}
        wF = w1 @ ((x["nSf1"] * psi1 + x["nSf2"] * psi2) / k0)
        wL = (w1 @ ((x["Sa1"] + x["Ss12"]) * psi1) + w2 @ (x["Sa2"] * psi2) - w2 @ (x["Ss12"] * psi1)
              + _diffusion_form(self.mesh, x["D1"], w1, psi1, bc)
              + _diffusion_form(self.mesh, x["D2"], w2, psi2, bc))
        rho = 1.0 - wL / wF
        # d<w,L psi>/d xs (per cell) and d<w,F psi>/d xs
        dL = {"Sa1": w1 * psi1, "Ss12": (w1 - w2) * psi1, "Sa2": w2 * psi2,
              "D1": _diffusion_form_dD(self.mesh, x["D1"], w1, psi1, bc),
              "D2": _diffusion_form_dD(self.mesh, x["D2"], w2, psi2, bc)}
        dF = {"nSf1": w1 * psi1 / k0, "nSf2": w1 * psi2 / k0}
        drho = {k: (-dL.get(k, 0.0) + (1.0 - rho) * dF.get(k, 0.0)) / wF for k in XS_NAMES}

        dTf_cell = self.M @ (Tf - self.T_ref)
        dTc_cell = self.M @ (Tc - self.T_ref)
        Sf = np.zeros(self.mesh.n_cells)
        Sc = np.zeros(self.mesh.n_cells)
        for _, f, vals, cc, ff in self.fb:
            for k in set(cc) | set(ff):
                if k in ("D1", "D2"):
                    D0 = vals[k]
                    r = cc.get(k, 0.0) * dTc_cell + ff.get(k, 0.0) * dTf_cell
                    Dm = D0 * (1 + r)
                    dD_dr = x[k] ** 2 * f * D0 / Dm**2          # d D_cell / d r
                else:
                    dD_dr = f * vals[k]                          # d Sigma_cell / d r
                Sc += drho[k] * dD_dr * cc.get(k, 0.0)
                Sf += drho[k] * dD_dr * ff.get(k, 0.0)
        return self.MT @ Sf, self.MT @ Sc, rho


# ----------------------------------------------------------------------------- helpers
def _faces(mesh):
    for axis, h, (lo_f, hi_f) in ((0, mesh.dx, ("x-", "x+")), (1, mesh.dy, ("y-", "y+")),
                                  (2, mesh.dz, ("z-", "z+"))):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis], hi[axis] = slice(0, -1), slice(1, None)
        first = [slice(None)] * 3
        last = [slice(None)] * 3
        first[axis], last[axis] = 0, -1
        yield h, tuple(lo), tuple(hi), (tuple(first), lo_f), (tuple(last), hi_f)


def _bc_coeff(D, h, bc):
    if bc[0] == "reflective":
        return np.zeros_like(D), np.zeros_like(D)
    if bc[0] == "zero":
        return 2 * D / h**2, np.full_like(D, 2 / h**2)
    g = bc[1]
    den = h / (2 * D) + 1 / g
    return 1 / (h * den), (h / (2 * D**2)) / (h * den**2)


def _diffusion_form(mesh, D, w, psi, bc):
    """<w, -div D grad psi> = sum over faces c (w_i - w_j)(psi_i - psi_j) + boundary terms."""
    D3, w3, p3 = (a.reshape(mesh.shape) for a in (D, w, psi))
    tot = 0.0
    for h, lo, hi, *bnd in _faces(mesh):
        c = 2 * D3[lo] * D3[hi] / ((D3[lo] + D3[hi]) * h**2)
        tot += np.sum(c * (w3[lo] - w3[hi]) * (p3[lo] - p3[hi]))
        for sl, face in bnd:
            cb, _ = _bc_coeff(D3[sl], h, bc[face])
            tot += np.sum(cb * w3[sl] * p3[sl])
    return tot


def _diffusion_form_dD(mesh, D, w, psi, bc):
    """Per-cell derivative of _diffusion_form with respect to that cell's D."""
    D3, w3, p3 = (a.reshape(mesh.shape) for a in (D, w, psi))
    out = np.zeros(mesh.shape)
    for h, lo, hi, *bnd in _faces(mesh):
        Dl, Dh = D3[lo], D3[hi]
        g = (w3[lo] - w3[hi]) * (p3[lo] - p3[hi])
        den = (Dl + Dh) ** 2 * h**2
        out[lo] += 2 * Dh**2 / den * g
        out[hi] += 2 * Dl**2 / den * g
        for sl, face in bnd:
            _, dcb = _bc_coeff(D3[sl], h, bc[face])
            out[sl] += dcb * w3[sl] * p3[sl]
    return out.ravel()
