"""
diffusion3d.py -- generic 3D two-group neutron diffusion solver on a Cartesian mesh.

Knows nothing about any particular reactor: it takes per-cell cross sections and boundary
conditions and returns k-effective and fluxes. Core-specific geometry lives elsewhere
(see core_builder.py), which is what lets one solver serve many configurations.

Equations (per cell, finite volume, 7-point stencil):
    -div D1 grad phi1 + (Sa1 + Ss12) phi1 = (1/k) (nuSf1 phi1 + nuSf2 phi2) [+ Q1]
    -div D2 grad phi2 +  Sa2         phi2 = Ss12 phi1                       [+ Q2]

Interface coupling uses the harmonic mean of D, so material discontinuities are handled
correctly. Boundary faces can be:
    ("albedo", gamma)  partial-current condition J_out = gamma * phi_surface
                       (gamma = 0.5 is the Marshak vacuum condition; the IAEA benchmarks
                        use 0.4692)
    ("reflective",)    zero net current (symmetry plane)
    ("zero",)          phi = 0 on the face (used for analytic verification)
"""
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

XS_NAMES = ("D1", "D2", "Sa1", "Sa2", "Ss12", "nSf1", "nSf2")
FACES = ("x-", "x+", "y-", "y+", "z-", "z+")


@dataclass
class Mesh:
    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    x0: float = 0.0
    y0: float = 0.0
    z0: float = 0.0

    @property
    def shape(self):
        return (self.nx, self.ny, self.nz)

    @property
    def n_cells(self):
        return self.nx * self.ny * self.nz

    @property
    def cell_volume(self):
        return self.dx * self.dy * self.dz

    def centers(self, axis):
        n, d, o = {"x": (self.nx, self.dx, self.x0), "y": (self.ny, self.dy, self.y0),
                   "z": (self.nz, self.dz, self.z0)}[axis]
        return o + d * (np.arange(n) + 0.5)


@dataclass
class Problem:
    mesh: Mesh
    xs: dict                                   # name -> array of shape mesh.shape
    bc: dict = field(default_factory=lambda: {f: ("albedo", 0.5) for f in FACES})


@dataclass
class Result:
    k: float
    phi1: np.ndarray
    phi2: np.ndarray
    power: np.ndarray        # relative fission rate density, mean over fissile cells = 1
    outer_iterations: int
    converged: bool


# ----------------------------------------------------------------------------- operators
def _boundary_coeff(D, h, bc):
    """Leakage coefficient (per unit volume) for a boundary face of a cell of width h."""
    kind = bc[0]
    if kind == "reflective":
        return np.zeros_like(D)
    if kind == "zero":
        return 2.0 * D / h**2
    if kind == "albedo":
        gamma = bc[1]
        return 1.0 / (h * (h / (2.0 * D) + 1.0 / gamma))
    raise ValueError(f"unknown boundary condition {bc}")


def build_loss_matrix(mesh, D, removal, bc):
    """Sparse SPD matrix for  -div(D grad .) + removal  on the mesh."""
    nx, ny, nz = mesh.shape
    idx = np.arange(mesh.n_cells).reshape(mesh.shape)
    diag = removal.astype(float).copy()
    rows, cols, vals = [], [], []

    for axis, h, (f_lo, f_hi) in ((0, mesh.dx, ("x-", "x+")), (1, mesh.dy, ("y-", "y+")),
                                  (2, mesh.dz, ("z-", "z+"))):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis], hi[axis] = slice(0, -1), slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        # interior faces: harmonic-mean coupling
        Dl, Dh = D[lo], D[hi]
        c = 2.0 * Dl * Dh / ((Dl + Dh) * h**2)
        diag[lo] += c
        diag[hi] += c
        a, b = idx[lo].ravel(), idx[hi].ravel()
        rows += [a, b]
        cols += [b, a]
        vals += [-c.ravel(), -c.ravel()]
        # boundary faces
        first = [slice(None)] * 3
        last = [slice(None)] * 3
        first[axis], last[axis] = 0, -1
        diag[tuple(first)] += _boundary_coeff(D[tuple(first)], h, bc[f_lo])
        diag[tuple(last)] += _boundary_coeff(D[tuple(last)], h, bc[f_hi])

    rows.append(idx.ravel())
    cols.append(idx.ravel())
    vals.append(diag.ravel())
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(mesh.n_cells, mesh.n_cells))
    return A


class _GroupSolver:
    """Solves A x = b repeatedly with warm-started, Jacobi-preconditioned CG.
    (A is symmetric positive definite.) Small systems use a direct LU instead."""

    def __init__(self, A, direct_limit=60_000):
        self.A = A
        self.lu = spla.splu(A.tocsc()) if A.shape[0] <= direct_limit else None
        self.M = sp.diags(1.0 / A.diagonal())

    def solve(self, b, x0, rtol):
        if self.lu is not None:
            return self.lu.solve(b)
        x, info = spla.cg(self.A, b, x0=x0, rtol=rtol, maxiter=5000, M=self.M)
        if info > 0:
            raise RuntimeError("CG did not converge")
        return x


# ----------------------------------------------------------------------------- solvers
def solve_k(problem, tol_k=1e-6, tol_src=1e-5, max_outer=3000, guess=None, verbose=False):
    """k-eigenvalue problem by power (source) iteration.

    guess: optional previous Result, used to warm-start (much faster for sweeps)."""
    m, xs = problem.mesh, problem.xs
    flat = {k: np.asarray(v, float).ravel() for k, v in xs.items()}
    A1 = build_loss_matrix(m, xs["D1"], xs["Sa1"] + xs["Ss12"], problem.bc)
    A2 = build_loss_matrix(m, xs["D2"], xs["Sa2"], problem.bc)
    g1, g2 = _GroupSolver(A1), _GroupSolver(A2)

    if guess is not None:
        phi1, phi2, k = guess.phi1.ravel().copy(), guess.phi2.ravel().copy(), guess.k
    else:
        phi1, phi2, k = np.ones(m.n_cells), np.ones(m.n_cells), 1.0
    src = flat["nSf1"] * phi1 + flat["nSf2"] * phi2
    src /= np.linalg.norm(src)

    converged = False
    for it in range(1, max_outer + 1):
        rtol = max(1e-10, min(1e-4, tol_src * 0.1))
        phi1 = g1.solve(src / k, phi1, rtol)
        phi2 = g2.solve(flat["Ss12"] * phi1, phi2, rtol)
        new = flat["nSf1"] * phi1 + flat["nSf2"] * phi2
        k_new = k * new.sum() / src.sum()
        norm = np.linalg.norm(new)
        new /= norm
        phi1 /= norm
        phi2 /= norm
        dsrc = np.linalg.norm(new - src)
        dk = abs(k_new - k)
        src, k = new, k_new
        if verbose and (it % 25 == 0 or it == 1):
            print(f"    outer {it:4d}  k = {k:.7f}  dk = {dk:.1e}  dsrc = {dsrc:.1e}")
        if dk < tol_k and dsrc < tol_src:
            converged = True
            break

    return _package(problem, k, phi1, phi2, it, converged)


def solve_fixed_source(problem, Q1, Q2=None, tol=1e-6, max_outer=5000):
    """Subcritical multiplication with an external source (k_eff < 1 required):
        (L - F) phi = Q, solved by source iteration  phi <- L^-1 (F phi + Q).
    This is the piece needed for simulating a startup from source level."""
    m, xs = problem.mesh, problem.xs
    flat = {k: np.asarray(v, float).ravel() for k, v in xs.items()}
    Q1 = np.asarray(Q1, float).ravel()
    Q2 = np.zeros(m.n_cells) if Q2 is None else np.asarray(Q2, float).ravel()
    g1 = _GroupSolver(build_loss_matrix(m, xs["D1"], xs["Sa1"] + xs["Ss12"], problem.bc))
    g2 = _GroupSolver(build_loss_matrix(m, xs["D2"], xs["Sa2"], problem.bc))
    phi1, phi2 = np.zeros(m.n_cells), np.zeros(m.n_cells)
    fis = np.zeros(m.n_cells)
    for it in range(1, max_outer + 1):
        phi1 = g1.solve(fis + Q1, phi1, 1e-10)
        phi2 = g2.solve(flat["Ss12"] * phi1 + Q2, phi2, 1e-10)
        new = flat["nSf1"] * phi1 + flat["nSf2"] * phi2
        if np.linalg.norm(new - fis) <= tol * max(np.linalg.norm(new), 1e-300):
            fis = new
            break
        fis = new
    return _package(problem, np.nan, phi1, phi2, it, True)


def _package(problem, k, phi1, phi2, it, converged):
    m, xs = problem.mesh, problem.xs
    shape = m.shape
    phi1, phi2 = phi1.reshape(shape), phi2.reshape(shape)
    fission = xs["nSf1"] * phi1 + xs["nSf2"] * phi2
    fissile = (xs["nSf1"] + xs["nSf2"]) > 0
    power = fission / fission[fissile].mean() if fissile.any() else fission
    return Result(k, phi1, phi2, power, it, converged)
