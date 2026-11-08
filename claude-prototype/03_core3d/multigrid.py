"""
multigrid.py -- smoothed-aggregation algebraic multigrid (SA-AMG) preconditioner for the
symmetric positive definite group diffusion matrices, used inside conjugate gradients.

Aggregates are 2 x 2 x 2 blocks of the structured mesh (cells in C order over (nx, ny, nz)),
so no graph aggregation is needed; everything else is algebraic, which handles the strong
coefficient jumps (absorbers, reflector) that a geometric multigrid with rediscretized coarse
operators would get wrong:
  tentative prolongator  T: piecewise constant on each aggregate, columns normalized
  smoothed prolongator   P = (I - w D^-1 A) T,  w = 4 / (3 rho(D^-1 A))
  coarse operator        A_c = P^T A P            (Galerkin: stays SPD)
repeated until the grid has <= `coarse_size` cells, where a sparse LU solves exactly.
One V(nu, nu)-cycle with damped Jacobi smoothing (same weight before and after) is a
symmetric positive definite operator, as CG requires.

Only scipy is used (pyamg would do the same with more options; it has native Apple Silicon
wheels if you prefer it). Setup cost is a few sparse matrix products, cheap next to the
solves it saves, so it can be rebuilt whenever the matrix changes (every time step).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _spectral_radius_DinvA(A, dinv, iters=15, seed=0):
    """Estimate rho(D^-1 A) by power iteration (an upper bound is what the smoother needs,
    so the estimate is inflated slightly)."""
    x = np.random.default_rng(seed).random(A.shape[0])
    lam = 1.0
    for _ in range(iters):
        y = dinv * (A @ x)
        lam = np.linalg.norm(y) / np.linalg.norm(x)
        x = y / np.linalg.norm(y)
    return 1.1 * lam


def _tentative(shape):
    """Piecewise-constant prolongator over 2 x 2 x 2 aggregates of a C-ordered grid, with
    unit-norm columns. Returns (T, coarse shape)."""
    nx, ny, nz = shape
    cx, cy, cz = (nx + 1) // 2, (ny + 1) // 2, (nz + 1) // 2
    i, j, k = np.meshgrid(np.arange(nx) // 2, np.arange(ny) // 2, np.arange(nz) // 2, indexing="ij")
    agg = ((i * cy + j) * cz + k).ravel()
    n, nc = nx * ny * nz, cx * cy * cz
    size = np.bincount(agg, minlength=nc).astype(float)
    T = sp.csr_matrix((1.0 / np.sqrt(size[agg]), (np.arange(n), agg)), shape=(n, nc))
    return T, (cx, cy, cz)


class _Level:
    __slots__ = ("A", "dinv", "omega", "wdinv", "P", "R")


class SAPreconditioner:
    """M ~ A^-1 as one V-cycle. Use M(r) or M.matvec(r).

    A: SPD sparse matrix on a structured grid of `shape` (C order).
    nu: smoothing sweeps before and after the coarse correction.
    """

    def __init__(self, A, shape, nu=1, coarse_size=500, max_levels=10, smooth=True, alpha=1.0):
        self.nu, self.alpha = nu, alpha
        self.levels = []
        A = sp.csr_matrix(A)
        while True:
            lev = _Level()
            lev.A = A
            lev.dinv = 1.0 / A.diagonal()
            rho = _spectral_radius_DinvA(A, lev.dinv)
            lev.omega = 4.0 / (3.0 * rho)                               # Jacobi weight
            lev.wdinv = lev.omega * lev.dinv
            self.levels.append(lev)
            if A.shape[0] <= coarse_size or len(self.levels) >= max_levels or min(shape) < 2:
                break
            T, shape = _tentative(shape)
            P = T - (lev.omega * sp.diags(lev.dinv)) @ (A @ T) if smooth else T
            lev.P = sp.csr_matrix(P)
            lev.R = sp.csr_matrix(P.T)
            A = sp.csr_matrix(lev.R @ A @ lev.P)
        self.coarse_lu = spla.splu(sp.csc_matrix(self.levels[-1].A))
        self.shape = (self.levels[0].A.shape[0],) * 2
        self.n_levels = len(self.levels)
        self.complexity = sum(l.A.nnz for l in self.levels) / self.levels[0].A.nnz

    def _vcycle(self, lvl, b):
        if lvl == len(self.levels) - 1:
            return self.coarse_lu.solve(b)
        L = self.levels[lvl]
        wd = L.wdinv
        x = wd * b                                       # first sweep from x = 0
        for _ in range(self.nu - 1):
            r = L.A @ x
            np.subtract(b, r, out=r)
            r *= wd
            x += r
        r = L.A @ x
        np.subtract(b, r, out=r)
        e = L.P @ self._vcycle(lvl + 1, L.R @ r)
        if self.alpha != 1.0:
            e *= self.alpha
        x += e
        for _ in range(self.nu):
            r = L.A @ x
            np.subtract(b, r, out=r)
            r *= wd
            x += r
        return x

    def __call__(self, r):
        return self._vcycle(0, r)

    matvec = __call__
