"""
verify_solver.py -- checks diffusion3d against problems with known answers.

1. Bare homogeneous cube, phi = 0 on all faces: analytic k from the buckling
   B^2 = 3 (pi/a)^2. Error should fall ~4x per mesh halving (2nd-order accuracy).
2. Symmetry: one octant with reflective planes must reproduce the full-cube k.
3. Albedo boundaries: a slab with reflective x/y faces is a 1D problem; with
   Marshak (gamma=0.5) z faces it must match a direct 1D eigen solve, which checks the
   boundary coefficient and heterogeneous interface coupling.
4. Fixed source: subcritical multiplication M = 1/(1-k) for a fundamental-mode source.
"""
import numpy as np
import scipy.sparse.linalg as spla

from diffusion3d import FACES, Mesh, Problem, build_loss_matrix, solve_fixed_source, solve_k

FUEL = dict(D1=1.5, D2=0.4, Sa1=0.01, Sa2=0.08, Ss12=0.02, nSf1=0.0, nSf2=0.135)
REFL = dict(D1=2.0, D2=0.3, Sa1=0.0, Sa2=0.01, Ss12=0.04, nSf1=0.0, nSf2=0.0)


def uniform(mesh, mat):
    return {k: np.full(mesh.shape, v, float) for k, v in mat.items()}


def k_bare(a, m=FUEL):
    B2 = 3 * (np.pi / a) ** 2
    return (m["nSf1"] / (m["D1"] * B2 + m["Sa1"] + m["Ss12"])
            + m["Ss12"] * m["nSf2"] / ((m["D1"] * B2 + m["Sa1"] + m["Ss12"]) * (m["D2"] * B2 + m["Sa2"])))


def test_bare_cube():
    a = 200.0
    k_exact = k_bare(a)
    print(f"[1] Bare cube a={a:.0f} cm, analytic k = {k_exact:.7f}")
    errs = []
    for n in (8, 16, 32, 64):
        mesh = Mesh(n, n, n, a / n, a / n, a / n)
        r = solve_k(Problem(mesh, uniform(mesh, FUEL), {f: ("zero",) for f in FACES}), tol_k=1e-9)
        errs.append(abs(r.k - k_exact))
        print(f"    {n:3d}^3 cells: k = {r.k:.7f}  error = {errs[-1]:.2e}  outer its = {r.outer_iterations}")
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    print(f"    convergence order: {np.round(orders, 2)}  (expect 2)")
    assert np.all(orders > 1.9), "not second order"
    return errs


def test_symmetry():
    a, n = 200.0, 32
    full = Mesh(n, n, n, a / n, a / n, a / n)
    r_full = solve_k(Problem(full, uniform(full, FUEL), {f: ("zero",) for f in FACES}), tol_k=1e-10)
    h = n // 2
    octant = Mesh(h, h, h, a / n, a / n, a / n)
    bc = {f: ("zero",) for f in FACES}
    bc.update({"x-": ("reflective",), "y-": ("reflective",), "z-": ("reflective",)})
    r_oct = solve_k(Problem(octant, uniform(octant, FUEL), bc), tol_k=1e-10)
    d = abs(r_full.k - r_oct.k)
    print(f"[2] Octant symmetry: full k = {r_full.k:.8f}, octant k = {r_oct.k:.8f}, diff = {d:.1e}")
    assert d < 1e-7


def test_reflected_slab_vs_1d():
    # z: 30 cm reflector | 100 cm fuel | 30 cm reflector, Marshak vacuum at both ends
    dz, nz = 2.0, 80
    zc = dz * (np.arange(nz) + 0.5)
    in_fuel = (zc > 30) & (zc < 130)
    mesh3 = Mesh(4, 4, nz, 10.0, 10.0, dz)
    xs3 = {k: np.where(in_fuel[None, None, :], FUEL[k], REFL[k]) * np.ones(mesh3.shape) for k in FUEL}
    bc = {f: ("reflective",) for f in FACES}
    bc.update({"z-": ("albedo", 0.5), "z+": ("albedo", 0.5)})
    r3 = solve_k(Problem(mesh3, xs3, bc), tol_k=1e-10, tol_src=1e-8)

    # independent 1D reference: dense generalized eigenproblem on the same z mesh
    mesh1 = Mesh(1, 1, nz, 1.0, 1.0, dz)
    xs1 = {k: np.where(in_fuel, FUEL[k], REFL[k]).reshape(mesh1.shape) for k in FUEL}
    L1 = build_loss_matrix(mesh1, xs1["D1"], xs1["Sa1"] + xs1["Ss12"], bc).toarray()
    L2 = build_loss_matrix(mesh1, xs1["D2"], xs1["Sa2"], bc).toarray()
    S = np.diag(xs1["Ss12"].ravel())
    L = np.block([[L1, np.zeros_like(L1)], [-S, L2]])
    F = np.block([[np.diag(xs1["nSf1"].ravel()), np.diag(xs1["nSf2"].ravel())],
                  [np.zeros_like(L1), np.zeros_like(L1)]])
    eig = np.linalg.eigvals(np.linalg.solve(L, F))
    k1 = np.max(eig.real)
    d = abs(r3.k - k1)
    print(f"[3] Reflected slab with albedo BCs: 3D k = {r3.k:.8f}, 1D dense eigen k = {k1:.8f}, diff = {d:.1e}")
    assert d < 1e-6


def test_fixed_source():
    a, n = 100.0, 16     # small cube -> subcritical
    mesh = Mesh(n, n, n, a / n, a / n, a / n)
    prob = Problem(mesh, uniform(mesh, FUEL), {f: ("zero",) for f in FACES})
    r = solve_k(prob, tol_k=1e-11, tol_src=1e-9)
    assert r.k < 1, 'fixed-source test needs a subcritical system'
    # source shaped like the fundamental mode -> multiplication is exactly 1/(1-k)
    shape = r.phi1 / r.phi1.max()
    Q = FUEL["Sa1"] * shape        # arbitrary scale
    # first-generation fission production (fission switched off) vs. fully multiplied
    no_mult = solve_fixed_source(Problem(mesh, {**uniform(mesh, FUEL), "nSf2": np.zeros(mesh.shape)},
                                         prob.bc), Q1=Q)
    with_mult = solve_fixed_source(prob, Q1=Q, tol=1e-10)
    f0 = (FUEL["nSf2"] * no_mult.phi2).sum()
    f1 = (FUEL["nSf2"] * with_mult.phi2).sum()
    M_sim, M_theory = f1 / f0, 1 / (1 - r.k)
    print(f"[4] Fixed source: k = {r.k:.5f}, multiplication {M_sim:.4f} vs 1/(1-k) = {M_theory:.4f}")
    assert abs(M_sim - M_theory) / abs(M_theory) < 1e-4


if __name__ == "__main__":
    test_bare_cube()
    test_symmetry()
    test_reflected_slab_vs_1d()
    test_fixed_source()
    print("\nAll verification tests passed.")
