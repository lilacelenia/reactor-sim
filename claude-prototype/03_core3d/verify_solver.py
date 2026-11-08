"""
verify_solver.py -- checks diffusion3d against problems with known answers.

1. Bare homogeneous cube, phi = 0 on all faces: analytic k from the buckling
   B^2 = 3 (pi/a)^2. Error should fall ~4x per mesh halving (2nd-order accuracy).
2. Symmetry: one octant with reflective planes must reproduce the full-cube k (both
   converged tightly: with the plain power iteration the octant iterates were an exact
   restriction of the full ones, so k matched at any tolerance; Anderson mixing is not).
3. Albedo boundaries: a slab with reflective x/y faces is a 1D problem; with
   Marshak (gamma=0.5) z faces it must match a direct 1D eigen solve, which checks the
   boundary coefficient and heterogeneous interface coupling.
4. Fixed source: subcritical multiplication M = 1/(1-k) for a fundamental-mode source.
5. Multigrid preconditioner (multigrid.py): symmetric and positive definite (as CG needs);
   CG with it reproduces a sparse direct solve; its iteration count stays flat as the mesh
   is refined (Jacobi's grows ~2x per halving); the MITR 2 cm eigenvalue is the same with
   and without it.
6. Anderson acceleration of the outer iteration: the accelerated and plain power iterations
   converge to the same k and flux on the MITR core (4 cm, tight tolerance), in far fewer
   outer iterations. (Tests 1-4 above also run accelerated, against analytic answers.)
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
        r = solve_k(Problem(mesh, uniform(mesh, FUEL), {f: ("zero",) for f in FACES}), tol_k=1e-9,
                    tol_src=1e-8)
        errs.append(abs(r.k - k_exact))
        print(f"    {n:3d}^3 cells: k = {r.k:.7f}  error = {errs[-1]:.2e}  outer its = {r.outer_iterations}")
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    print(f"    convergence order: {np.round(orders, 2)}  (expect 2)")
    assert np.all(orders > 1.9), "not second order"
    return errs


def test_symmetry():
    a, n = 200.0, 32
    full = Mesh(n, n, n, a / n, a / n, a / n)
    r_full = solve_k(Problem(full, uniform(full, FUEL), {f: ("zero",) for f in FACES}), tol_k=1e-10,
                     tol_src=1e-9)
    h = n // 2
    octant = Mesh(h, h, h, a / n, a / n, a / n)
    bc = {f: ("zero",) for f in FACES}
    bc.update({"x-": ("reflective",), "y-": ("reflective",), "z-": ("reflective",)})
    r_oct = solve_k(Problem(octant, uniform(octant, FUEL), bc), tol_k=1e-10, tol_src=1e-9)
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


def test_multigrid():
    import time
    import diffusion3d as D
    from core_builder import build, load_config
    from multigrid import SAPreconditioner
    print("5. Multigrid preconditioner")
    rng = np.random.default_rng(1)
    cfg = load_config("mitr2_like.yaml")
    cfg["mesh"]["cell_size"] = 4.0
    p = build(cfg).problem_at(18.7)
    for g, (dk, sig) in enumerate((("D1", p.xs["Sa1"] + p.xs["Ss12"]), ("D2", p.xs["Sa2"]))):
        A = build_loss_matrix(p.mesh, p.xs[dk], sig, p.bc).tocsr()
        M = SAPreconditioner(A, p.mesh.shape)
        x, y = rng.random(A.shape[0]) - 0.5, rng.random(A.shape[0]) - 0.5
        asym = abs(x @ M(y) - y @ M(x)) / (np.linalg.norm(x) * np.linalg.norm(M(y)))
        pos = min((v @ M(v)) for v in rng.random((5, A.shape[0])) - 0.5)
        b = rng.random(A.shape[0])
        xd = spla.spsolve(A.tocsc(), b)
        xm = D.pcg(A, b, None, 1e-12, M)
        err = np.linalg.norm(xm - xd) / np.linalg.norm(xd)
        print(f"   MITR 4 cm group {g + 1}: {M.n_levels} levels; |x.My - y.Mx| {asym:.1e}; "
              f"min v.Mv {pos:.2e} > 0; CG(MG) vs direct {err:.1e}")
        assert asym < 1e-12 and pos > 0 and err < 1e-9
    its = {}
    for n in (16, 32, 64):
        mesh = Mesh(n, n, n, 200.0 / n, 200.0 / n, 200.0 / n)
        xs = uniform(mesh, FUEL)
        A = build_loss_matrix(mesh, xs["D1"], xs["Sa1"] + xs["Ss12"], {f: ("zero",) for f in FACES}).tocsr()
        b = np.ones(A.shape[0])
        row = []
        for M in (1.0 / A.diagonal(), SAPreconditioner(A, mesh.shape)):
            cnt = [0]
            Mf = M if callable(M) else (lambda r, d=M: d * r)

            def counted(r, Mf=Mf):
                cnt[0] += 1
                return Mf(r)
            D.pcg(A, b, None, 1e-10, counted)
            row.append(cnt[0])
        its[n] = row
        print(f"   bare cube {n}^3: CG iterations Jacobi {row[0]:4d}, multigrid {row[1]:3d}")
    # 16^3 reaches the coarse grid a level sooner; from 32^3 to 64^3 multigrid stays flat
    assert its[64][1] <= 1.3 * its[32][1] and its[64][0] >= 1.8 * its[32][0]
    cfg = load_config("mitr2_like.yaml")
    m2 = build(cfg)
    res = []
    for lim in (D.MG_LIMIT, 10**12):
        old, D.MG_LIMIT = D.MG_LIMIT, lim
        t = time.time()
        res.append((solve_k(m2.problem_at(18.7), tol_k=1e-10, tol_src=1e-9).k, time.time() - t))
        D.MG_LIMIT = old
    d = abs(res[0][0] - res[1][0])
    print(f"   MITR 2 cm eigenvalue: multigrid {res[0][0]:.12f} ({res[0][1]:.1f} s), Jacobi "
          f"{res[1][0]:.12f} ({res[1][1]:.1f} s): difference {d:.1e}")
    assert d < 1e-9


def test_anderson():
    from core_builder import build, load_config
    print("6. Anderson-accelerated outer iteration vs plain power iteration (MITR, 4 cm)")
    cfg = load_config("mitr2_like.yaml")
    cfg["mesh"]["cell_size"] = 4.0
    p = build(cfg).problem_at(18.7)
    a = solve_k(p, tol_k=1e-12, tol_src=1e-11, accelerate=True)
    b = solve_k(p, tol_k=1e-12, tol_src=1e-11, accelerate=False)
    df = np.abs(a.phi2 / a.phi2.max() - b.phi2 / b.phi2.max()).max()
    print(f"   k {a.k:.12f} ({a.outer_iterations} outers) vs {b.k:.12f} ({b.outer_iterations} "
          f"outers): difference {abs(a.k - b.k):.1e}; max thermal-flux difference {df:.1e}")
    assert a.converged and b.converged and abs(a.k - b.k) < 1e-10 and df < 1e-8
    assert a.outer_iterations < b.outer_iterations / 2


if __name__ == "__main__":
    test_bare_cube()
    test_symmetry()
    test_reflected_slab_vs_1d()
    test_fixed_source()
    test_multigrid()
    test_anderson()
    print("\nAll verification tests passed.")
