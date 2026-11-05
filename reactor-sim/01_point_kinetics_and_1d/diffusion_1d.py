"""
1D two-group neutron diffusion eigenvalue solver for a bare slab reactor.

    -D1 phi1'' + Sigma_r1 phi1 = (1/k) [nuSf1 phi1 + nuSf2 phi2]     (fast group)
    -D2 phi2'' + Sigma_a2 phi2 = Sigma_s12 phi1                        (thermal group)

Sigma_r1 = Sigma_a1 + Sigma_s12 (removal from the fast group). Boundary condition: phi = 0
at both slab faces (ignoring extrapolation distance for simplicity).

Discretized with central finite differences, solved for k-effective by power iteration
(source iteration). Verified against the analytic bare-slab result and checked for
2nd-order convergence under mesh refinement.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# --- Two-group constants [cm, 1/cm] (illustrative, roughly LWR-like) ---
D1, D2 = 1.5, 0.4
Sa1, Sa2 = 0.010, 0.100
Ss12 = 0.020
nuSf1, nuSf2 = 0.005, 0.135
Sr1 = Sa1 + Ss12
WIDTH = 200.0  # slab width [cm]


def analytic_k(width):
    B2 = (np.pi / width) ** 2  # geometric buckling of a bare slab
    return nuSf1 / (D1 * B2 + Sr1) + Ss12 * nuSf2 / ((D1 * B2 + Sr1) * (D2 * B2 + Sa2))


def laplacian(n, dx):
    """Matrix for -d2/dx2 on n interior points with zero-flux Dirichlet BCs."""
    main = 2.0 * np.ones(n)
    off = -1.0 * np.ones(n - 1)
    return sp.diags([off, main, off], [-1, 0, 1], format="csc") / dx**2


def solve(n_cells=200, width=WIDTH, tol=1e-10, max_iter=2000):
    n = n_cells - 1                 # interior nodes
    dx = width / n_cells
    x = np.linspace(dx, width - dx, n)
    L = laplacian(n, dx)
    I = sp.identity(n, format="csc")

    # Loss operators for each group, factorized once (LU) and reused every iteration
    A1 = spla.splu((D1 * L + Sr1 * I).tocsc())
    A2 = spla.splu((D2 * L + Sa2 * I).tocsc())

    phi1, phi2 = np.ones(n), np.ones(n)
    k = 1.0
    fission = nuSf1 * phi1 + nuSf2 * phi2
    for it in range(max_iter):
        phi1 = A1.solve(fission / k)            # fast flux from fission source
        phi2 = A2.solve(Ss12 * phi1)            # thermal flux from downscatter
        new_fission = nuSf1 * phi1 + nuSf2 * phi2
        k_new = k * new_fission.sum() / fission.sum()
        converged = abs(k_new - k) < tol and np.linalg.norm(new_fission - fission) / np.linalg.norm(new_fission) < 1e-8
        # normalize source to keep numbers bounded
        scale = new_fission.max()
        fission, phi1, phi2, k = new_fission / scale, phi1 / scale, phi2 / scale, k_new
        if converged:
            break
    return x, phi1, phi2, k, it + 1


if __name__ == "__main__":
    k_exact = analytic_k(WIDTH)
    print(f"Analytic k_eff = {k_exact:.8f}")

    print("\nMesh refinement study:")
    errs = []
    cells = [25, 50, 100, 200, 400]
    for nc in cells:
        _, _, _, k, iters = solve(nc)
        err = abs(k - k_exact)
        errs.append(err)
        print(f"  cells = {nc:4d}  k = {k:.8f}  |error| = {err:.2e}  iterations = {iters}")
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    print("  observed convergence order:", np.round(orders, 2), "(expect ~2)")

    x, phi1, phi2, k, _ = solve(400)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, phi1 / phi1.max(), label="fast flux (group 1)")
    ax.plot(x, phi2 / phi1.max(), label="thermal flux (group 2)")
    ax.plot(x, np.sin(np.pi * x / WIDTH), "k--", lw=0.8, label="analytic shape sin(pi x / a)")
    ax.set(xlabel="x [cm]", ylabel="flux (normalized to fast peak)",
           title=f"Bare slab, two-group diffusion, k_eff = {k:.5f}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("diffusion_1d.png", dpi=120)
    print("\nSaved diffusion_1d.png")
