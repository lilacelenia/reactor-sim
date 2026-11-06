"""
Point reactor kinetics with 6 delayed-neutron groups + optional lumped fuel-temperature feedback.

Equations (n = power normalized to 1 at t=0, C_i = precursor concentrations):
    dn/dt   = (rho(t) - beta) / Lambda * n + sum_i lambda_i * C_i
    dC_i/dt = beta_i / Lambda * n - lambda_i * C_i
With feedback, a lumped fuel temperature T evolves as
    dT/dt   = (P0 * n - h * (T - T_cool)) / (m * c)
    rho     = rho_ext(t) + alpha_D * (T - T0)       (alpha_D < 0: Doppler feedback)

The system is stiff (prompt timescale ~1e-4 s vs precursor timescales ~ seconds-minutes),
so an implicit solver (BDF/Radau) is used.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Kinetics data: U-235 thermal fission, Keepin 6-group ---
beta_i = np.array([0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273])
lam_i = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])  # 1/s
beta = beta_i.sum()                                          # ~0.0065
Lambda = 1e-4                                                # prompt generation time [s]

# --- Lumped thermal model (illustrative values, not a real plant) ---
P0 = 3000e6        # nominal power [W]
mc = 5.0e7         # fuel heat capacity m*c [J/K]
T_cool = 560.0     # coolant temperature [K]
h = P0 / (900.0 - T_cool)   # chosen so steady state fuel temp T0 = 900 K at n = 1
T0 = 900.0
alpha_D = -3.0e-5  # Doppler coefficient [1/K]


def rho_ext(t, step_dollars=0.1, t_step=1.0):
    """External reactivity: step insertion of `step_dollars` at t_step."""
    return step_dollars * beta if t >= t_step else 0.0


def rhs(t, y, feedback, step_dollars):
    n, C, T = y[0], y[1:7], y[7]
    rho = rho_ext(t, step_dollars)
    if feedback:
        rho += alpha_D * (T - T0)
    dn = (rho - beta) / Lambda * n + lam_i @ C
    dC = beta_i / Lambda * n - lam_i * C
    dT = (P0 * n - h * (T - T_cool)) / mc if feedback else 0.0
    return np.concatenate(([dn], dC, [dT]))


def initial_state():
    # Steady state at n = 1: dC_i/dt = 0  =>  C_i = beta_i / (lambda_i * Lambda)
    C0 = beta_i / (lam_i * Lambda)
    return np.concatenate(([1.0], C0, [T0]))


def simulate(feedback, step_dollars=0.1, t_end=60.0):
    return solve_ivp(rhs, (0, t_end), initial_state(), args=(feedback, step_dollars),
                     method="Radau", rtol=1e-8, atol=1e-10, max_step=0.05)


if __name__ == "__main__":
    step = 0.1  # $0.10 reactivity insertion
    no_fb = simulate(feedback=False, step_dollars=step)
    fb = simulate(feedback=True, step_dollars=step)

    # Sanity check: prompt jump approximation says n jumps to beta/(beta - rho) right after the step
    rho = step * beta
    prompt_jump = beta / (beta - rho)
    idx = np.searchsorted(no_fb.t, 1.05)
    print(f"beta = {beta:.5f}, inserted rho = {rho:.6f} (${step})")
    print(f"Prompt-jump estimate: n = {prompt_jump:.4f}; simulated n(1.05 s) = {no_fb.y[0, idx]:.4f}")
    print(f"No feedback:   n(60 s) = {no_fb.y[0, -1]:.3f}")
    print(f"With feedback: n(60 s) = {fb.y[0, -1]:.3f}, fuel T = {fb.y[7, -1]:.1f} K")
    # With feedback the new equilibrium is where Doppler cancels the insertion:
    T_eq = T0 - rho / alpha_D
    n_eq = h * (T_eq - T_cool) / P0
    print(f"Analytic feedback equilibrium: n = {n_eq:.3f}, T = {T_eq:.1f} K")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(no_fb.t, no_fb.y[0], label="no feedback")
    ax1.plot(fb.t, fb.y[0], label="Doppler feedback")
    ax1.set(xlabel="time [s]", ylabel="relative power n(t)",
            title=f"${step} step reactivity insertion")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(fb.t, fb.y[7], color="C1")
    ax2.set(xlabel="time [s]", ylabel="fuel temperature [K]", title="Fuel temperature (feedback case)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("point_kinetics.png", dpi=120)
    print("Saved point_kinetics.png")
