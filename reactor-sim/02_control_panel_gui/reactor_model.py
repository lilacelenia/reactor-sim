"""
reactor_model.py -- the simulation, with no GUI code at all.

Keeping the physics separate from the interface is the single most important design
choice in a GUI simulation: you can test it headless, swap the GUI later, or run it in
batch for analysis.

Model: point kinetics (6 delayed groups) + lumped fuel/coolant temperatures with
Doppler and moderator-temperature feedback, and a control rod bank with an S-shaped
integral worth curve. Values are illustrative, not a real plant.
"""
import numpy as np

# ---- Kinetics data (U-235 thermal, Keepin 6-group) ----
BETA_I = np.array([0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273])
LAM_I = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])   # 1/s
BETA = BETA_I.sum()
LAMBDA = 1e-4                                                  # prompt generation time [s]


class Reactor:
    # ---- Plant parameters ----
    P_NOM = 3000.0            # nominal thermal power [MW]
    ROD_WORTH = 0.04          # total bank worth (~$6)
    ROD_CRIT = 50.0           # % withdrawn at which the hot core is exactly critical
    ROD_SPEED = 0.5           # %/s normal drive speed (~$0.06/s near mid-core)
    SCRAM_SPEED = 50.0        # %/s when tripped (rods fall in ~2 s)
    ALPHA_FUEL = -3.0e-5      # Doppler coefficient [1/K]
    ALPHA_MOD = -2.0e-4       # moderator temperature coefficient [1/K]
    T_FUEL0, T_MOD0 = 900.0, 580.0   # reference (critical) temperatures [K]
    T_INLET0 = 560.0                 # nominal coolant inlet temp [K]
    MC_FUEL = 5.0e7 / 1e6            # fuel heat capacity [MJ/K]
    MC_MOD = 2.0e8 / 1e6             # coolant heat capacity in core [MJ/K]
    HIGH_POWER_TRIP = 1.18           # auto-SCRAM above 118% power
    HIGH_TEMP_TRIP = 1400.0          # auto-SCRAM above this fuel temperature [K]

    def __init__(self):
        # heat-transfer constants chosen so n=1 at reference temps is a steady state
        self.h_fm = self.P_NOM / (self.T_FUEL0 - self.T_MOD0)          # fuel->coolant [MW/K]
        self.w_cool = self.P_NOM / (2 * (self.T_MOD0 - self.T_INLET0))  # flow term [MW/K]
        self.reset()

    # ---------- state ----------
    def reset(self):
        self.t = 0.0
        self.n = 1.0                                  # relative power
        self.C = BETA_I / (LAM_I * LAMBDA)            # equilibrium precursors at n=1
        self.T_fuel, self.T_mod = self.T_FUEL0, self.T_MOD0
        self.rod = self.ROD_CRIT                      # actual rod position [% withdrawn]
        self.rod_target = self.ROD_CRIT               # where the operator wants it
        self.T_inlet = self.T_INLET0
        self.flow = 1.0                               # relative coolant flow
        self.scrammed = False
        self.trip_reason = ""

    # ---------- operator inputs (called by the GUI) ----------
    def set_rod_target(self, pct):
        if not self.scrammed:
            self.rod_target = float(np.clip(pct, 0, 100))

    def scram(self, reason="Manual SCRAM"):
        if not self.scrammed:
            self.scrammed = True
            self.trip_reason = reason
            self.rod_target = 0.0

    # ---------- physics ----------
    def rod_fraction(self, pct):
        """Integral rod worth: S-curve, 0 when fully inserted, 1 when fully withdrawn."""
        x = pct / 100.0
        return x - np.sin(2 * np.pi * x) / (2 * np.pi)

    def reactivity(self):
        rho_rod = self.ROD_WORTH * (self.rod_fraction(self.rod) - self.rod_fraction(self.ROD_CRIT))
        rho_fb = (self.ALPHA_FUEL * (self.T_fuel - self.T_FUEL0)
                  + self.ALPHA_MOD * (self.T_mod - self.T_MOD0))
        return rho_rod + rho_fb

    @property
    def power_mw(self):
        return self.n * self.P_NOM

    def step(self, dt):
        """Advance the simulation by dt seconds (sub-stepped for stability)."""
        n_sub = max(1, int(np.ceil(dt / 0.01)))
        h = dt / n_sub
        for _ in range(n_sub):
            self._substep(h)
        self._check_trips()

    def _substep(self, h):
        # 1) rods move toward target at a limited speed
        speed = self.SCRAM_SPEED if self.scrammed else self.ROD_SPEED
        self.rod += np.clip(self.rod_target - self.rod, -speed * h, speed * h)

        # 2) neutronics: kinetics are stiff, so use backward (implicit) Euler on
        #    the linear system [n, C] with reactivity frozen over the substep.
        rho = self.reactivity()
        A = np.zeros((7, 7))
        A[0, 0] = (rho - BETA) / LAMBDA
        A[0, 1:] = LAM_I
        A[1:, 0] = BETA_I / LAMBDA
        A[1:, 1:] = np.diag(-LAM_I)
        y = np.linalg.solve(np.eye(7) - h * A, np.concatenate(([self.n], self.C)))
        self.n, self.C = max(y[0], 1e-12), y[1:]

        # 3) thermal-hydraulics: lumped fuel and coolant energy balances
        q_fm = self.h_fm * (self.T_fuel - self.T_mod)                 # fuel -> coolant [MW]
        q_out = 2 * self.w_cool * self.flow * (self.T_mod - self.T_inlet)  # carried away [MW]
        self.T_fuel += h * (self.power_mw - q_fm) / self.MC_FUEL
        self.T_mod += h * (q_fm - q_out) / self.MC_MOD
        self.t += h

    def _check_trips(self):
        if self.n > self.HIGH_POWER_TRIP:
            self.scram(f"High power trip ({self.n*100:.0f}%)")
        elif self.T_fuel > self.HIGH_TEMP_TRIP:
            self.scram(f"High fuel temp trip ({self.T_fuel:.0f} K)")

    def period(self):
        """Approximate stable period [s] (one-delayed-group formula); inf when ~critical."""
        rho = self.reactivity()
        if abs(rho) < 1e-6:
            return np.inf
        lam_eff = BETA / np.sum(BETA_I / LAM_I)        # effective decay constant ~0.08 /s
        return (BETA - rho) / (rho * lam_eff) if rho < BETA else LAMBDA / (rho - BETA)


if __name__ == "__main__":
    # Headless self-test: no GUI needed to check the physics.
    r = Reactor()
    for _ in range(100):
        r.step(0.1)
    print(f"Steady state 10 s:  n={r.n:.5f}  T_fuel={r.T_fuel:.2f}  T_mod={r.T_mod:.2f}")

    r.set_rod_target(51)        # small withdrawal
    for _ in range(1200):
        r.step(0.1)
    print(f"After rods to 51%, 120 s: n={r.n:.3f}  T_fuel={r.T_fuel:.1f}  rho=${r.reactivity()/BETA:.4f}")

    r.set_rod_target(70)        # big withdrawal -> should trip on high power
    for _ in range(600):
        r.step(0.1)
        if r.scrammed:
            break
    print(f"Big withdrawal: scrammed={r.scrammed} ({r.trip_reason}) at t={r.t:.1f} s")
    for _ in range(100):
        r.step(0.1)
    print(f"10 s after SCRAM: n={r.n:.4f}, rods={r.rod:.1f}%")
