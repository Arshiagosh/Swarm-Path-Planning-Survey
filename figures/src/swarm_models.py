"""Models from Chapter 2 and Chapter 4: trail choice, double bridge, leader-follower.

* trail_bifurcation: equilibria of the dimensionless trail model
      dc_i/dt = phi (1 + c_i)^2 / ((1 + c_1)^2 + (1 + c_2)^2) - c_i
  (Garnier et al., 2007), checked against numerical integration.
* aco_bridge: Monte-Carlo double-bridge experiment with the choice rule of
  Eq. (trail-choice) and deposits inversely proportional to branch length.
* leader_follower: consensus-based formation tracking of a moving leader.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from common import ACCENT, GRAY, PRIMARY, save


# ---------------------------------------------------------- trail choice
def trail_rhs(_, c, phi):
    p = (1 + c) ** 2 / np.sum((1 + c) ** 2)
    return phi * p - c


def trail_bifurcation():
    phis = np.linspace(0.05, 6, 300)
    sym = phis / 2
    hi = np.where(phis > 2, 0.5 * (phis + np.sqrt(np.clip(phis**2 - 4, 0, None))), np.nan)
    lo = np.where(phis > 2, 0.5 * (phis - np.sqrt(np.clip(phis**2 - 4, 0, None))), np.nan)
    # numerical check: start near the symmetric state and integrate
    sims = []
    for phi in np.linspace(0.25, 6, 24):
        c0 = np.array([phi / 2 + 1e-3, phi / 2 - 1e-3])
        sol = solve_ivp(trail_rhs, (0, 400), c0, args=(phi,), rtol=1e-9, atol=1e-12)
        sims.append((phi, sol.y[0, -1], sol.y[1, -1]))
    sims = np.array(sims)
    for phi, c1, c2 in sims:
        expected = 0.5 * (phi + np.sqrt(phi**2 - 4)) if phi > 2 else phi / 2
        assert abs(max(c1, c2) - expected) < 1e-3, (phi, c1, c2, expected)

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.7))
    a.plot(phis[phis <= 2], sym[phis <= 2], color=PRIMARY, lw=1.8, label="stable")
    a.plot(phis[phis > 2], sym[phis > 2], "--", color=PRIMARY, lw=1.2, label="unstable")
    a.plot(phis, hi, color=ACCENT, lw=1.8, label="collective choice")
    a.plot(phis, lo, color=ACCENT, lw=1.8)
    a.scatter(sims[:, 0], np.maximum(sims[:, 1], sims[:, 2]), s=10, color="k",
              zorder=5, label="numerical integration")
    a.set_xlabel(r"incoming flux $\varphi$")
    a.set_ylabel(r"flux on bridge 1, $c_1$")
    a.set_title("(a) Equilibria of the trail model")
    a.legend(frameon=False, loc="upper left")
    delta = np.where(phis > 2, np.sqrt(np.clip(phis**2 - 4, 0, None)) / phis, 0.0)
    b.plot(phis, delta, color=ACCENT, lw=1.8)
    b.axvline(2, color=GRAY, ls=":", lw=1)
    b.text(2.08, 0.05, r"$\varphi_c = 2$", color=GRAY, fontsize=8)
    b.set_xlabel(r"incoming flux $\varphi$")
    b.set_ylabel(r"asymmetry $\Delta C$")
    b.set_ylim(-0.03, 1)
    b.set_title("(b) Emergence of a collective choice")
    save(fig, "collective-choice-flow")


# ---------------------------------------------------------- double bridge
def aco_bridge(n_ants=1000, runs=20, k=20.0, alpha=2.0, ratio=2.0, seed=5):
    """Ants cross one at a time; an ant on the long branch deposits 1/ratio."""
    rng = np.random.default_rng(seed)
    share = np.zeros((runs, n_ants))
    for r in range(runs):
        c = np.zeros(2)  # pheromone on (short, long)
        short_count = 0
        for t in range(n_ants):
            p_short = (k + c[0]) ** alpha / ((k + c[0]) ** alpha + (k + c[1]) ** alpha)
            if rng.random() < p_short:
                c[0] += 1.0
                short_count += 1
            else:
                c[1] += 1.0 / ratio
            share[r, t] = short_count / (t + 1)
    return share


def aco_bridge_figure():
    share = aco_bridge()
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    for s in share:
        ax.plot(np.arange(1, s.size + 1), s, color=PRIMARY, alpha=0.25, lw=0.8)
    ax.plot(np.arange(1, share.shape[1] + 1), share.mean(axis=0), color=ACCENT, lw=2,
            label="mean of 20 colonies")
    ax.axhline(0.5, color=GRAY, ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("number of ants that have crossed")
    ax.set_ylabel("share on the short branch")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "aco-path-selection")
    return share[:, -1]


# ---------------------------------------------------------- leader-follower
def leader_follower(T=40.0, dt=0.02, k_p=1.5, k_c=0.8):
    offsets = np.array([[0, 0], [-1, 1], [-1, -1], [-2, 2], [-2, 0], [-2, -2]], float)
    n = len(offsets)
    adjacency = np.zeros((n, n))
    for i, j in [(1, 0), (2, 0), (3, 1), (4, 1), (4, 2), (5, 2), (1, 2), (3, 4), (4, 5)]:
        adjacency[i, j] = adjacency[j, i] = 1
    adjacency[0, :] = 0  # the leader ignores the followers
    rng = np.random.default_rng(2)
    x = rng.uniform(-4, 1, (n, 2))
    x[0] = [0, 0]
    steps = int(T / dt)
    traj = np.zeros((steps, n, 2))
    prev_targets = None
    for s in range(steps):
        t = s * dt
        heading = np.array([1.0, 0.6 * np.cos(0.25 * t)])
        heading /= np.linalg.norm(heading)
        rot = np.array([[heading[0], -heading[1]], [heading[1], heading[0]]])
        x[0] = x[0] + dt * 0.5 * heading
        targets = x[0] + offsets @ rot.T
        u = np.zeros_like(x)
        for i in range(1, n):
            u[i] = k_p * (targets[i] - x[i]) if adjacency[i, 0] else 0.0
            for j in range(n):
                if adjacency[i, j] and j != 0:
                    u[i] += k_c * ((x[j] - targets[j]) - (x[i] - targets[i]))
            if prev_targets is not None:  # feed-forward of the slot velocity
                u[i] += (targets[i] - prev_targets[i]) / dt
        prev_targets = targets
        x[1:] += dt * u[1:]
        traj[s] = x
    err = np.linalg.norm(traj[-1] - (traj[-1, 0] + offsets @ rot.T), axis=1).max()
    return traj, offsets, err


def leader_follower_figure():
    traj, _, err = leader_follower()
    assert err < 0.05, err
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    for i in range(traj.shape[1]):
        color = ACCENT if i == 0 else PRIMARY
        ax.plot(traj[:, i, 0], traj[:, i, 1], color=color, lw=1.4 if i == 0 else 0.8,
                alpha=1 if i == 0 else 0.7)
    for s in [0, 300, 800, 1999]:
        pts = traj[s]
        ax.scatter(pts[1:, 0], pts[1:, 1], s=14, color=PRIMARY, zorder=4)
        ax.scatter(*pts[0], s=26, marker="D", color=ACCENT, zorder=5)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")
    ax.set_title("Leader (red) and five followers converging to a wedge formation")
    save(fig, "leader-follower")
    return err


if __name__ == "__main__":
    trail_bifurcation()
    finals = aco_bridge_figure()
    print("final short-branch share per colony:", np.round(finals, 2))
    print("leader-follower final formation error:", leader_follower_figure())
