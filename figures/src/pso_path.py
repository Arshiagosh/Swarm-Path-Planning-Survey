"""PSO path planning: optimise the interior waypoints of a path around obstacles.

Each particle encodes N_WP free waypoints between a fixed start and goal; the
cost is path length plus a large penalty on the length of the path that runs
inside an (inflated) obstacle. The update is the inertia-weight PSO of
Eq. (pso) in the report, with a linearly decreasing inertia weight.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

from common import ACCENT, GRAY, PRIMARY, World, mark_start_goal, save

WORLD = World((10, 10), circles=[((3.0, 3.2), 1.3), ((6.2, 6.6), 1.5),
                                 ((7.6, 2.8), 1.0), ((3.2, 7.4), 0.9)])
START, GOAL = np.array([0.6, 0.8]), np.array([9.4, 9.2])
N_WP, MARGIN = 4, 0.2


def cost(flat):
    pts = np.vstack([START, flat.reshape(N_WP, 2), GOAL])
    seg = np.diff(pts, axis=0)
    length = np.sum(np.linalg.norm(seg, axis=1))
    penalty = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        for t in np.linspace(0, 1, 25):
            d = WORLD.clearance(a + t * (b - a))
            if d < MARGIN:
                penalty += (MARGIN - d)
    return length + 60.0 * penalty


def pso(n=40, iters=150, c1=1.5, c2=1.5, w0=0.9, w1=0.4, seed=1):
    rng = np.random.default_rng(seed)
    dim = 2 * N_WP
    x = rng.uniform(0, 10, (n, dim))
    v = rng.uniform(-1, 1, (n, dim))
    pbest, pcost = x.copy(), np.array([cost(p) for p in x])
    g = pbest[np.argmin(pcost)].copy()
    history = []
    for k in range(iters):
        w = w0 - (w0 - w1) * k / (iters - 1)
        r1, r2 = rng.random((n, dim)), rng.random((n, dim))
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (g - x)
        v = np.clip(v, -1.5, 1.5)
        x = np.clip(x + v, 0, 10)
        c = np.array([cost(p) for p in x])
        better = c < pcost
        pbest[better], pcost[better] = x[better], c[better]
        g = pbest[np.argmin(pcost)].copy()
        history.append(pcost.min())
    return g, np.array(history), x


def smooth(pts, n=200):
    t = np.linspace(0, 1, len(pts))
    spline = make_interp_spline(t, pts, k=3)
    return spline(np.linspace(0, 1, n))


def main():
    best, hist, swarm = pso()
    pts = np.vstack([START, best.reshape(N_WP, 2), GOAL])
    straight = np.linalg.norm(GOAL - START)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 3.1),
                                  gridspec_kw={"width_ratios": [1.05, 1]})
    WORLD.draw(ax)
    for p in swarm:
        q = p.reshape(N_WP, 2)
        ax.scatter(q[:, 0], q[:, 1], s=2, color=GRAY, alpha=0.5, zorder=2)
    ax.plot(pts[:, 0], pts[:, 1], "--", color=PRIMARY, lw=1, zorder=3,
            label="best waypoints")
    s = smooth(pts)
    ax.plot(s[:, 0], s[:, 1], color=ACCENT, lw=2, zorder=4, label="B-spline smoothed")
    ax.scatter(pts[1:-1, 0], pts[1:-1, 1], s=14, color=PRIMARY, zorder=5)
    mark_start_goal(ax, START, GOAL)
    ax.set_title("(a) Best path after 150 iterations")
    ax.legend(loc="center right", frameon=False, fontsize=7)
    ax2.plot(np.arange(1, len(hist) + 1), hist, color=PRIMARY)
    ax2.axhline(straight, color=GRAY, ls=":", lw=1)
    ax2.text(len(hist), straight, "straight-line\nlower bound", ha="right",
             va="bottom", fontsize=7, color=GRAY)
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("best cost (length + penalty)")
    ax2.set_ylim(straight * 0.95, min(hist[0], straight * 2.2))
    ax2.set_title(f"(b) Convergence, final cost {hist[-1]:.2f}")
    save(fig, "pso-path-planning")


if __name__ == "__main__":
    main()
