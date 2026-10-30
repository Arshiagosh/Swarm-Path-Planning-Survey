"""Artificial potential field: the force field, a successful run and a local minimum."""

import matplotlib.pyplot as plt
import numpy as np

from common import ACCENT, PRIMARY, World, mark_start_goal, save

K_ATT, K_REP, D0 = 1.0, 2.5, 1.5


def force(world, p, goal):
    """F = -grad(U_att + U_rep) with a quadratic attractive and Khatib's repulsive term."""
    f = -K_ATT * (p - goal)
    for c, r in world.circles:
        diff = p - c
        dist = np.linalg.norm(diff)
        d = dist - r
        if 1e-6 < d < D0:
            f += K_REP * (1 / d - 1 / D0) / d**2 * diff / dist
    for x0, y0, x1, y1 in world.rects:
        q = np.array([np.clip(p[0], x0, x1), np.clip(p[1], y0, y1)])
        diff = p - q
        d = np.linalg.norm(diff)
        if 1e-6 < d < D0:
            f += K_REP * (1 / d - 1 / D0) / d**2 * diff / d
    return f


def descend(world, start, goal, step=0.02, n=4000):
    p = np.array(start, float)
    path = [p.copy()]
    for _ in range(n):
        f = force(world, p, goal)
        nf = np.linalg.norm(f)
        if nf < 1e-3 or np.linalg.norm(p - goal) < 0.05:
            break
        p = p + step * f / max(nf, 1.0)
        path.append(p.copy())
    return np.array(path)


def panel(ax, world, start, goal, title):
    world.draw(ax)
    xs, ys = np.meshgrid(np.linspace(0.3, 9.7, 22), np.linspace(0.3, 9.7, 22))
    u, v = np.full_like(xs, np.nan), np.full_like(ys, np.nan)
    for i in range(xs.shape[0]):
        for j in range(xs.shape[1]):
            p = np.array([xs[i, j], ys[i, j]])
            if world.clearance(p) > 0.1:
                f = force(world, p, goal)
                f /= max(np.linalg.norm(f), 1e-9)
                u[i, j], v[i, j] = f
    ax.quiver(xs, ys, u, v, color=PRIMARY, alpha=0.45, scale=30, width=0.004)
    path = descend(world, start, goal)
    ax.plot(path[:, 0], path[:, 1], color=ACCENT, lw=2, zorder=4)
    mark_start_goal(ax, start, goal)
    ax.set_title(title)
    return path


def main():
    goal = np.array([8.5, 8.0])
    open_world = World((10, 10), circles=[((4.5, 4.8), 1.0), ((7.0, 5.5), 0.7)])
    trap = World((10, 10), rects=[(5.0, 3.0, 5.6, 8.0), (3.0, 7.4, 5.6, 8.0),
                                  (3.0, 3.0, 5.6, 3.6)])
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.3))
    p1 = panel(axes[0], open_world, np.array([1.0, 1.5]), goal,
               "(a) Goal reached")
    p2 = panel(axes[1], trap, np.array([1.0, 5.5]), goal,
               "(b) Trapped in a local minimum")
    assert np.linalg.norm(p1[-1] - goal) < 0.1
    assert np.linalg.norm(p2[-1] - goal) > 1.0
    save(fig, "apf-forces")


if __name__ == "__main__":
    main()
