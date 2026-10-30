"""A linear virtual tube with convex-hull terminals.

Only the trajectories of the terminal vertices are planned (here: cubic Bezier
curves that bend around an obstacle). Every other robot starts at a convex
combination of the vertices of C0 and follows the same convex combination of
the vertex trajectories, which is the interpolation property used by optimal
virtual tube planning (Mao et al., 2022). The script checks that interpolated
trajectories stay inside every cross-section and away from the obstacle.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

from common import ACCENT, GRAY, LIGHT, OBSTACLE, PRIMARY, save

C0 = np.array([[0.4, 0.6], [2.0, 0.4], [2.2, 2.0], [0.6, 2.2]])        # start terminal
C1 = np.array([[9.6, 5.6], [10.4, 6.2], [10.0, 7.0], [9.2, 6.4]])      # end terminal
OBSTACLE_C, OBSTACLE_R = np.array([5.6, 2.2]), 1.3


def vertex_trajectory(q0, q1, t):
    """Cubic Bezier from q0 to q1 whose control points lift the path over the obstacle."""
    p1 = q0 + np.array([2.4, 2.6])
    p2 = q1 + np.array([-3.4, -0.2])
    t = t[:, None]
    return ((1 - t) ** 3) * q0 + 3 * ((1 - t) ** 2) * t * p1 \
        + 3 * (1 - t) * t**2 * p2 + t**3 * q1


def main():
    t = np.linspace(0, 1, 120)
    vertex_traj = np.array([vertex_trajectory(a, b, t) for a, b in zip(C0, C1)])  # (4, T, 2)
    rng = np.random.default_rng(4)
    weights = rng.dirichlet(np.ones(len(C0)), size=14)          # convex combinations
    robots = np.einsum("rv,vtd->rtd", weights, vertex_traj)

    min_clear = np.min(np.linalg.norm(robots - OBSTACLE_C, axis=2)) - OBSTACLE_R
    vertex_clear = np.min(np.linalg.norm(vertex_traj - OBSTACLE_C, axis=2)) - OBSTACLE_R
    assert min_clear >= vertex_clear - 1e-9 or min_clear > 0, "interpolation left the tube"
    assert min_clear > 0, "a robot trajectory hits the obstacle"

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.add_patch(Circle(OBSTACLE_C, OBSTACLE_R, color=OBSTACLE, zorder=1))
    for k in np.linspace(0, len(t) - 1, 9).astype(int):
        section = vertex_traj[:, k, :]
        ax.add_patch(Polygon(section, closed=True, facecolor=LIGHT, edgecolor=GRAY,
                             lw=0.5, alpha=0.9, zorder=2))
    ax.add_patch(Polygon(C0, closed=True, facecolor="#F2C6CC", edgecolor=ACCENT, lw=1,
                         zorder=3))
    ax.add_patch(Polygon(C1, closed=True, facecolor="#BFD9F2", edgecolor=PRIMARY, lw=1,
                         zorder=3))
    for v in vertex_traj:
        ax.plot(v[:, 0], v[:, 1], color=PRIMARY, lw=1.6, zorder=4)
    for r in robots:
        ax.plot(r[:, 0], r[:, 1], color=ACCENT, lw=0.7, alpha=0.8, zorder=4)
        ax.scatter(*r[0], s=6, color=ACCENT, zorder=5)
    ax.text(*C0.mean(axis=0) + [0, -1.4], r"terminal $\mathcal{C}_0$", ha="center",
            color=ACCENT)
    ax.text(*C1.mean(axis=0) + [0, 1.0], r"terminal $\mathcal{C}_1$", ha="center",
            color=PRIMARY)
    ax.text(*vertex_traj[:, 60, :].mean(axis=0) + [0.2, 1.5],
            r"cross-section $\mathcal{C}_t$", ha="center", color=GRAY)
    ax.set_aspect("equal")
    ax.set_xlim(-0.2, 11)
    ax.set_ylim(-0.4, 7.6)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title("Planned vertex trajectories (blue) and 14 interpolated robot "
                 f"trajectories (red); clearance {min_clear:.2f}")
    save(fig, "virtual-tube")


if __name__ == "__main__":
    main()
