"""Shared style, geometry helpers and output paths for the figure scripts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

# Figures are written straight into the report's figure folder.
OUT = Path(__file__).resolve().parents[2] / "report" / "figures"

# Colours taken from the report (IUST blue and red) plus two neutrals.
PRIMARY = "#1B2A6B"
ACCENT = "#B5122B"
LIGHT = "#EEF1F8"
GRAY = "#6B7280"
OBSTACLE = "#9CA3AF"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def save(fig, name):
    """Save a figure as a vector PDF in report/figures."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.pdf"
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(f"wrote {path.relative_to(OUT.parents[1])}")


class World:
    """A 2-D workspace with circular and axis-aligned rectangular obstacles."""

    def __init__(self, size, circles=(), rects=()):
        self.size = np.asarray(size, dtype=float)
        self.circles = [(np.asarray(c, float), float(r)) for c, r in circles]
        self.rects = [np.asarray(r, float) for r in rects]  # (xmin, ymin, xmax, ymax)

    def clearance(self, p):
        """Signed distance from point p to the nearest obstacle (negative inside)."""
        p = np.asarray(p, float)
        d = np.inf
        for c, r in self.circles:
            d = min(d, np.linalg.norm(p - c) - r)
        for x0, y0, x1, y1 in self.rects:
            dx = max(x0 - p[0], 0.0, p[0] - x1)
            dy = max(y0 - p[1], 0.0, p[1] - y1)
            if dx == 0.0 and dy == 0.0:
                d = min(d, -min(p[0] - x0, x1 - p[0], p[1] - y0, y1 - p[1]))
            else:
                d = min(d, np.hypot(dx, dy))
        return d

    def free(self, p, margin=0.0):
        inside = np.all(p >= 0) and np.all(p <= self.size)
        return inside and self.clearance(p) > margin

    def segment_free(self, a, b, margin=0.0, step=0.05):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)))
        return all(self.free(a + t * (b - a), margin) for t in np.linspace(0, 1, n))

    def draw(self, ax, color=OBSTACLE):
        for c, r in self.circles:
            ax.add_patch(Circle(c, r, facecolor=color, edgecolor="none", zorder=1))
        for x0, y0, x1, y1 in self.rects:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   facecolor=color, edgecolor="none", zorder=1))
        ax.set_xlim(0, self.size[0])
        ax.set_ylim(0, self.size[1])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color(GRAY)
            ax.spines[side].set_linewidth(0.6)


def mark_start_goal(ax, start, goal, size=30):
    ax.scatter(*start, s=size, marker="o", color=PRIMARY, zorder=5, label="start")
    ax.scatter(*goal, s=size * 1.6, marker="*", color=ACCENT, zorder=5, label="goal")
