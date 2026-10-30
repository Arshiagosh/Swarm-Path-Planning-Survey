"""Generalized Voronoi diagram as a maximum-clearance roadmap through narrow passages.

The GVD is approximated in the usual way: sample points on the obstacle and
workspace boundaries, build their ordinary Voronoi diagram, and keep the edges
that lie in free space and separate samples of two *different* obstacles.
"""

import heapq

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi

from common import ACCENT, PRIMARY, World, mark_start_goal, save

WORLD = World(
    size=(10, 6),
    rects=[(2.0, 0.0, 3.0, 2.6), (2.0, 3.4, 3.0, 6.0),   # wall with a narrow door
           (6.0, 1.2, 7.0, 4.8)],                        # block in the second room
    circles=[((4.6, 4.4), 0.6)],
)


def boundary_samples(step=0.12):
    pts, owner = [], []
    for k, (x0, y0, x1, y1) in enumerate(WORLD.rects):
        for x in np.arange(x0, x1 + 1e-9, step):
            pts += [(x, y0), (x, y1)]
            owner += [k, k]
        for y in np.arange(y0, y1 + 1e-9, step):
            pts += [(x0, y), (x1, y)]
            owner += [k, k]
    base = len(WORLD.rects)
    for k, (c, r) in enumerate(WORLD.circles):
        for a in np.linspace(0, 2 * np.pi, int(2 * np.pi * r / step), endpoint=False):
            pts.append(tuple(c + r * np.array([np.cos(a), np.sin(a)])))
            owner.append(base + k)
    w, h = WORLD.size
    walls = base + len(WORLD.circles)
    for x in np.arange(0, w + 1e-9, step):
        pts += [(x, 0.0), (x, h)]
        owner += [walls, walls + 1]
    for y in np.arange(0, h + 1e-9, step):
        pts += [(0.0, y), (w, y)]
        owner += [walls + 2, walls + 3]
    return np.array(pts), np.array(owner)


def gvd_edges():
    pts, owner = boundary_samples()
    vor = Voronoi(pts)
    edges = []
    for (i, j), (a, b) in zip(vor.ridge_points, vor.ridge_vertices):
        if a < 0 or b < 0 or owner[i] == owner[j]:
            continue
        pa, pb = vor.vertices[a], vor.vertices[b]
        if WORLD.free(pa, 0.05) and WORLD.free(pb, 0.05):
            edges.append((a, b))
    return vor.vertices, edges


def shortest_on_graph(vertices, edges, start, goal):
    used = sorted({v for e in edges for v in e})
    adj = {v: [] for v in used}
    for a, b in edges:
        w = np.linalg.norm(vertices[a] - vertices[b])
        adj[a].append((b, w))
        adj[b].append((a, w))
    near = lambda p: min(used, key=lambda v: np.linalg.norm(vertices[v] - p))
    s, g = near(start), near(goal)
    dist, parent, pq = {s: 0.0}, {s: None}, [(0.0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == g:
            break
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            if d + w < dist.get(v, np.inf):
                dist[v], parent[v] = d + w, u
                heapq.heappush(pq, (d + w, v))
    path, u = [], g
    while u is not None:
        path.append(vertices[u])
        u = parent[u]
    return np.array([start] + path[::-1] + [goal])


def main():
    vertices, edges = gvd_edges()
    start, goal = np.array([0.8, 1.0]), np.array([9.2, 5.0])
    path = shortest_on_graph(vertices, edges, start, goal)
    clear = min(WORLD.clearance(p) for p in path[1:-1])
    fig, ax = plt.subplots(figsize=(6.0, 3.7))
    WORLD.draw(ax)
    for a, b in edges:
        ax.plot(*np.array([vertices[a], vertices[b]]).T, color=PRIMARY, lw=0.8,
                alpha=0.75, zorder=2)
    ax.plot(path[:, 0], path[:, 1], color=ACCENT, lw=2.2, zorder=4)
    mark_start_goal(ax, start, goal)
    ax.set_title("GVD roadmap (blue) and the path along it (red); "
                 f"minimum clearance on the roadmap path {clear:.2f}")
    save(fig, "voronoi-narrow-regions")


if __name__ == "__main__":
    main()
