"""A* on an 8-connected occupancy grid, compared with Dijkstra (h = 0)."""

import heapq

import matplotlib.pyplot as plt
import numpy as np

from common import ACCENT, GRAY, LIGHT, PRIMARY, save

MOVES = [(dx, dy, np.hypot(dx, dy)) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
         if (dx, dy) != (0, 0)]


def make_grid():
    grid = np.zeros((20, 20), dtype=bool)
    grid[3:17, 9] = True      # long vertical wall
    grid[16, 3:10] = True     # bottom wall of a pocket
    grid[6, 12:18] = True     # shelf on the right
    grid[11:14, 13:15] = True  # small block
    return grid


def octile(a, b):
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dx, dy) + (np.sqrt(2) - 1) * min(dx, dy)


def search(grid, start, goal, heuristic):
    """Return (path, set of expanded cells)."""
    g = {start: 0.0}
    parent = {start: None}
    frontier = [(heuristic(start, goal), start)]
    closed = set()
    while frontier:
        _, node = heapq.heappop(frontier)
        if node in closed:
            continue
        closed.add(node)
        if node == goal:
            break
        for dx, dy, cost in MOVES:
            nxt = (node[0] + dx, node[1] + dy)
            if not (0 <= nxt[0] < grid.shape[0] and 0 <= nxt[1] < grid.shape[1]):
                continue
            if grid[nxt]:
                continue
            # no corner cutting through obstacles
            if dx and dy and (grid[node[0] + dx, node[1]] or grid[node[0], node[1] + dy]):
                continue
            new_g = g[node] + cost
            if new_g < g.get(nxt, np.inf):
                g[nxt] = new_g
                parent[nxt] = node
                heapq.heappush(frontier, (new_g + heuristic(nxt, goal), nxt))
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = parent[node]
    return path[::-1], closed, g[goal]


def draw(ax, grid, path, closed, title):
    img = np.ones(grid.shape + (3,))
    light = np.array([int(LIGHT[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    for c in closed:
        img[c] = light * 0.92
    img[grid] = 0.62
    ax.imshow(img.transpose(1, 0, 2), origin="lower", interpolation="nearest")
    p = np.array(path)
    ax.plot(p[:, 0], p[:, 1], color=ACCENT, lw=2, zorder=3)
    ax.scatter(*path[0], color=PRIMARY, s=28, zorder=4)
    ax.scatter(*path[-1], color=ACCENT, marker="*", s=60, zorder=4)
    ax.set_xticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.4)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    for side in ax.spines.values():
        side.set_visible(True)
        side.set_color(GRAY)
    ax.set_title(title)


def main():
    grid = make_grid()
    start, goal = (3, 10), (17, 3)
    path_a, closed_a, cost_a = search(grid, start, goal, octile)
    path_d, closed_d, cost_d = search(grid, start, goal, lambda a, b: 0.0)
    assert abs(cost_a - cost_d) < 1e-9, "A* with an admissible heuristic must match Dijkstra"
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.2))
    draw(axes[0], grid, path_d, closed_d,
         f"Dijkstra: {len(closed_d)} cells expanded")
    draw(axes[1], grid, path_a, closed_a,
         f"A* (octile heuristic): {len(closed_a)} cells expanded")
    fig.text(0.5, 0.02, f"Both find the same optimal path (cost {cost_a:.2f}); "
             "shaded cells were expanded by the search.",
             ha="center", fontsize=8, color=GRAY)
    save(fig, "astar-grid")


if __name__ == "__main__":
    main()
