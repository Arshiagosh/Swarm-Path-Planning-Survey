"""PRM, RRT and RRT-Connect in the same cluttered 2-D workspace."""

import heapq

import matplotlib.pyplot as plt
import numpy as np

from common import ACCENT, GRAY, PRIMARY, World, mark_start_goal, save

WORLD = World(
    size=(10, 10),
    circles=[((2.6, 6.8), 1.0), ((7.4, 3.0), 1.1), ((5.0, 8.4), 0.7)],
    rects=[(4.2, 0.0, 5.0, 6.2), (6.2, 5.6, 9.2, 6.4), (1.0, 2.6, 3.4, 3.4)],
)
START, GOAL = np.array([1.0, 1.0]), np.array([9.0, 9.0])
MARGIN = 0.15


# ---------------------------------------------------------------- PRM
def prm(n_samples=220, k=10, seed=3):
    rng = np.random.default_rng(seed)
    nodes = [START, GOAL]
    while len(nodes) < n_samples + 2:
        p = rng.uniform(0, 10, 2)
        if WORLD.free(p, MARGIN):
            nodes.append(p)
    nodes = np.array(nodes)
    edges = {i: [] for i in range(len(nodes))}
    for i, p in enumerate(nodes):
        d = np.linalg.norm(nodes - p, axis=1)
        for j in np.argsort(d)[1:k + 1]:
            if j in [e for e, _ in edges[i]]:
                continue
            if WORLD.segment_free(p, nodes[j], MARGIN):
                edges[i].append((j, d[j]))
                edges[j].append((i, d[j]))
    # Dijkstra from START (index 0) to GOAL (index 1)
    dist, parent, pq = {0: 0.0}, {0: None}, [(0.0, 0)]
    while pq:
        du, u = heapq.heappop(pq)
        if u == 1:
            break
        if du > dist[u]:
            continue
        for v, w in edges[u]:
            if du + w < dist.get(v, np.inf):
                dist[v], parent[v] = du + w, u
                heapq.heappush(pq, (du + w, v))
    path, u = [], 1
    while u is not None:
        path.append(nodes[u])
        u = parent[u]
    return nodes, edges, np.array(path[::-1]), dist[1]


# ---------------------------------------------------------------- RRT
def steer(a, b, step):
    d = b - a
    n = np.linalg.norm(d)
    return b if n <= step else a + d / n * step


def extend(tree, parents, target, step):
    i = int(np.argmin(np.linalg.norm(np.array(tree) - target, axis=1)))
    new = steer(tree[i], target, step)
    if WORLD.free(new, MARGIN) and WORLD.segment_free(tree[i], new, MARGIN):
        tree.append(new)
        parents.append(i)
        return len(tree) - 1
    return None


def trace(tree, parents, i):
    out = []
    while i is not None and i >= 0:
        out.append(tree[i])
        i = parents[i]
    return out


def rrt(step=0.5, goal_bias=0.05, seed=7, max_iter=5000):
    rng = np.random.default_rng(seed)
    tree, parents = [START], [-1]
    for it in range(1, max_iter + 1):
        target = GOAL if rng.random() < goal_bias else rng.uniform(0, 10, 2)
        i = extend(tree, parents, target, step)
        if i is not None and np.linalg.norm(tree[i] - GOAL) < step \
                and WORLD.segment_free(tree[i], GOAL, MARGIN):
            tree.append(GOAL)
            parents.append(i)
            return tree, parents, np.array(trace(tree, parents, len(tree) - 1)[::-1]), it
    raise RuntimeError("RRT did not reach the goal")


def rrt_connect(step=0.5, seed=7, max_iter=5000):
    rng = np.random.default_rng(seed)
    ta, pa, tb, pb = [START], [-1], [GOAL], [-1]
    swapped = False
    for it in range(1, max_iter + 1):
        q = rng.uniform(0, 10, 2)
        i = extend(ta, pa, q, step)
        if i is not None:
            # CONNECT: greedily grow the other tree towards the new node
            while True:
                j = extend(tb, pb, ta[i], step)
                if j is None:
                    break
                if np.allclose(tb[j], ta[i]):
                    a, b = trace(ta, pa, i), trace(tb, pb, j)
                    path = (a[::-1] + b) if not swapped else (b[::-1] + a)
                    trees = (ta, pa, tb, pb) if not swapped else (tb, pb, ta, pa)
                    return trees, np.array(path), it
        ta, pa, tb, pb = tb, pb, ta, pa
        swapped = not swapped
    raise RuntimeError("RRT-Connect did not connect")


def length(path):
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def draw_tree(ax, tree, parents, color):
    for i, p in enumerate(parents):
        if p >= 0:
            ax.plot([tree[p][0], tree[i][0]], [tree[p][1], tree[i][1]],
                    color=color, lw=0.5, alpha=0.7, zorder=2)


def main():
    nodes, edges, path, cost = prm()
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    WORLD.draw(ax)
    for i, nbrs in edges.items():
        for j, _ in nbrs:
            if i < j:
                ax.plot(*np.array([nodes[i], nodes[j]]).T, color=GRAY, lw=0.35,
                        alpha=0.55, zorder=2)
    ax.scatter(nodes[:, 0], nodes[:, 1], s=3, color=PRIMARY, zorder=3)
    ax.plot(path[:, 0], path[:, 1], color=ACCENT, lw=2, zorder=4)
    mark_start_goal(ax, START, GOAL)
    ax.set_title(f"PRM: {len(nodes)} milestones, path length {cost:.1f}")
    save(fig, "prm-example")

    tree, parents, p_rrt, it_rrt = rrt()
    (ta, pa, tb, pb), p_con, it_con = rrt_connect()
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.3))
    WORLD.draw(axes[0])
    draw_tree(axes[0], tree, parents, PRIMARY)
    axes[0].plot(p_rrt[:, 0], p_rrt[:, 1], color=ACCENT, lw=2, zorder=4)
    mark_start_goal(axes[0], START, GOAL)
    axes[0].set_title(f"RRT: {len(tree)} nodes, {it_rrt} iterations\n"
                      f"path length {length(p_rrt):.1f}")
    WORLD.draw(axes[1])
    draw_tree(axes[1], ta, pa, PRIMARY)
    draw_tree(axes[1], tb, pb, "#2E8B57")
    axes[1].plot(p_con[:, 0], p_con[:, 1], color=ACCENT, lw=2, zorder=4)
    mark_start_goal(axes[1], START, GOAL)
    axes[1].set_title(f"RRT-Connect: {len(ta) + len(tb)} nodes, {it_con} iterations\n"
                      f"path length {length(p_con):.1f}")
    save(fig, "rrt-vs-rrt-connect")


if __name__ == "__main__":
    main()
