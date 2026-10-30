"""Regenerate every data figure of the report into report/figures/.

    python figures/src/make_figures.py
"""

import astar_grid
import potential_field
import pso_path
import sampling_planners
import swarm_models
import virtual_tube
import voronoi_roadmap


def main():
    astar_grid.main()
    sampling_planners.main()
    voronoi_roadmap.main()
    potential_field.main()
    pso_path.main()
    swarm_models.trail_bifurcation()
    swarm_models.aco_bridge_figure()
    swarm_models.leader_follower_figure()
    virtual_tube.main()


if __name__ == "__main__":
    main()
