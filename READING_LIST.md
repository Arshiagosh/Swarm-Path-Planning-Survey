# Reading list: path planning for swarm robots

A short, opinionated guide to the sources behind the survey, grouped by topic. Every entry is cited in the report ([report/Swarm-Path-Planning-Survey.pdf](report/Swarm-Path-Planning-Survey.pdf)); the full list is in [report/references.bib](report/references.bib).

## Start here

| Source | Why read it |
|---|---|
| Brambilla, Ferrante, Birattari, Dorigo, *Swarm robotics: a review from the swarm engineering perspective*, Swarm Intelligence 7 (2013) | The standard map of swarm behaviors and design methods (behavior-based, automatic, PFSMs). |
| Dias et al., *Swarm Robotics: A Perspective on the Latest Reviewed Concepts and Applications*, Sensors 21 (2021) | A recent overview of concepts, platforms and simulators. |
| Lin, Liu, Wang, Kong, *A Review of Path-Planning Approaches for Multiple Mobile Robots*, Machines 10 (2022) | Classical, heuristic, bio-inspired and learning planners for multi-robot teams, with tables of methods. |
| AbuJabal et al., *Path Planning Techniques for Real-Time Multi-Robot Systems: A Systematic Review*, Electronics 13 (2024) | The deterministic / AI-based / hybrid taxonomy used in the report, and the 25% share of hybrid methods. |
| Tan & Zheng, *Research Advance in Swarm Robotics*, Defence Technology 9 (2013) | How swarm robotics differs from multi-robot systems, sensor networks and multi-agent systems. |

## Foundations of swarm intelligence

| Source | Why read it |
|---|---|
| Bonabeau, Dorigo, Theraulaz, *Swarm Intelligence: From Natural to Artificial Systems* (1999) | Stigmergy, ant foraging, division of labor, sorting and clustering; the book the field grew from. |
| Garnier, Gautrais, Theraulaz, *The biological principles of swarm intelligence*, Swarm Intelligence 1 (2007) | Positive and negative feedback, and the trail-choice bifurcation reproduced in Chapter 2. |
| Hamann, *Swarm Robotics: A Formal Approach* (2018) | Modeling swarms formally, from microscopic to macroscopic. |
| Bouffanais, *Design and Control of Swarm Dynamics* (2016) | The swarm signaling network and an information-theoretic view of swarm control. |
| Reynolds, *Flocks, herds and schools*, SIGGRAPH (1987) | Separation, alignment and cohesion. |

## Control and coordination

| Source | Why read it |
|---|---|
| Mesbahi & Egerstedt (2010); Bullo, Cortés, Martínez (2009) | Graph theory and distributed control of robotic networks. |
| Li & Duan, *Cooperative Control of Multi-Agent Systems* (2014) | Consensus, leader-follower tracking and containment. |
| Liu et al. (2018), two papers in Applied Sciences 8 | Virtual linkage formations and their decentralization through consensus. |
| Tan (ed.), *Handbook of Research on Design, Control, and Modeling of Swarm Robotics* (2016) | HASP, multilevel shape control, plume tracing, key predistribution and many more chapters. |
| Kolling et al., *Human Interaction With Robot Swarms: A Survey*, IEEE THMS 46 (2016) | Levels of automation, neglect benevolence, interfaces. |

## Bio-inspired planners

| Source | Why read it |
|---|---|
| Dorigo et al. (1996); Dorigo & Gambardella (1997) | Ant System and Ant Colony System. |
| Kennedy & Eberhart (1995); Poli, Kennedy, Blackwell (2007) | The original PSO and a later overview of its variants. |
| Panigrahi, Shi, Lim (eds.), *Handbook of Swarm Intelligence* (2011) | MOPSO, IPSO-DAI, parallel ABC, glowworms and controller tuning by ACO. |
| Karaboga (2005); Yang (2009, 2012); Tan & Zhu (2010); Krishnanand & Ghose (2009) | ABC, firefly, flower pollination, fireworks and glowworm algorithms. |

## Graph-, sampling- and field-based planners

| Source | Why read it |
|---|---|
| Hart, Nilsson, Raphael (1968); Stentz (1994); Koenig & Likhachev (2002) | A*, D* and D* Lite. |
| Kavraki et al. (1996); LaValle (1998); Kuffner & LaValle (2000); Karaman & Frazzoli (2011) | PRM, RRT, RRT-Connect and RRT*. |
| Khatib (1986) | Artificial potential fields. |
| Choset (2000); Acar, Choset, Lee (2006) | Boustrophedon and Morse decompositions, and GVD-based coverage. |
| van den Berg et al. (2011); Zhou et al. (2017) | ORCA and buffered Voronoi cells for reciprocal collision avoidance. |
| Mao, Fu, Quan (2023); Mao, Lv, Quan (2024) | Optimal virtual tubes and Tube RRT* for large swarms in clutter. |

## Learning-based planners

| Source | Why read it |
|---|---|
| Hüttenrauch, Šošić, Neumann, *Deep Reinforcement Learning for Swarm Systems*, JMLR 20 (2019) | Scaling deep RL to many agents. |
| Azzam, Boiko, Zweiri, Drones 7 (2023); Lowe et al. (2017) | Centralized training with decentralized execution. |
| Zhang et al., Complex & Intelligent Systems 11 (2025) | GAP_SAC: SAC with heuristic rewards, prioritized replay and gated attention. |
| Li & Tan, Neurocomputing (2019); Guo et al., Applied Sciences 11 (2021) | Imitation learning for multi-target search and for robust flocking. |
| Kuckling, Frontiers in Robotics and AI 10 (2023) | Recent trends in learning and evolution for swarms. |

## Applications and outlook

| Source | Why read it |
|---|---|
| Shahzad et al., *A Review of Swarm Robotics in a NutShell*, Drones 7 (2023) | Real projects: CoCoRo, Platypus, Swarmies, Marsbees, OFFSET, Kiva. |
| Bhowmick, Das, Arvin (eds.), *Bio-Inspired Swarm Robotics and Control* (2024) | Agriculture, search and rescue, construction and medical swarms. |
| Garcia-Aunon & Barrientos, Applied Sciences 8 (2018) | Search versus surveillance with aerial swarms, and how to measure them. |
| Kegeleirs & Birattari, Frontiers in Robotics and AI 12 (2025) | What still keeps swarms out of real deployments. |
