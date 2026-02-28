"""
Grid setup for redistricting diffusion model experiments.

Two partitions are generated:

  reference_assignment  — produced by recursive_tree_part (matching grid.ipynb).
                          Used ONLY to assign the density attribute to each node.
                          Not used as a chain starting state.

  initial_assignment    — a second independent recursive_tree_part draw.
                          This is the actual starting state for all Markov chains
                          (ReCom, QP-MH, MEW, Hybrid).

Density kernel
--------------
    district d  →  density = d × 100   (d = 0 … NUM_DISTRICTS-1)
Kernel weight:  w(e) = exp(−|Δdensity|)

Exports
-------
graph                networkx.Graph    — used for all custom QP / MEW logic
gc_graph             gerrychain.Graph  — used for Partition / ReCom
reference_assignment dict {node → district_id}   (density-setup partition)
initial_assignment   dict {node → district_id}   (chain starting state)
pop_target           float
NUM_DISTRICTS        int
ns                   int    (node draw size for matplotlib)
plot_boundary_nodes  callable(assignment, ax, title)
"""

import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from gerrychain import Graph
from gerrychain.tree import recursive_tree_part

# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

gn            = 6      # sub-grid side length
k             = 5      # number of districts
ns            = 50     # matplotlib node draw size
NUM_DISTRICTS = k

# ─── networkx graph ───────────────────────────────────────────────────────────
graph = nx.grid_graph([k * gn, k * gn])

for n in graph.nodes():
    graph.nodes[n]["population"] = 50
    graph.nodes[n]["density"]    = 0      # placeholder; updated after reference partition
    if 0 in n or k * gn - 1 in n:
        graph.nodes[n]["boundary_node"]  = True
        graph.nodes[n]["boundary_perim"] = 1
    else:
        graph.nodes[n]["boundary_node"] = False

total_pop  = sum(graph.nodes[v]["population"] for v in graph.nodes())
pop_target = total_pop / NUM_DISTRICTS

# ─── gerrychain graph ─────────────────────────────────────────────────────────
gc_graph = Graph.from_networkx(graph)

# ─── reference partition (grid.ipynb scheme) — density assignment only ────────
# Matches the recursive_tree_part call from grid.ipynb / main.ipynb.
# This partition is NOT the chain starting state; it only defines which density
# level each node belongs to.
reference_assignment = recursive_tree_part(
    graph      = gc_graph,
    parts      = range(NUM_DISTRICTS),
    pop_target = pop_target,
    pop_col    = "population",
    epsilon    = 0.1,
)

# ─── density kernel: district d → density = d × 100 ─────────────────────────
# Matches grid.ipynb (districts 0–3: 0, 100, 200, 300) extended to k districts.
# Nodes in the same reference district share a density value; the kernel
#   w(e) = exp(−|Δdensity|)
# strongly weights intra-district edges and down-weights cross-district edges,
# giving the QP solver a meaningful geographic compactness signal.
_density_map = {d: d * 100 for d in range(NUM_DISTRICTS)}

for node, dist in reference_assignment.items():
    density = _density_map[dist]
    graph.nodes[node]["density"]    = density
    gc_graph.nodes[node]["density"] = density

# ─── initial partition — actual chain starting state ─────────────────────────
# A second independent draw; the random state has advanced since the reference
# partition, so this produces a genuinely different partition.
initial_assignment = recursive_tree_part(
    graph      = gc_graph,
    parts      = range(NUM_DISTRICTS),
    pop_target = pop_target,
    pop_col    = "population",
    epsilon    = 0.1,
)


# ─────────────────────────────────────────────────────────────────────────────
# Boundary-node visualisation
# Replicates the grid.ipynb style: grey full graph + coloured boundary nodes.
# ─────────────────────────────────────────────────────────────────────────────

def plot_boundary_nodes(assignment, ax, title="Boundary Nodes by District"):
    """
    Draw the grid in light grey; overlay boundary nodes per district in colour.

    Parameters
    ----------
    assignment : dict {node → district_id}
    ax         : matplotlib Axes
    title      : str
    """
    pos = {node: node for node in graph.nodes()}

    # Base layer: full graph in grey
    nx.draw(
        graph, pos=pos,
        node_color="lightgray",
        node_size=ns,
        node_shape="s",
        edge_color="gray",
        alpha=0.3,
        with_labels=False,
        ax=ax,
    )

    # Collect boundary nodes per district (nodes incident to any cut edge)
    district_ids = set(assignment.values())
    district_boundaries = {d: set() for d in district_ids}
    for u, v in graph.edges():
        d_u = assignment.get(u)
        d_v = assignment.get(v)
        if d_u != d_v:
            if d_u is not None:
                district_boundaries[d_u].add(u)
            if d_v is not None:
                district_boundaries[d_v].add(v)

    colors = plt.cm.tab10.colors
    for d_id in sorted(district_boundaries):
        b_nodes = district_boundaries[d_id]
        if b_nodes:
            nx.draw_networkx_nodes(
                graph, pos=pos,
                nodelist=list(b_nodes),
                node_color=[colors[d_id % len(colors)]],
                node_size=100,
                node_shape="s",
                label=f"D_{d_id}",
                ax=ax,
            )

    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title, fontsize=11)
    try:
        ax.set_aspect("equal")
    except NotImplementedError:
        pass
