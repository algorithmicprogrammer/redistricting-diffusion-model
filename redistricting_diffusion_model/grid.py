"""Grid construction utilities for small redistricting diffusion experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, Tuple

import networkx as nx

Node = Tuple[int, int]
Assignment = Dict[Node, int]


@dataclass(frozen=True)
class GridConfig:
    width: int = 5
    height: int = 5
    num_districts: int = 5
    population: int = 1
    seed: int = 42


def make_grid(config: GridConfig | None = None) -> nx.Graph:
    """Create a rectangular grid graph with population and boundary metadata."""
    cfg = config or GridConfig()
    graph = nx.grid_graph([cfg.width, cfg.height])
    for node in graph.nodes:
        x, y = node
        graph.nodes[node]["population"] = cfg.population
        graph.nodes[node]["boundary_node"] = x in {0, cfg.width - 1} or y in {0, cfg.height - 1}
        graph.nodes[node]["boundary_perim"] = int(graph.nodes[node]["boundary_node"])
        # A simple synthetic density field useful for weighted Laplacians.
        graph.nodes[node]["density"] = 100 * (x // max(1, cfg.width // cfg.num_districts))
    return graph


def vertical_stripes(graph: nx.Graph, num_districts: int) -> Assignment:
    """Initial assignment: vertical stripe districts.

    On a 5x5 grid with 5 districts, this gives five connected districts of 5
    nodes each. It is the smallest clean analogue of the larger stripe setup in
    the uploaded notebooks/scripts.
    """
    xs = sorted({node[0] for node in graph.nodes})
    width = len(xs)
    stripe_width = max(1, width // num_districts)
    assignment: Assignment = {}
    for x, y in graph.nodes:
        d = min(num_districts - 1, x // stripe_width)
        assignment[(x, y)] = d
    return assignment


def horizontal_stripes(graph: nx.Graph, num_districts: int) -> Assignment:
    ys = sorted({node[1] for node in graph.nodes})
    height = len(ys)
    stripe_height = max(1, height // num_districts)
    return {(x, y): min(num_districts - 1, y // stripe_height) for x, y in graph.nodes}
