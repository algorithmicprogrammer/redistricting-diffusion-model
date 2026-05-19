"""Metrics and validity checks for grid districting plans."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Tuple

import networkx as nx
import numpy as np

Node = Tuple[int, int]
Assignment = Dict[Node, int]


def districts(assignment: Assignment) -> list[int]:
    return sorted(set(assignment.values()))


def nodes_by_district(assignment: Assignment) -> dict[int, list[Node]]:
    out: dict[int, list[Node]] = defaultdict(list)
    for node, district in assignment.items():
        out[district].append(node)
    return dict(out)


def cut_edges(graph: nx.Graph, assignment: Assignment) -> list[tuple[Node, Node]]:
    return [(u, v) for u, v in graph.edges if assignment[u] != assignment[v]]


def n_cut_edges(graph: nx.Graph, assignment: Assignment) -> int:
    return len(cut_edges(graph, assignment))


def boundary_nodes(graph: nx.Graph, assignment: Assignment) -> dict[int, set[Node]]:
    out = {d: set() for d in districts(assignment)}
    for u, v in cut_edges(graph, assignment):
        out[assignment[u]].add(u)
        out[assignment[v]].add(v)
    return out


def district_populations(graph: nx.Graph, assignment: Assignment, pop_col: str = "population") -> dict[int, float]:
    pops = {d: 0.0 for d in districts(assignment)}
    for node, district in assignment.items():
        pops[district] += float(graph.nodes[node].get(pop_col, 1.0))
    return pops


def is_contiguous(graph: nx.Graph, assignment: Assignment) -> bool:
    for _, nodes in nodes_by_district(assignment).items():
        if not nodes or not nx.is_connected(graph.subgraph(nodes)):
            return False
    return True


def is_balanced_by_size(assignment: Assignment, district_size: int) -> bool:
    return all(len(nodes) == district_size for nodes in nodes_by_district(assignment).values())


def polsby_popper_grid(graph: nx.Graph, assignment: Assignment) -> float:
    """Mean Polsby-Popper-style compactness on a unit-square grid.

    Area is node count. Perimeter is approximated by exposed unit sides:
    4*area - 2*internal_edges.
    """
    scores = []
    for nodes in nodes_by_district(assignment).values():
        sub = graph.subgraph(nodes)
        area = len(nodes)
        perim = 4 * area - 2 * sub.number_of_edges()
        if perim > 0:
            scores.append(4 * np.pi * area / (perim**2))
    return float(np.mean(scores)) if scores else 0.0


def density_score(graph: nx.Graph, assignment: Assignment, attr: str = "density") -> float:
    intra = 0.0
    total = 0.0
    for u, v in graph.edges:
        w = np.exp(-abs(float(graph.nodes[u].get(attr, 0)) - float(graph.nodes[v].get(attr, 0))))
        total += w
        if assignment[u] == assignment[v]:
            intra += w
    return float(intra / total) if total else 0.0


def record_metrics(graph: nx.Graph, assignment: Assignment) -> dict[str, float]:
    pops = district_populations(graph, assignment)
    ideal = sum(pops.values()) / len(pops)
    pop_dev = float(np.mean([abs(p - ideal) / ideal for p in pops.values()])) if ideal else 0.0
    return {
        "cut_edges": float(n_cut_edges(graph, assignment)),
        "polsby_popper": polsby_popper_grid(graph, assignment),
        "density_score": density_score(graph, assignment),
        "pop_dev": pop_dev,
        "contiguous": float(is_contiguous(graph, assignment)),
    }
