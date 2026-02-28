"""
metrics.py — Redistricting quality measures.

Functions
---------
n_cut_edges(graph, assignment)
    Number of edges crossing district boundaries.

border_pairs_set(assignment, graph)
    Set of adjacent district-id pairs that share at least one cut edge.

polsby_popper(graph, assignment)
    Mean Polsby-Popper compactness score over all districts.

mean_pop_deviation(graph, assignment)
    Mean absolute fractional population deviation from the ideal district size.

density_score(graph, assignment)
    Fraction of total kernel weight that lies within districts.

    Formula
    -------
        w(e) = exp(-|Δdensity(e)|)

        DS(x) = Σ_{e intra-district} w(e)
               ─────────────────────────────
               Σ_{e all edges}      w(e)

    Range [0, 1].  Higher = districts better aligned with the density kernel.
    DS = 1  when all high-weight (same-density) edges are intra-district.

record_metrics(graph, assignment)
    Returns dict with keys: density_score, pp, pop_dev.
"""

import numpy as np


def n_cut_edges(graph, assignment):
    return sum(1 for u, v in graph.edges() if assignment[u] != assignment[v])


def border_pairs_set(assignment, graph):
    bp = set()
    for u, v in graph.edges():
        d1, d2 = assignment[u], assignment[v]
        if d1 != d2:
            bp.add((min(d1, d2), max(d1, d2)))
    return bp


def polsby_popper(graph, assignment):
    scores = []
    for d in set(assignment.values()):
        nodes_d  = [v for v, dist in assignment.items() if dist == d]
        area     = len(nodes_d)
        internal = graph.subgraph(nodes_d).number_of_edges()
        perim    = 4 * area - 2 * internal
        if perim > 0:
            scores.append(4 * np.pi * area / perim**2)
    return float(np.mean(scores)) if scores else 0.0


def mean_pop_deviation(graph, assignment):
    districts = set(assignment.values())
    pops  = {d: sum(graph.nodes[v]["population"]
                    for v in graph.nodes() if assignment[v] == d)
             for d in districts}
    ideal = sum(pops.values()) / len(districts)
    return float(np.mean([abs(pop - ideal) / ideal for pop in pops.values()]))


def density_score(graph, assignment):
    intra_w = 0.0
    total_w = 0.0
    for u, v in graph.edges():
        w = np.exp(-abs(graph.nodes[u].get("density", 0)
                        - graph.nodes[v].get("density", 0)))
        total_w += w
        if assignment[u] == assignment[v]:
            intra_w += w
    return intra_w / total_w if total_w > 0 else 0.0


def record_metrics(graph, assignment):
    return {
        "density_score": density_score(graph, assignment),
        "pp":            polsby_popper(graph, assignment),
        "pop_dev":       mean_pop_deviation(graph, assignment),
    }
