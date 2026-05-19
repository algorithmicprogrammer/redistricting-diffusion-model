"""Small convex-QP tools for diffusion-style redistricting proposals.

This module intentionally has a SciPy implementation so the 5x5 starter
experiment runs without Gurobi. If Gurobi is available later, you can add a
backend while keeping the same API.
"""
from __future__ import annotations

from typing import Dict, Tuple

import networkx as nx
import numpy as np
from scipy.optimize import minimize

from .laplacian import build_laplacian
from .metrics import is_contiguous

Node = Tuple[int, int]
Assignment = Dict[Node, int]


def solve_pair_qp(
    graph: nx.Graph,
    assignment: Assignment,
    d1: int,
    d2: int,
    *,
    alpha: float = 20.0,
    beta: float = 1.0,
    epsilon: float = 0.0,
    attr: str = "density",
) -> tuple[list[Node], np.ndarray]:
    """Solve min alpha*x'Lx + beta*||x-x0||^2 for a district pair.

    For unit-population 5x5 fiber experiments, epsilon=0 forces exactly half of
    the merged pair into d1 when the merged pair has even size. For odd merged
    pairs, use epsilon > 0 or hard rounding separately.
    """
    nodes = [node for node, district in assignment.items() if district in {d1, d2}]
    L, _ = build_laplacian(graph, nodes, attr=attr)
    x0 = np.array([1.0 if assignment[node] == d1 else 0.0 for node in nodes], dtype=float)
    pop = np.array([float(graph.nodes[node].get("population", 1.0)) for node in nodes])
    target = pop.sum() / 2.0

    def objective(x: np.ndarray) -> float:
        diff = x - x0
        return float(alpha * x @ L @ x + beta * diff @ diff)

    def gradient(x: np.ndarray) -> np.ndarray:
        return 2.0 * alpha * (L @ x) + 2.0 * beta * (x - x0)

    constraints = [
        {"type": "ineq", "fun": lambda x: float(pop @ x - (1 - epsilon) * target),
         "jac": lambda x: pop},
        {"type": "ineq", "fun": lambda x: float((1 + epsilon) * target - pop @ x),
         "jac": lambda x: -pop},
    ]
    result = minimize(
        objective,
        x0,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(nodes),
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"QP solve failed: {result.message}")
    return nodes, np.asarray(result.x)


def hard_round_pair(nodes: list[Node], x_star: np.ndarray, assignment: Assignment, d1: int, d2: int) -> Assignment:
    """Round by selecting the highest x* values for district d1.

    This preserves the current district sizes for the selected pair.
    """
    new_assignment = dict(assignment)
    n_d1 = sum(1 for node in nodes if assignment[node] == d1)
    order = np.argsort(-x_star)
    d1_nodes = {nodes[i] for i in order[:n_d1]}
    for node in nodes:
        new_assignment[node] = d1 if node in d1_nodes else d2
    return new_assignment
