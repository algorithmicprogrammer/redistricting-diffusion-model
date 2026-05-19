"""Weighted graph Laplacian helpers."""
from __future__ import annotations

from typing import Iterable, Tuple

import networkx as nx
import numpy as np

Node = Tuple[int, int]


def build_laplacian(graph: nx.Graph, nodes: list[Node], attr: str = "density") -> tuple[np.ndarray, dict[Node, int]]:
    """Return L = D - W for an induced subgraph.

    The kernel follows the uploaded code's idea: adjacent nodes with similar
    attributes get larger weights, so the QP penalizes separating them.
    """
    idx = {node: i for i, node in enumerate(nodes)}
    W = np.zeros((len(nodes), len(nodes)), dtype=float)
    for u, v in graph.subgraph(nodes).edges:
        w = np.exp(-abs(float(graph.nodes[u].get(attr, 0)) - float(graph.nodes[v].get(attr, 0))))
        i, j = idx[u], idx[v]
        W[i, j] = W[j, i] = w
    L = np.diag(W.sum(axis=1)) - W
    return L, idx


def build_symmetric_normalized_laplacian(graph: nx.Graph, nodes: list[Node], attr: str = "density") -> tuple[np.ndarray, dict[Node, int]]:
    L, idx = build_laplacian(graph, nodes, attr=attr)
    degrees = np.diag(L).copy()
    inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
    D_inv = np.diag(inv_sqrt)
    return D_inv @ L @ D_inv, idx
