"""Sampling and exact enumeration of a small 5x5 redistricting fiber.

Fiber used here: all plans on a grid graph with ``num_districts`` connected
pieces, each containing exactly ``district_size`` unit-population nodes.

For a 5x5 grid with 5 districts of size 5, exact enumeration is feasible.
The exact enumerator returns one canonical representative per unlabeled plan;
that is, two assignments that differ only by district-name permutations are
counted once.
"""
from __future__ import annotations

import itertools
import random
from typing import Dict, Iterable, Iterator, Tuple

import networkx as nx

from .metrics import is_balanced_by_size, is_contiguous

Node = Tuple[int, int]
Assignment = Dict[Node, int]


def canonical_key(assignment: Assignment) -> tuple[tuple[Node, int], ...]:
    return tuple(sorted(assignment.items()))


def canonicalize_assignment(assignment: Assignment) -> Assignment:
    """Relabel district ids by first appearance in sorted node order.

    This removes duplicate copies of the same geometric partition caused only
    by arbitrary district labels.
    """
    label_map: dict[int, int] = {}
    next_label = 0
    out: Assignment = {}
    for node in sorted(assignment):
        old = assignment[node]
        if old not in label_map:
            label_map[old] = next_label
            next_label += 1
        out[node] = label_map[old]
    return out


def canonical_unlabeled_key(assignment: Assignment) -> tuple[tuple[Node, int], ...]:
    return canonical_key(canonicalize_assignment(assignment))


def valid_fiber_state(graph: nx.Graph, assignment: Assignment, district_size: int) -> bool:
    return is_balanced_by_size(assignment, district_size) and is_contiguous(graph, assignment)


def propose_connected_swap(graph: nx.Graph, assignment: Assignment, rng: random.Random) -> Assignment | None:
    """Swap labels of two adjacent boundary nodes from different districts.

    Swaps preserve district sizes exactly; we then reject if either affected
    district becomes disconnected.
    """
    boundary_edges = [(u, v) for u, v in graph.edges if assignment[u] != assignment[v]]
    if not boundary_edges:
        return None
    rng.shuffle(boundary_edges)
    for u, v in boundary_edges:
        if rng.random() < 0.5:
            u, v = v, u
        d_u, d_v = assignment[u], assignment[v]
        proposal = dict(assignment)
        proposal[u], proposal[v] = d_v, d_u
        affected = {d_u, d_v}
        ok = True
        for d in affected:
            nodes_d = [node for node, dist in proposal.items() if dist == d]
            if not nodes_d or not nx.is_connected(graph.subgraph(nodes_d)):
                ok = False
                break
        if ok:
            return proposal
    return None


def _connected_subsets_of_size(subgraph: nx.Graph, size: int) -> list[set[Node]]:
    """Enumerate connected subsets of a small graph."""
    nodes = list(subgraph.nodes)
    out: list[set[Node]] = []
    for combo in itertools.combinations(nodes, size):
        subset = set(combo)
        if nx.is_connected(subgraph.subgraph(subset)):
            out.append(subset)
    return out


def propose_pair_resplit(graph: nx.Graph, assignment: Assignment, district_size: int, rng: random.Random) -> Assignment | None:
    """Re-split two adjacent districts into two connected equal-size pieces."""
    pairs = sorted({tuple(sorted((assignment[u], assignment[v]))) for u, v in graph.edges if assignment[u] != assignment[v]})
    if not pairs:
        return None
    rng.shuffle(pairs)
    for d1, d2 in pairs:
        merged_nodes = [node for node, d in assignment.items() if d in {d1, d2}]
        sub = graph.subgraph(merged_nodes)
        candidates = _connected_subsets_of_size(sub, district_size)
        rng.shuffle(candidates)
        for subset in candidates:
            complement = set(merged_nodes) - subset
            if not complement or not nx.is_connected(sub.subgraph(complement)):
                continue
            proposal = dict(assignment)
            # Randomize which connected side gets which district label.
            if rng.random() < 0.5:
                d_subset, d_complement = d1, d2
            else:
                d_subset, d_complement = d2, d1
            for node in subset:
                proposal[node] = d_subset
            for node in complement:
                proposal[node] = d_complement
            if proposal != assignment:
                return proposal
    return None


def sample_fiber(
    graph: nx.Graph,
    initial_assignment: Assignment,
    *,
    district_size: int = 5,
    n_samples: int = 36,
    burn_in: int = 200,
    thinning: int = 20,
    seed: int = 42,
) -> list[Assignment]:
    """Sample valid fiber states by connected label swaps and pair re-splits."""
    if not valid_fiber_state(graph, initial_assignment, district_size):
        raise ValueError("Initial assignment is not in the requested fiber.")

    rng = random.Random(seed)
    current = dict(initial_assignment)
    samples: list[Assignment] = []
    seen = set()
    total_steps = burn_in + n_samples * thinning * 10
    for step in range(total_steps):
        proposal = propose_pair_resplit(graph, current, district_size, rng)
        if proposal is None:
            proposal = propose_connected_swap(graph, current, rng)
        if proposal is not None:
            current = proposal
        if step >= burn_in and (step - burn_in) % thinning == 0:
            key = canonical_unlabeled_key(current)
            if key not in seen:
                samples.append(canonicalize_assignment(current))
                seen.add(key)
                if len(samples) >= n_samples:
                    break
    return samples




def _assignment_from_masks(masks: list[int], nodes: list[Node]) -> Assignment:
    assignment: Assignment = {}
    for district, mask in enumerate(masks):
        for i, node in enumerate(nodes):
            if mask & (1 << i):
                assignment[node] = district
    return assignment


def enumerate_full_fiber(
    graph: nx.Graph,
    *,
    district_size: int = 5,
    num_districts: int = 5,
    max_plans: int | None = None,
) -> list[Assignment]:
    """Exactly enumerate the connected equal-size fiber on a small graph.

    Returns canonical unlabeled assignments. To avoid generating district-label
    permutations, each recursive step anchors the next district at the smallest
    still-unassigned node and assigns district labels in construction order.

    For the 5x5 / five-district case this uses bit masks, so full enumeration
    is fast enough to run as part of the experiment script.
    """
    nodes = sorted(graph.nodes)
    n = len(nodes)
    expected_nodes = district_size * num_districts
    if n != expected_nodes:
        raise ValueError(
            f"Expected {expected_nodes} nodes from district_size*num_districts, got {n}."
        )
    if n > 60:
        raise ValueError("Bit-mask exact enumeration is intended for small graphs only.")

    node_to_i = {node: i for i, node in enumerate(nodes)}
    adjacency_masks = [0] * n
    for u, v in graph.edges:
        i, j = node_to_i[u], node_to_i[v]
        adjacency_masks[i] |= 1 << j
        adjacency_masks[j] |= 1 << i

    def lowbit_index(mask: int) -> int:
        return (mask & -mask).bit_length() - 1

    def mask_is_connected(mask: int) -> bool:
        start = lowbit_index(mask)
        seen = 0
        stack = [start]
        while stack:
            i = stack.pop()
            bit = 1 << i
            if seen & bit:
                continue
            seen |= bit
            nbrs = adjacency_masks[i] & mask & ~seen
            while nbrs:
                b = nbrs & -nbrs
                nbrs ^= b
                stack.append(b.bit_length() - 1)
        return seen == mask

    # Precompute every connected district-shaped subset once.
    subset_masks: list[int] = []
    subsets_by_anchor: list[list[int]] = [[] for _ in range(n)]
    for combo in itertools.combinations(range(n), district_size):
        mask = 0
        for i in combo:
            mask |= 1 << i
        if mask_is_connected(mask):
            subset_masks.append(mask)
            for i in combo:
                subsets_by_anchor[i].append(mask)

    def remaining_components_are_tileable(mask: int) -> bool:
        """Necessary condition: no district can cross disconnected components."""
        remaining = mask
        while remaining:
            start = lowbit_index(remaining)
            comp = 0
            stack = [start]
            while stack:
                i = stack.pop()
                bit = 1 << i
                if comp & bit:
                    continue
                comp |= bit
                nbrs = adjacency_masks[i] & mask & ~comp
                while nbrs:
                    b = nbrs & -nbrs
                    nbrs ^= b
                    stack.append(b.bit_length() - 1)
            if comp.bit_count() % district_size != 0:
                return False
            remaining &= ~comp
        return True

    full_mask = (1 << n) - 1
    plan_masks: list[list[int]] = []

    def recurse(available: int, parts: list[int]) -> None:
        if max_plans is not None and len(plan_masks) >= max_plans:
            return
        if len(parts) == num_districts:
            if available == 0:
                plan_masks.append(parts.copy())
            return

        # Symmetry break: the smallest unassigned node belongs to the next
        # district, so permutations of district labels are not duplicated.
        anchor = lowbit_index(available)
        for subset in subsets_by_anchor[anchor]:
            if subset & available != subset:
                continue
            remaining = available ^ subset
            if not remaining_components_are_tileable(remaining):
                continue
            parts.append(subset)
            recurse(remaining, parts)
            parts.pop()

    recurse(full_mask, [])
    return [_assignment_from_masks(masks, nodes) for masks in plan_masks]
