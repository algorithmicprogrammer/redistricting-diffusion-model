"""Plotting helpers for fiber and QP visualizations."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .metrics import boundary_nodes, n_cut_edges, polsby_popper_grid, record_metrics

Node = Tuple[int, int]
Assignment = Dict[Node, int]


def draw_partition(graph: nx.Graph, assignment: Assignment, ax, title: str = "Partition", node_size: int = 500):
    pos = {node: node for node in graph.nodes}
    nx.draw(
        graph,
        pos=pos,
        node_color=[assignment[node] for node in graph.nodes],
        node_size=node_size,
        node_shape="s",
        cmap="tab20",
        edge_color="white",
        linewidths=0.5,
        with_labels=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_boundary_nodes(graph: nx.Graph, assignment: Assignment, ax, title: str = "Boundary nodes"):
    pos = {node: node for node in graph.nodes}
    nx.draw(graph, pos=pos, node_color="lightgray", node_size=420, node_shape="s", edge_color="gray", alpha=0.35, ax=ax)
    bnodes = boundary_nodes(graph, assignment)
    colors = plt.cm.tab10.colors
    for d, nodes in sorted(bnodes.items()):
        if nodes:
            nx.draw_networkx_nodes(graph, pos=pos, nodelist=list(nodes), node_color=[colors[d % len(colors)]], node_size=520, node_shape="s", label=f"D{d}", ax=ax)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(fontsize=8, loc="upper right")


def plot_fiber_gallery(graph: nx.Graph, samples: list[Assignment], path: Path, ncols: int = 6):
    n = len(samples)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.4 * nrows))
    axes_arr = np.array(axes).reshape(-1)
    for i, (ax, asn) in enumerate(zip(axes_arr, samples)):
        draw_partition(graph, asn, ax, title=f"#{i+1}: cut={n_cut_edges(graph, asn)}", node_size=140)
    for ax in axes_arr[n:]:
        ax.axis("off")
    fig.suptitle("Sampled fiber: connected 5x5 plans with five districts of size 5", y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_metric_projection(graph: nx.Graph, samples: list[Assignment], path: Path):
    cuts = [n_cut_edges(graph, asn) for asn in samples]
    pps = [polsby_popper_grid(graph, asn) for asn in samples]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(cuts, pps)
    ax.set_xlabel("cut edges / discrete perimeter")
    ax.set_ylabel("mean grid Polsby-Popper")
    ax.set_title("Fiber projection by compactness metrics")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_fiber_gallery_pages(
    graph: nx.Graph,
    samples: list[Assignment],
    out_dir: Path,
    *,
    prefix: str = "full_fiber_gallery",
    ncols: int = 6,
    nrows: int = 6,
) -> list[Path]:
    """Write a multi-page PNG gallery for a large fiber.

    A single image containing the full fiber can become unreadable, so this
    creates numbered gallery pages such as ``full_fiber_gallery_001.png``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    per_page = ncols * nrows
    paths: list[Path] = []
    for page_start in range(0, len(samples), per_page):
        page_samples = samples[page_start : page_start + per_page]
        page = page_start // per_page + 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.2 * nrows))
        axes_arr = np.array(axes).reshape(-1)
        for offset, (ax, asn) in enumerate(zip(axes_arr, page_samples)):
            idx = page_start + offset + 1
            draw_partition(graph, asn, ax, title=f"#{idx}: cut={n_cut_edges(graph, asn)}", node_size=110)
        for ax in axes_arr[len(page_samples):]:
            ax.axis("off")
        fig.suptitle(f"Full fiber gallery, page {page}", y=1.01, fontsize=14)
        fig.tight_layout()
        path = out_dir / f"{prefix}_{page:03d}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
