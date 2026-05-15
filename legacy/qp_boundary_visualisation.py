"""
QP Boundary Movement Visualisation
====================================
Shows exactly what happens inside a single QP proposal step:

Panel layout (one figure per step, saved as a sequence)
─────────────────────────────────────────────────────────
 A  │  B  │  C  │  D  │  E
────┼─────┼─────┼─────┼─────
 Before   │ x*  │Prob │After │ Δ
partition │field│map  │part  │boundary

A. Full partition before the step  (all districts, pair D1/D2 highlighted)
B. The merged subgraph coloured by x* (continuous [0,1] field from QP)
C. Rounding probability map  p_v = sigmoid((x*-0.5)/T)
D. Full partition after the step
E. Node-level difference: which nodes switched district

Also produces:
  • boundary_animation.png  — 4×N grid showing N consecutive steps
  • fiedler_vs_qp.png       — compares Fiedler vector to QP solution x*
"""

import random, warnings
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import gurobipy as gp
from gurobipy import GRB

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Graph + initial partition
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED)

gn = 6; k = 5; ns = 60
graph = nx.grid_graph([k * gn, k * gn])

for n in graph.nodes():
    graph.nodes[n]["population"] = 50
    if random.random() < 0.5:
        graph.nodes[n]["pink"] = 1;  graph.nodes[n]["purple"] = 0
    else:
        graph.nodes[n]["pink"] = 0;  graph.nodes[n]["purple"] = 1
    if 0 in n or k * gn - 1 in n:
        graph.nodes[n]["boundary_node"]  = True
        graph.nodes[n]["boundary_perim"] = 1
    else:
        graph.nodes[n]["boundary_node"] = False

cddict = {x: int(x[0] / gn) for x in graph.nodes()}
pos    = {x: x for x in graph.nodes()}           # node positions = grid coords

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Core functions
# ─────────────────────────────────────────────────────────────────────────────

def build_sym_laplacian(graph, nodes):
    n   = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    sub = graph.subgraph(nodes)
    W   = np.zeros((n, n))
    for u, v in sub.edges():
        w = np.exp(-abs(graph.nodes[u].get("population", 0)
                        - graph.nodes[v].get("population", 0)))
        i, j = idx[u], idx[v]; W[i, j] = w; W[j, i] = w
    d       = W.sum(axis=1)
    d_isqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_isqrt = np.diag(d_isqrt)
    L_sym   = D_isqrt @ (np.diag(d) - W) @ D_isqrt
    return L_sym, idx


def solve_qp(L_sym, x0, pop, p_bar, alpha, beta, epsilon=0.10):
    n = len(x0)
    try:
        m = gp.Model(); m.setParam("OutputFlag", 0)
        x = m.addVars(n, lb=0.0, ub=1.0)
        obj = gp.QuadExpr()
        for i in range(n):
            for j in range(i, n):
                val = L_sym[i, j]
                if abs(val) > 1e-12:
                    obj += (alpha * val * x[i] * x[i] if i == j
                            else 2 * alpha * val * x[i] * x[j])
        for i in range(n):
            obj += beta * (x[i] * x[i] - 2.0 * x0[i] * x[i])
        m.setObjective(obj, GRB.MINIMIZE)
        pe = gp.LinExpr(pop.tolist(), [x[i] for i in range(n)])
        m.addConstr(pe >= (1 - epsilon) * p_bar)
        m.addConstr(pe <= (1 + epsilon) * p_bar)
        m.optimize()
        if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            return np.array([x[i].X for i in range(n)])
    except gp.GurobiError:
        pass
    return None


def sigmoid_round(x_star, T=0.05, max_tries=20,
                  graph=None, nodes=None, asn=None, d1_id=None, d2_id=None):
    """Sigmoid-temperature rounding with connectivity check."""
    n     = len(x_star)
    probs = 1.0 / (1.0 + np.exp(-(x_star - 0.5) / T))
    probs = np.clip(probs, 1e-9, 1 - 1e-9)
    for _ in range(max_tries):
        bits    = np.random.rand(n) < probs
        new_asn = dict(asn)
        for i, v in enumerate(nodes):
            new_asn[v] = d1_id if bits[i] else d2_id
        sub1 = graph.subgraph([v for v, d in new_asn.items() if d == d1_id])
        sub2 = graph.subgraph([v for v, d in new_asn.items() if d == d2_id])
        if nx.is_connected(sub1) and nx.is_connected(sub2):
            return new_asn, probs, bits
    return None, probs, None


def border_pairs(asn, graph):
    bp = set()
    for u, v in graph.edges():
        d1, d2 = asn[u], asn[v]
        if d1 != d2:
            bp.add((min(d1, d2), max(d1, d2)))
    return bp


def n_cut_edges(graph, asn):
    return sum(1 for u, v in graph.edges() if asn[u] != asn[v])


def boundary_edges(graph, asn, d1_id, d2_id):
    """Edges crossing the D1/D2 boundary."""
    return [(u, v) for u, v in graph.edges()
            if {asn[u], asn[v]} == {d1_id, d2_id}]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Colour helpers
# ─────────────────────────────────────────────────────────────────────────────
TAB20   = plt.cm.tab20
N_DIST  = 5

def district_cmap():
    """Fixed colours for districts 0-4."""
    return {d: TAB20(d / 10) for d in range(N_DIST)}

DCOL = district_cmap()

def node_colors_full(asn, highlight_pair=None, alpha_other=0.25):
    """
    Return colour list for all nodes.
    Nodes NOT in highlight_pair are faded.
    """
    colors = []
    for v in graph.nodes():
        c = list(mcolors.to_rgba(DCOL[asn[v]]))
        if highlight_pair and asn[v] not in highlight_pair:
            c[3] = alpha_other
        colors.append(tuple(c))
    return colors


def draw_boundary_edges(ax, graph, asn, d1_id, d2_id,
                        color="black", lw=2.5, style="solid"):
    be = boundary_edges(graph, asn, d1_id, d2_id)
    nx.draw_networkx_edges(graph, pos=pos, edgelist=be,
                           edge_color=color, width=lw,
                           style=style, ax=ax)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Single-step boundary visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualise_qp_step(asn_before, step_num,
                      alpha=20.0, beta=1.0, T_round=0.05, epsilon=0.10,
                      fixed_pair=None):
    """
    Produce a 5-panel figure showing one QP boundary-move step.
    Returns (asn_after, success).
    """
    # ── choose district pair ──────────────────────────────────────────────────
    bp = border_pairs(asn_before, graph)
    if not bp:
        return asn_before, False

    if fixed_pair and fixed_pair in bp:
        d1_id, d2_id = fixed_pair
    else:
        d1_id, d2_id = random.choice(sorted(bp))

    nodes   = [v for v, d in asn_before.items() if d in (d1_id, d2_id)]
    n_nodes = len(nodes)
    node_set = set(nodes)

    # ── QP ───────────────────────────────────────────────────────────────────
    L_sym, idx = build_sym_laplacian(graph, nodes)
    x0  = np.array([1.0 if asn_before[v] == d1_id else 0.0 for v in nodes])
    pop = np.array([graph.nodes[v]["population"] for v in nodes])
    p_bar = pop.sum() / 2.0

    x_star = solve_qp(L_sym, x0, pop, p_bar, alpha, beta, epsilon)
    if x_star is None:
        return asn_before, False

    # ── Fiedler vector (for comparison panel) ────────────────────────────────
    import scipy.linalg as la
    eigs, evecs = la.eigh(L_sym)
    fiedler = evecs[:, 1]
    # Align sign so fiedler correlates with x0
    if np.corrcoef(fiedler, x0)[0, 1] < 0:
        fiedler = -fiedler
    # Normalise to [0,1]
    f_min, f_max = fiedler.min(), fiedler.max()
    fiedler_01 = (fiedler - f_min) / (f_max - f_min + 1e-12)

    # ── Rounding ──────────────────────────────────────────────────────────────
    asn_after, probs, bits = sigmoid_round(
        x_star, T=T_round, graph=graph, nodes=nodes,
        asn=asn_before, d1_id=d1_id, d2_id=d2_id)

    if asn_after is None:
        return asn_before, False

    # ── Nodes that switched ───────────────────────────────────────────────────
    switched = [v for v in nodes if asn_before[v] != asn_after[v]]
    d2_to_d1 = [v for v in switched if asn_after[v] == d1_id]   # boundary moved out
    d1_to_d2 = [v for v in switched if asn_after[v] == d2_id]

    cut_before = n_cut_edges(graph, asn_before)
    cut_after  = n_cut_edges(graph, asn_after)

    # ── Build node → x* / prob maps ──────────────────────────────────────────
    xstar_map   = {v: x_star[i]   for i, v in enumerate(nodes)}
    prob_map    = {v: probs[i]     for i, v in enumerate(nodes)}
    fiedler_map = {v: fiedler_01[i] for i, v in enumerate(nodes)}
    x0_map      = {v: x0[i]       for i, v in enumerate(nodes)}

    # ─────────────────────────────────────────────────────────────────────────
    # Figure layout: 1 row, 5 panels
    # ─────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(30, 7))
    fig.suptitle(
        f"Step {step_num}  │  Districts ({d1_id}, {d2_id})  │  "
        f"Merged: {n_nodes} nodes  │  "
        f"Cut edges: {cut_before} → {cut_after}  (Δ={cut_after-cut_before:+d})  │  "
        f"Switched nodes: {len(switched)} "
        f"({len(d2_to_d1)}→D{d1_id}, {len(d1_to_d2)}→D{d2_id})",
        fontsize=13, fontweight="bold", y=1.01
    )

    # ── A: Partition BEFORE ───────────────────────────────────────────────────
    ax = axes[0]
    nc = node_colors_full(asn_before, highlight_pair={d1_id, d2_id})
    nx.draw(graph, pos=pos, node_color=nc, node_size=ns,
            node_shape="s", ax=ax, with_labels=False)
    draw_boundary_edges(ax, graph, asn_before, d1_id, d2_id,
                        color="black", lw=2.5)
    ax.set_title(f"A.  Partition BEFORE\n"
                 f"D{d1_id} (light) / D{d2_id} (dark) highlighted", fontsize=11)

    # ── B: x* continuous field ────────────────────────────────────────────────
    ax = axes[1]
    # Background: grey for non-pair nodes
    nc_bg = ["#cccccc"] * len(graph.nodes())
    all_nodes_list = list(graph.nodes())
    node_order = {v: i for i, v in enumerate(all_nodes_list)}
    x_star_colors = []
    cmap_field = plt.cm.RdBu_r
    for v in graph.nodes():
        if v in node_set:
            x_star_colors.append(cmap_field(xstar_map[v]))
        else:
            x_star_colors.append((0.85, 0.85, 0.85, 0.4))

    nx.draw(graph, pos=pos, node_color=x_star_colors, node_size=ns,
            node_shape="s", ax=ax, with_labels=False)
    # Overlay old boundary
    draw_boundary_edges(ax, graph, asn_before, d1_id, d2_id,
                        color="black", lw=1.5, style="dashed")

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap_field,
                               norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label="x* (0=D2, 1=D1)")
    ax.set_title(f"B.  QP solution x*\n"
                 f"(α={alpha}, β={beta}, ε={epsilon})\n"
                 f"Dashed = old boundary", fontsize=11)

    # ── C: Rounding probability map ───────────────────────────────────────────
    ax = axes[2]
    cmap_prob = plt.cm.PiYG
    prob_colors = []
    for v in graph.nodes():
        if v in node_set:
            prob_colors.append(cmap_prob(prob_map[v]))
        else:
            prob_colors.append((0.85, 0.85, 0.85, 0.4))

    nx.draw(graph, pos=pos, node_color=prob_colors, node_size=ns,
            node_shape="s", ax=ax, with_labels=False)
    draw_boundary_edges(ax, graph, asn_before, d1_id, d2_id,
                        color="black", lw=1.5, style="dashed")

    sm2 = plt.cm.ScalarMappable(cmap=cmap_prob,
                                norm=mcolors.Normalize(vmin=0, vmax=1))
    sm2.set_array([])
    plt.colorbar(sm2, ax=ax, fraction=0.03, pad=0.02,
                 label=f"P(→D{d1_id})  [T={T_round}]")
    ax.set_title(f"C.  Sigmoid rounding probability\n"
                 f"p = σ((x*−0.5)/{T_round})\n"
                 f"Pink>0.5 → D{d1_id},  Green<0.5 → D{d2_id}", fontsize=11)

    # ── D: Partition AFTER ────────────────────────────────────────────────────
    ax = axes[3]
    nc_after = node_colors_full(asn_after, highlight_pair={d1_id, d2_id})
    nx.draw(graph, pos=pos, node_color=nc_after, node_size=ns,
            node_shape="s", ax=ax, with_labels=False)
    draw_boundary_edges(ax, graph, asn_after, d1_id, d2_id,
                        color="black", lw=2.5)
    # Show old boundary as dashed
    draw_boundary_edges(ax, graph, asn_before, d1_id, d2_id,
                        color="grey", lw=1.0, style="dashed")
    ax.set_title(f"D.  Partition AFTER\n"
                 f"Solid = new boundary, dashed = old boundary", fontsize=11)

    # ── E: Switched nodes ─────────────────────────────────────────────────────
    ax = axes[4]
    # Base: faded full partition
    nc_diff = []
    for v in graph.nodes():
        if v in set(d2_to_d1):
            nc_diff.append("red")         # gained by D1
        elif v in set(d1_to_d2):
            nc_diff.append("blue")        # lost by D1
        elif v in node_set:
            c = list(mcolors.to_rgba(DCOL[asn_after[v]]));  c[3] = 0.35
            nc_diff.append(tuple(c))
        else:
            nc_diff.append((0.85, 0.85, 0.85, 0.25))

    nx.draw(graph, pos=pos, node_color=nc_diff, node_size=ns,
            node_shape="s", ax=ax, with_labels=False)
    draw_boundary_edges(ax, graph, asn_after, d1_id, d2_id,
                        color="black", lw=2.5)
    draw_boundary_edges(ax, graph, asn_before, d1_id, d2_id,
                        color="grey", lw=1.0, style="dashed")

    legend_els = [
        mpatches.Patch(color="red",  label=f"→ D{d1_id}  ({len(d2_to_d1)} nodes)"),
        mpatches.Patch(color="blue", label=f"→ D{d2_id}  ({len(d1_to_d2)} nodes)"),
        mpatches.Patch(color="grey", alpha=0.5, label="unchanged"),
    ]
    ax.legend(handles=legend_els, loc="upper right", fontsize=9,
              framealpha=0.9)
    ax.set_title(f"E.  Boundary movement\n"
                 f"Red = flipped to D{d1_id},  Blue = flipped to D{d2_id}\n"
                 f"Total switched: {len(switched)}", fontsize=11)

    plt.tight_layout()
    fname = f"qp_step_{step_num:02d}.png"
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}  │  cut {cut_before}→{cut_after}  "
          f"│  switched {len(switched)} nodes")

    return asn_after, True


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Fiedler vs QP comparison (one-off diagnostic)
# ─────────────────────────────────────────────────────────────────────────────

def visualise_fiedler_vs_qp(asn, d1_id, d2_id,
                             alpha=20.0, beta=1.0, epsilon=0.10):
    """
    Side-by-side: Fiedler vector vs x* vs hard-round of each.
    Helps show that large α/β makes QP ≈ spectral bisection.
    """
    import scipy.linalg as la

    nodes   = [v for v, d in asn.items() if d in (d1_id, d2_id)]
    n_nodes = len(nodes)
    L_sym, idx = build_sym_laplacian(graph, nodes)
    x0  = np.array([1.0 if asn[v] == d1_id else 0.0 for v in nodes])
    pop = np.array([graph.nodes[v]["population"] for v in nodes])
    p_bar = pop.sum() / 2.0

    # Fiedler
    eigs, evecs = la.eigh(L_sym)
    fiedler = evecs[:, 1]
    if np.corrcoef(fiedler, x0)[0, 1] < 0:
        fiedler = -fiedler
    fiedler_01 = (fiedler - fiedler.min()) / (fiedler.max() - fiedler.min() + 1e-12)

    # QP solutions for different alpha/beta
    configs = [
        (1,  10, "α=1, β=10\n(default — prior dominates)"),
        (5,  2,  "α=5, β=2"),
        (20, 1,  "α=20, β=1\n(spectral bisection regime)"),
        (50, 1,  "α=50, β=1\n(very close to Fiedler)"),
    ]

    fig, axes = plt.subplots(2, len(configs) + 1, figsize=(7*(len(configs)+1), 12))
    cmap_f = plt.cm.RdBu_r

    # ── Row 0: continuous fields ───────────────────────────────────────────────
    node_set = set(nodes)

    def _field_colors(field_map):
        cols = []
        for v in graph.nodes():
            if v in field_map:
                cols.append(cmap_f(field_map[v]))
            else:
                cols.append((0.85, 0.85, 0.85, 0.3))
        return cols

    # Fiedler
    ax = axes[0][0]
    fm = {v: fiedler_01[i] for i, v in enumerate(nodes)}
    nx.draw(graph, pos=pos, node_color=_field_colors(fm),
            node_size=ns, node_shape="s", ax=ax, with_labels=False)
    draw_boundary_edges(ax, graph, asn, d1_id, d2_id, color="k", lw=2)
    ax.set_title(f"Fiedler vector (λ₁={eigs[1]:.4f})\n"
                 f"= optimal spectral bisection", fontsize=11, fontweight="bold")
    sm = plt.cm.ScalarMappable(cmap=cmap_f, norm=mcolors.Normalize(0, 1))
    sm.set_array([]); plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)

    for col_i, (al, be, title) in enumerate(configs):
        xs = solve_qp(L_sym, x0, pop, p_bar, al, be, epsilon)
        ax = axes[0][col_i + 1]
        if xs is not None:
            xm = {v: xs[i] for i, v in enumerate(nodes)}
            nx.draw(graph, pos=pos, node_color=_field_colors(xm),
                    node_size=ns, node_shape="s", ax=ax, with_labels=False)
            corr = np.corrcoef(xs, fiedler_01)[0, 1]
            ax.set_title(f"QP: {title}\ncorr(x*, Fiedler)={corr:.3f}", fontsize=10)
        else:
            ax.set_title(f"QP: {title}\n[infeasible]", fontsize=10)
        draw_boundary_edges(ax, graph, asn, d1_id, d2_id, color="k", lw=1.5,
                            style="dashed")
        plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)

    # ── Row 1: hard-rounded partitions ────────────────────────────────────────
    def _round_hard(field_map, nodes, asn, d1_id, d2_id):
        new_asn = dict(asn)
        for v in nodes:
            new_asn[v] = d1_id if field_map.get(v, 0) >= 0.5 else d2_id
        return new_asn

    def _dist_colors(asn_r, highlight_pair):
        return node_colors_full(asn_r, highlight_pair=highlight_pair)

    ax = axes[1][0]
    asn_f = _round_hard(fm, nodes, asn, d1_id, d2_id)
    nc_f  = _dist_colors(asn_f, {d1_id, d2_id})
    nx.draw(graph, pos=pos, node_color=nc_f, node_size=ns,
            node_shape="s", ax=ax, with_labels=False)
    draw_boundary_edges(ax, graph, asn_f, d1_id, d2_id, color="k", lw=2)
    draw_boundary_edges(ax, graph, asn, d1_id, d2_id, color="grey", lw=1, style="dashed")
    ax.set_title(f"Fiedler rounded\ncut={n_cut_edges(graph,asn_f)}  "
                 f"(orig={n_cut_edges(graph,asn)})", fontsize=10)

    for col_i, (al, be, title) in enumerate(configs):
        xs = solve_qp(L_sym, x0, pop, p_bar, al, be, epsilon)
        ax = axes[1][col_i + 1]
        if xs is not None:
            xm   = {v: xs[i] for i, v in enumerate(nodes)}
            asn_r = _round_hard(xm, nodes, asn, d1_id, d2_id)
            nc_r  = _dist_colors(asn_r, {d1_id, d2_id})
            nx.draw(graph, pos=pos, node_color=nc_r, node_size=ns,
                    node_shape="s", ax=ax, with_labels=False)
            draw_boundary_edges(ax, graph, asn_r, d1_id, d2_id, color="k", lw=2)
            draw_boundary_edges(ax, graph, asn, d1_id, d2_id, color="grey",
                                lw=1, style="dashed")
            cut_r = n_cut_edges(graph, asn_r)
            switched = sum(1 for v in nodes if asn[v] != asn_r[v])
            ax.set_title(f"QP rounded  (α={al}, β={be})\n"
                         f"cut={cut_r}  switched={switched}", fontsize=10)
        else:
            ax.set_title("infeasible", fontsize=10)

    fig.suptitle(
        f"Fiedler Vector vs QP Solution — Districts ({d1_id},{d2_id})\n"
        f"Row 1: continuous field  │  Row 2: hard-rounded partition\n"
        f"As α/β ↑, QP solution → Fiedler (spectral bisection = minimum normalised cut)",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig("fiedler_vs_qp.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("Saved: fiedler_vs_qp.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Animation strip: N consecutive QP steps
# ─────────────────────────────────────────────────────────────────────────────

def make_animation_strip(asn_init, n_steps=6,
                         alpha=20.0, beta=1.0, T_round=0.05,
                         fixed_pair=None):
    """
    Compact 4-row strip: Before │ x* │ After │ Δ  for each step.
    Easier to read as a sequence than individual 5-panel figures.
    """
    asn     = dict(asn_init)
    history = []  # list of (before, x_star_map, after, d1, d2, switched)

    for step in range(n_steps):
        bp = border_pairs(asn, graph)
        if not bp: break
        if fixed_pair and fixed_pair in bp:
            d1_id, d2_id = fixed_pair
        else:
            d1_id, d2_id = random.choice(sorted(bp))

        nodes    = [v for v, d in asn.items() if d in (d1_id, d2_id)]
        node_set = set(nodes)
        L_sym, idx = build_sym_laplacian(graph, nodes)
        x0   = np.array([1.0 if asn[v] == d1_id else 0.0 for v in nodes])
        pop  = np.array([graph.nodes[v]["population"] for v in nodes])
        p_bar = pop.sum() / 2.0

        x_star = solve_qp(L_sym, x0, pop, p_bar, alpha, beta, 0.10)
        if x_star is None:
            continue

        asn_new, probs, bits = sigmoid_round(
            x_star, T=T_round, graph=graph, nodes=nodes,
            asn=asn, d1_id=d1_id, d2_id=d2_id)
        if asn_new is None:
            continue

        xstar_map = {v: x_star[i] for i, v in enumerate(nodes)}
        switched  = [v for v in nodes if asn[v] != asn_new[v]]
        history.append((dict(asn), xstar_map, dict(asn_new),
                        d1_id, d2_id, switched, node_set))
        asn = dict(asn_new)

    if not history:
        print("No steps recorded for animation strip.")
        return

    n_cols  = len(history)
    n_rows  = 4
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 6 * n_rows))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    cmap_f = plt.cm.RdBu_r
    row_titles = ["Before", "x* field", "After", "Δ (switched)"]

    for col, (asn_b, xsm, asn_a, d1, d2, switched, nset) in enumerate(history):
        cut_b = n_cut_edges(graph, asn_b)
        cut_a = n_cut_edges(graph, asn_a)

        # row 0: before
        ax = axes[0][col]
        nc = node_colors_full(asn_b, highlight_pair={d1, d2})
        nx.draw(graph, pos=pos, node_color=nc, node_size=40,
                node_shape="s", ax=ax, with_labels=False)
        draw_boundary_edges(ax, graph, asn_b, d1, d2, color="k", lw=2)
        ax.set_title(f"Step {col+1}\nBefore  cut={cut_b}", fontsize=10)
        if col == 0: ax.set_ylabel(row_titles[0], fontsize=11, fontweight="bold")

        # row 1: x* field
        ax = axes[1][col]
        xcolors = [cmap_f(xsm[v]) if v in nset else (0.85, 0.85, 0.85, 0.3)
                   for v in graph.nodes()]
        nx.draw(graph, pos=pos, node_color=xcolors, node_size=40,
                node_shape="s", ax=ax, with_labels=False)
        draw_boundary_edges(ax, graph, asn_b, d1, d2, color="k", lw=1, style="dashed")
        ax.set_title(f"x* field", fontsize=10)
        if col == 0: ax.set_ylabel(row_titles[1], fontsize=11, fontweight="bold")

        # row 2: after
        ax = axes[2][col]
        nc2 = node_colors_full(asn_a, highlight_pair={d1, d2})
        nx.draw(graph, pos=pos, node_color=nc2, node_size=40,
                node_shape="s", ax=ax, with_labels=False)
        draw_boundary_edges(ax, graph, asn_a, d1, d2, color="k", lw=2)
        draw_boundary_edges(ax, graph, asn_b, d1, d2, color="grey", lw=0.8, style="dashed")
        ax.set_title(f"After  cut={cut_a}  (Δ={cut_a-cut_b:+d})", fontsize=10)
        if col == 0: ax.set_ylabel(row_titles[2], fontsize=11, fontweight="bold")

        # row 3: switched
        ax = axes[3][col]
        sw_set = set(switched)
        d2d1   = set(v for v in switched if asn_a[v] == d1)
        d1d2   = set(v for v in switched if asn_a[v] == d2)
        diff_colors = []
        for v in graph.nodes():
            if v in d2d1:    diff_colors.append("red")
            elif v in d1d2:  diff_colors.append("blue")
            elif v in nset:
                c = list(mcolors.to_rgba(DCOL[asn_a[v]])); c[3] = 0.3
                diff_colors.append(tuple(c))
            else:
                diff_colors.append((0.85, 0.85, 0.85, 0.2))
        nx.draw(graph, pos=pos, node_color=diff_colors, node_size=40,
                node_shape="s", ax=ax, with_labels=False)
        draw_boundary_edges(ax, graph, asn_a, d1, d2, color="k", lw=2)
        ax.set_title(f"Switched: {len(switched)}\n"
                     f"Red→D{d1}  Blue→D{d2}", fontsize=10)
        if col == 0: ax.set_ylabel(row_titles[3], fontsize=11, fontweight="bold")

    fig.suptitle(
        f"QP Boundary Movement — {n_cols} Consecutive Steps  "
        f"(α={alpha}, β={beta}, T={T_round})",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig("boundary_animation.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("Saved: boundary_animation.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  RUN
# ─────────────────────────────────────────────────────────────────────────────
ALPHA   = 20.0
BETA    = 1.0
T_ROUND = 0.05

# ── (a) Fiedler vs QP diagnostic (fixed pair 0,1) ────────────────────────────
print("Generating Fiedler vs QP comparison …")
visualise_fiedler_vs_qp(cddict, d1_id=0, d2_id=1, alpha=ALPHA, beta=BETA)

# ── (b) Step-by-step 5-panel figures (6 steps, fixed pair for clarity) ───────
print("\nGenerating per-step 5-panel figures …")
asn_current = dict(cddict)
for step in range(1, 7):
    asn_current, ok = visualise_qp_step(
        asn_current, step_num=step,
        alpha=ALPHA, beta=BETA, T_round=T_ROUND,
        fixed_pair=(0, 1)          # fix pair so we watch one boundary evolve
    )
    if not ok:
        print(f"  Step {step}: rejected / infeasible")

# ── (c) Animation strip (different pairs each step, 8 steps) ─────────────────
print("\nGenerating animation strip …")
make_animation_strip(cddict, n_steps=8, alpha=ALPHA, beta=BETA, T_round=T_ROUND)

print("\nAll done. Output files:")
print("  fiedler_vs_qp.png        — Fiedler vector vs QP for 4 α/β regimes")
print("  qp_step_01.png … _06.png — 5-panel per-step boundary visualisation")
print("  boundary_animation.png   — compact 4×N strip of N consecutive steps")
