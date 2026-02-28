"""
qp_model.py — Quadratic Programming redistricting model.

Contains
--------
build_laplacian(graph, nodes, attr)
    Unnormalised graph Laplacian L = D − W with kernel w(e) = exp(-|Δattr|).
    x^T L x = Σ w(e)·(x_u−x_v)² is the weighted cut; L is PSD (convex QP).

make_partition(gc_graph, assignment)
    Wrap a district assignment in a GerryChain Partition object.

_solve_qp(graph, nodes, assignment, d1_id, alpha, beta, epsilon)
    Solve the continuous QP relaxation for one district pair.
    Objective: min  α·x^T L x  +  β·‖x−x₀‖²  (strictly convex, no NonConvex flag).
    Minimising the weighted cut pushes boundaries to cross low-weight edges,
    directly increasing the Density Score.
    Returns x_star ∈ [0,1]^n or None on failure.

_randomized_round(x_star, nodes, d1_id, d2_id, graph, assignment, max_tries)
    Bernoulli rounding with connectivity check.
    Returns (new_assignment, log_q_fwd, bits) or (None, None, None).

qp_mh_proposal(graph, assignment, alpha, beta, epsilon, lam)
    One QP-MH step with exact Metropolis-Hastings correction.
    Returns (new_assignment, accepted).

run_qp_only(graph, gc_graph, initial_assignment, steps, alpha, beta, epsilon, lam)
    Run the standalone QP-MH chain for `steps` steps.
    Returns (metrics_list, times_list, accepts_list, final_assignment).

tune_qp_parameters(graph, gc_graph, initial_assignment, param_grid, steps)
    Grid search over (alpha, beta, epsilon, lam).
    Evaluates each combo by mean density_score over `steps` steps.
    Returns a sorted list of (mean_ds, params_dict) — best first.

Parameter guide
---------------
alpha   : weight of the Laplacian term (weighted-cut signal).
          Higher → QP minimises weighted cut more aggressively (stronger
          density alignment); lower → solution stays closer to x0 (β dominates).
          Typical range: [0.1, 5.0]

beta    : weight of the quadratic proximity term  β‖x − x₀‖².
          Higher → proposed partition stays close to current.  Lower → larger
          jumps but lower acceptance.  Typical range: [1.0, 50.0]

epsilon : population balance tolerance (fraction of ideal district pop).
          Typical range: [0.05, 0.20]

lam     : MH target temperature  π(x) ∝ exp(−λ · cut_edges(x)).
          Higher → strongly prefers fewer cut edges (less exploration).
          Typical range: [0.01, 0.20]
"""

import random
import time
import numpy as np
import networkx as nx

import gurobipy as gp
from gurobipy import GRB

from gerrychain import Partition
from gerrychain.updaters import Tally, cut_edges

from metrics import n_cut_edges, border_pairs_set, density_score, record_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Laplacian & partition helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_laplacian(graph, nodes, attr="density"):
    """
    Returns (L, idx) where L = D − W is the unnormalised graph Laplacian
    with kernel w(e) = exp(-|Δattr|).

    Key identity:  x^T L x = Σ_{e=(u,v)} w(e)·(x_u − x_v)²

    This equals the total kernel weight on cut edges weighted by how far the
    relaxed indicators differ.  Minimising x^T L x is therefore equivalent to
    minimising the weighted cut — boundaries are pushed to cross low-weight
    (cross-density) edges, directly maximising the Density Score.

    L is positive semi-definite, so minimising x^T L x is a CONVEX QP.
    No NonConvex flag is needed.
    """
    n   = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    sub = graph.subgraph(nodes)
    W   = np.zeros((n, n))
    for u, v in sub.edges():
        w = np.exp(-abs(graph.nodes[u].get(attr, 0) - graph.nodes[v].get(attr, 0)))
        i, j = idx[u], idx[v]
        W[i, j] = w; W[j, i] = w
    L = np.diag(W.sum(axis=1)) - W      # L = D − W  (PSD)
    return L, idx


def make_partition(gc_graph, assignment):
    return Partition(
        gc_graph,
        assignment=assignment,
        updaters={"population": Tally("population"), "cut_edges": cut_edges},
    )


# ─────────────────────────────────────────────────────────────────────────────
# QP solver
# ─────────────────────────────────────────────────────────────────────────────

def _solve_qp(graph, nodes, assignment, d1_id,
              alpha=1.0, beta=10.0, epsilon=0.10):
    """
    Minimise   α · x^T L x  +  β · ‖x − x₀‖²
    subject to  (1−ε)·p̄ ≤ popᵀx ≤ (1+ε)·p̄,   x ∈ [0,1]^n

    x^T L x = Σ_{e=(u,v)} w(e)·(x_u−x_v)²  is the weighted cut.
    Minimising it pushes boundaries to cross low-weight (cross-density)
    edges — directly increasing the Density Score.

    L is PSD  ⟹  objective is CONVEX.  No NonConvex flag needed.
    Gurobi solves this as a standard convex QP, which is significantly
    faster than the non-convex formulation.

    x₀  = current indicator (1 if node in D1, 0 if in D2).
    p̄   = half the total population of the merged pair.

    Returns x_star ∈ [0,1]^n or None on solver failure.
    """
    n      = len(nodes)
    L, _   = build_laplacian(graph, nodes, attr="density")
    x0     = np.array([1.0 if assignment[v] == d1_id else 0.0 for v in nodes])
    pop    = np.array([graph.nodes[v]["population"] for v in nodes])
    p_bar  = pop.sum() / 2.0

    try:
        m = gp.Model("qp")
        m.setParam("OutputFlag", 0)
        x = m.addVars(n, lb=0.0, ub=1.0, name="x")

        obj = gp.QuadExpr()
        for i in range(n):
            for j in range(i, n):
                val = L[i, j]
                if abs(val) > 1e-12:
                    if i == j:
                        obj += alpha * val * x[i] * x[i]            # α L diagonal
                    else:
                        obj += 2 * alpha * val * x[i] * x[j]        # α L off-diagonal
        for i in range(n):
            obj += beta * (x[i] * x[i] - 2.0 * x0[i] * x[i])       # β ‖x − x₀‖² (const ignored)

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


# ─────────────────────────────────────────────────────────────────────────────
# Randomized rounding
# ─────────────────────────────────────────────────────────────────────────────

def _randomized_round(x_star, nodes, d1_id, d2_id, graph, assignment,
                      max_tries=20):
    """
    Bernoulli(x*_v) rounding with district-connectivity check.

    Returns (new_assignment, log_q_fwd, bits) on success,
            (None, None, None)               after max_tries failures.
    """
    n = len(nodes)
    for _ in range(max_tries):
        bits    = (np.random.rand(n) < x_star).astype(float)
        new_asn = dict(assignment)
        for i, v in enumerate(nodes):
            new_asn[v] = d1_id if bits[i] == 1 else d2_id

        sub1 = graph.subgraph([v for v, d in new_asn.items() if d == d1_id])
        sub2 = graph.subgraph([v for v, d in new_asn.items() if d == d2_id])
        if nx.is_connected(sub1) and nx.is_connected(sub2):
            eps      = 1e-12
            x_star_c = np.clip(x_star, eps, 1 - eps)
            log_q    = np.sum(bits * np.log(x_star_c)
                              + (1 - bits) * np.log(1 - x_star_c))
            return new_asn, log_q, bits
    return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# QP-MH proposal
# ─────────────────────────────────────────────────────────────────────────────

def qp_mh_proposal(graph, assignment,
                   alpha=1.0, beta=10.0, epsilon=0.10, lam=0.05):
    """
    One QP-MH step.

    MH target : π(x) ∝ exp(−λ · cut_edges(x))
    Acceptance : log α = −λ·Δcut
                        + log q(x|x′,pair) − log q(x′|x,pair)
                        + log|bp(x)|       − log|bp(x′)|

    The QP proposal (min α·xᵀLx + β‖x−x₀‖²) generates candidates that
    minimise the weighted cut (= maximise density alignment).  The MH step
    then accepts/rejects them against the cut-edge target π, ensuring detailed balance.

    Returns (new_assignment, accepted_bool).
    """
    bp_cur   = border_pairs_set(assignment, graph)
    n_bp_cur = len(bp_cur)
    if n_bp_cur == 0:
        return dict(assignment), False

    d1_id, d2_id = random.choice(sorted(bp_cur))
    nodes        = [n for n, d in assignment.items() if d in (d1_id, d2_id)]

    # Forward QP + randomized round
    x_star_fwd = _solve_qp(graph, nodes, assignment, d1_id, alpha, beta, epsilon)
    if x_star_fwd is None:
        return dict(assignment), False

    proposed, log_q_fwd, _ = _randomized_round(
        x_star_fwd, nodes, d1_id, d2_id, graph, assignment)
    if proposed is None:
        return dict(assignment), False

    # Reverse QP (exact MH ratio)
    x_star_rev = _solve_qp(graph, nodes, proposed, d1_id, alpha, beta, epsilon)
    if x_star_rev is None:
        return dict(assignment), False

    eps       = 1e-12
    x_rev_c   = np.clip(x_star_rev, eps, 1 - eps)
    orig_bits = np.array([1.0 if assignment[v] == d1_id else 0.0 for v in nodes])
    log_q_rev = np.sum(orig_bits * np.log(x_rev_c)
                       + (1 - orig_bits) * np.log(1 - x_rev_c))

    cut_cur      = n_cut_edges(graph, assignment)
    cut_prop     = n_cut_edges(graph, proposed)
    bp_prop      = border_pairs_set(proposed, graph)
    n_bp_prop    = max(len(bp_prop), 1)

    log_alpha = (
        -lam * (cut_prop - cut_cur)
        + (log_q_rev - log_q_fwd)
        + (np.log(n_bp_cur) - np.log(n_bp_prop))
    )

    if np.log(random.random() + 1e-300) < log_alpha:
        return proposed, True
    return dict(assignment), False


# ─────────────────────────────────────────────────────────────────────────────
# Standalone QP-only runner
# ─────────────────────────────────────────────────────────────────────────────

def run_qp_only(graph, gc_graph, initial_assignment, steps=50,
                alpha=1.0, beta=10.0, epsilon=0.10, lam=0.05,
                verbose=True):
    """
    Run the QP-MH chain in isolation for `steps` steps.

    Parameters
    ----------
    graph              : networkx.Graph  (with 'density' and 'population' attrs)
    gc_graph           : gerrychain.Graph
    initial_assignment : dict {node → district_id}
    steps              : number of QP-MH steps
    alpha, beta, epsilon, lam : QP-MH hyperparameters (see module docstring)
    verbose            : print per-step diagnostics

    Returns
    -------
    metrics  : list of dicts (density_score, pp, pop_dev per step)
    times    : list of wall-clock seconds per step
    accepts  : list of 0/1 per step
    final    : final assignment dict
    """
    asn     = dict(initial_assignment)
    metrics = []
    times   = []
    accepts = []

    for s in range(steps):
        t0       = time.perf_counter()
        asn, acc = qp_mh_proposal(graph, asn, alpha=alpha, beta=beta,
                                  epsilon=epsilon, lam=lam)
        elapsed  = time.perf_counter() - t0
        m        = record_metrics(graph, asn)
        metrics.append(m)
        times.append(elapsed)
        accepts.append(int(acc))

        if verbose:
            print(f"  [QP-only] step {s+1:3d}  "
                  f"DS={m['density_score']:.4f}  PP={m['pp']:.4f}  "
                  f"acc={acc}  t={elapsed:.3f}s")

    ar = 100 * np.mean(accepts) if accepts else float("nan")
    if verbose:
        print(f"  [QP-only] accept rate: {ar:.1f}%  "
              f"mean step: {1e3*np.mean(times):.1f}ms\n")

    return metrics, times, accepts, asn


# ─────────────────────────────────────────────────────────────────────────────
# Parameter tuning via grid search
# ─────────────────────────────────────────────────────────────────────────────

def tune_qp_parameters(graph, gc_graph, initial_assignment,
                        param_grid=None, steps=30, verbose=True):
    """
    Grid search over QP-MH hyperparameters.

    Parameters
    ----------
    graph, gc_graph, initial_assignment : as above
    param_grid : dict with lists of values for each hyperparameter.
        Default grid:
            alpha   : [0.5, 1.0, 2.0]
            beta    : [5.0, 10.0, 20.0]
            epsilon : [0.05, 0.10, 0.15]
            lam     : [0.02, 0.05, 0.10]
    steps   : QP-MH steps per combo (short runs — for speed)
    verbose : print each combo result

    Returns
    -------
    results : list of (mean_density_score, params_dict), sorted best-first.

    Notes
    -----
    mean_density_score measures how well the chain's output aligns districts
    with the density kernel.  Higher is better.

    Tie-breaking: among combos with similar DS, lower pop_dev is preferred.
    The returned list is sorted by (−mean_DS, mean_pop_dev).
    """
    if param_grid is None:
        param_grid = {
            "alpha":   [0.5, 1.0, 2.0],
            "beta":    [5.0, 10.0, 20.0],
            "epsilon": [0.05, 0.10, 0.15],
            "lam":     [0.02, 0.05, 0.10],
        }

    # Build all combos (Cartesian product without itertools to keep deps minimal)
    keys = list(param_grid.keys())
    combos = [{}]
    for k in keys:
        combos = [dict(**c, **{k: v}) for c in combos for v in param_grid[k]]

    if verbose:
        print(f"Tuning QP parameters: {len(combos)} combos × {steps} steps each")
        print("-" * 72)

    results = []
    for i, params in enumerate(combos):
        metrics, _, accepts, _ = run_qp_only(
            graph, gc_graph, initial_assignment,
            steps=steps, verbose=False, **params
        )
        ds_vals  = [m["density_score"] for m in metrics]
        pd_vals  = [m["pop_dev"]       for m in metrics]
        mean_ds  = float(np.mean(ds_vals))
        mean_pd  = float(np.mean(pd_vals))
        acc_rate = 100 * np.mean(accepts)

        results.append((mean_ds, mean_pd, params))

        if verbose:
            print(f"  [{i+1:3d}/{len(combos)}] "
                  f"α={params['alpha']:.1f}  β={params['beta']:.1f}  "
                  f"ε={params['epsilon']:.2f}  λ={params['lam']:.2f}  │  "
                  f"mean DS={mean_ds:.4f}  pop_dev={mean_pd:.4f}  "
                  f"acc={acc_rate:.1f}%")

    # Sort: highest DS first; break ties by lowest pop_dev
    results.sort(key=lambda r: (-r[0], r[1]))

    if verbose:
        print("\n── Top 5 parameter sets ─────────────────────────────────────────────")
        for rank, (ds, pd, p) in enumerate(results[:5], 1):
            print(f"  #{rank}  mean DS={ds:.4f}  pop_dev={pd:.4f}  "
                  f"α={p['alpha']}  β={p['beta']}  ε={p['epsilon']}  λ={p['lam']}")

    return [(ds, p) for ds, _, p in results]


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    warnings.filterwarnings("ignore")

    from grid_setup import graph, gc_graph, initial_assignment, ns

    PLOTS_DIR = Path(__file__).parent.parent / "plots"
    PLOTS_DIR.mkdir(exist_ok=True)

    # ── 1. Parameter tuning ───────────────────────────────────────────────────
    print("=" * 72)
    print("QP-MH Parameter Tuning")
    print("=" * 72)
    tuning_results = tune_qp_parameters(
        graph, gc_graph, initial_assignment,
        param_grid={
            "alpha":   [0.5, 1.0, 2.0],
            "beta":    [5.0, 10.0, 20.0],
            "epsilon": [0.05, 0.10],
            "lam":     [0.02, 0.05, 0.10],
        },
        steps=30,
        verbose=True,
    )

    best_ds, best_params = tuning_results[0]
    print(f"\nBest params: {best_params}  (mean DS={best_ds:.4f})")

    # ── 2. Full run with best params ──────────────────────────────────────────
    print("\n" + "=" * 72)
    print("QP-MH Full Run (100 steps, best parameters)")
    print("=" * 72)
    metrics, times, accepts, final_asn = run_qp_only(
        graph, gc_graph, initial_assignment,
        steps=100, verbose=True, **best_params
    )

    # ── 3. Tuning heatmap: mean DS by (alpha, beta) for best epsilon+lam ─────
    best_eps = best_params["epsilon"]
    best_lam = best_params["lam"]
    alphas   = [0.5, 1.0, 2.0, 3.0]
    betas    = [5.0, 10.0, 20.0, 30.0]

    heatmap = np.zeros((len(alphas), len(betas)))
    print("\nGenerating alpha×beta heatmap …")
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            m, _, _, _ = run_qp_only(
                graph, gc_graph, initial_assignment,
                steps=20, verbose=False,
                alpha=a, beta=b, epsilon=best_eps, lam=best_lam,
            )
            heatmap[i, j] = np.mean([x["density_score"] for x in m])

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(heatmap, aspect="auto", origin="lower",
                   cmap="RdYlGn", vmin=heatmap.min(), vmax=heatmap.max())
    ax.set_xticks(range(len(betas)));  ax.set_xticklabels(betas)
    ax.set_yticks(range(len(alphas))); ax.set_yticklabels(alphas)
    ax.set_xlabel("beta  (proximity weight)");  ax.set_ylabel("alpha  (weighted-cut weight)")
    ax.set_title(
        f"Mean Density Score — QP-MH\n"
        f"ε={best_eps}  λ={best_lam}  (20 steps each)\n"
        r"$DS(x)=\Sigma_{\rm intra}w(e)/\Sigma_{\rm all}w(e)$"
    )
    plt.colorbar(im, ax=ax, label="Mean DS")
    for i in range(len(alphas)):
        for j in range(len(betas)):
            ax.text(j, i, f"{heatmap[i,j]:.3f}", ha="center", va="center",
                    fontsize=8, color="black")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "qp_tuning_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: qp_tuning_heatmap.png")

    # ── 4. Full-run density-score trace plot ──────────────────────────────────
    ds_trace = [m["density_score"] for m in metrics]
    fig, ax  = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(ds_trace) + 1), ds_trace, "b-", alpha=0.5, lw=0.8)
    cum_mean = np.cumsum(ds_trace) / np.arange(1, len(ds_trace) + 1)
    ax.plot(range(1, len(ds_trace) + 1), cum_mean, "b--", lw=2,
            label=f"Running mean  (final={cum_mean[-1]:.4f})")
    ax.set_xlabel("Step"); ax.set_ylabel("Density Score")
    ax.set_title(
        f"QP-MH  Density Score Trace  "
        f"(α={best_params['alpha']} β={best_params['beta']} "
        f"ε={best_params['epsilon']} λ={best_params['lam']})\n"
        r"$DS(x)=\Sigma_{\rm intra}w(e)/\Sigma_{\rm all}w(e)$,"
        r"  $w(e)=\exp(-|\Delta\mathrm{density}|)$"
    )
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "qp_trace.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: qp_trace.png")

    print("\nDone.")
