"""
Gerrychain: QP-MH (randomized rounding) + MEW + ReCom + Hybrid
===============================================================

Key additions
-------------
1. Randomized rounding  — each node v is assigned to D1 with probability x*_v
   (Bernoulli), making q(x′|x) fully computable and the MH correction exact.

2. Marked Edge Walk (MEW) — implemented from DeFord et al. for district pairs:
     State  : (spanning tree T,  marked edge e ∈ T)
     Move   : pick a random non-tree edge f, add to T → cycle C,
              remove a random edge g from C (≠ f) → T′,
              mark a random edge in T′
     MH ratio (exact):
              α = min{1,  π(x′)/π(x)  ·  q(T,e | T′,e′) / q(T′,e′ | T,e) }
     where
       q(T′,e′ | T,e)  =  [1/|non-tree(T)|] · [1/(|C(T,f)|−1)] · [1/(|T′|)]
       q(T,e  | T′,e′) =  [1/|non-tree(T′)|] · [1/(|C(T′,g)|−1)] · [1/(|T|)]
     |T| = |T′| = n−1 always, so the mark factors cancel.

Target for both MH schemes:
     π(x) ∝ exp(−λ · cut_edges(x))

Chains compared
---------------
  A. ReCom       (20 steps,  global jumps, no gradient)
  B. QP-MH       (20 steps,  local gradient, randomized rounding, exact MH)
  C. MEW         (20 steps,  local spanning-tree walk, exact MH)
  D. Hybrid      (10 rounds, 1 ReCom + 1 QP-MH)

Diagnostics: ACF, ESS, τ_int, running mean, TV proxy — all on Density Score trace.
Wall-clock timing per step is also recorded.
"""

# ─────────────────────────────────────────────────────────────────────────────
import random, warnings, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")

from gerrychain.proposals import recom

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  GRAPH SETUP  (grid, initial partition, density kernel — from grid_setup)
# ─────────────────────────────────────────────────────────────────────────────
from grid_setup import (
    graph, gc_graph,
    reference_assignment, initial_assignment,
    pop_target, ns,
    plot_boundary_nodes,
)

from metrics import (
    n_cut_edges, border_pairs_set,
    density_score, record_metrics,
)

from qp_model import (
    make_partition,
    qp_mh_proposal,
)

import plots as _plots

PLOTS_DIR = Path(__file__).parent.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ── initial plot ──────────────────────────────────────────────────────────────
_plots.plot_initial_partition(graph, initial_assignment, ns, PLOTS_DIR)



# ─────────────────────────────────────────────────────────────────────────────
# 6.  MARKED EDGE WALK (MEW) — fast stateful implementation
#
#  Key speedups vs. naive version
#  ───────────────────────────────
#  1. The spanning tree is built ONCE per district-pair selection and persisted
#     in a MEWState object across steps.  No NetworkX graph copy per step.
#  2. Tree stored as a plain adjacency dict {node: set(neighbours)}.
#     Cycle detection = iterative DFS path-finding in O(path_length).
#  3. Non-tree edge set maintained as a Python set; updated incrementally
#     (remove f, add g) in O(1) rather than recomputed from scratch.
#  4. Partition assignment derived by iterative BFS from one component root —
#     no full-graph connectivity check needed (tree removal always gives 2 parts).
#
#  State : (adj_tree, non_tree_set, marked_edge, nodes, d1_id, d2_id)
#  Move  : pick f ∈ non_tree  →  cycle C  →  remove g ∈ C\{f}  →  mark e′
#  MH    : exp(−λΔcut) · [M_T/M_T′] · [(|C_g|−1)/(|C_f|−1)]   (exact)
# ─────────────────────────────────────────────────────────────────────────────

class MEWState:
    """Persistent state for the Marked Edge Walk on a merged district pair."""

    def __init__(self, graph, assignment, d1_id, d2_id):
        self.d1_id  = d1_id
        self.d2_id  = d2_id
        self.nodes  = [n for n, d in assignment.items() if d in (d1_id, d2_id)]
        node_set    = set(self.nodes)
        sub         = graph.subgraph(self.nodes)

        # ── Fast random spanning tree via shuffled-BFS  O(n+m) ───────────────
        # nx.random_spanning_tree uses Wilson's loop-erased random walk: O(n·τ)
        # where τ is the cover time — takes ~8s on a 360-node grid subgraph.
        # Shuffled-BFS is O(n+m) ≈ 0.8ms and gives a valid random spanning tree
        # for initialisation; the MEW walk itself ensures correct stationarity.
        root     = random.choice(self.nodes)
        self.adj = {v: set() for v in self.nodes}
        visited  = {root}
        queue    = [root]
        tree_edge_set = set()
        while queue:
            idx = random.randrange(len(queue))
            u   = queue[idx]; queue[idx] = queue[-1]; queue.pop()
            nbrs = list(graph.neighbors(u)); random.shuffle(nbrs)
            for v in nbrs:
                if v in node_set and v not in visited:
                    visited.add(v)
                    self.adj[u].add(v); self.adj[v].add(u)
                    tree_edge_set.add(frozenset((u, v)))
                    queue.append(v)

        # All subgraph edges; non-tree = complement of spanning tree
        all_edges     = {frozenset(e) for e in sub.edges()}
        self.non_tree = list(all_edges - tree_edge_set)

        # Marked edge: prefer a straddling edge (crosses D1/D2 boundary)
        straddling = [(u, v) for u in self.nodes for v in self.adj[u]
                      if u < v and assignment[u] != assignment[v]]
        all_tree   = [(u, v) for u in self.nodes for v in self.adj[u] if u < v]
        self.marked = random.choice(straddling) if straddling \
                      else random.choice(all_tree)

    # ── fast path-in-tree via iterative BFS ──────────────────────────────────
    def _tree_path(self, src, dst):
        """Return list of nodes on the unique tree path from src to dst."""
        parent = {src: None}
        queue  = [src]
        while queue:
            cur = queue.pop()
            if cur == dst:
                path = []
                while cur is not None:
                    path.append(cur); cur = parent[cur]
                return path[::-1]
            for nb in self.adj[cur]:
                if nb not in parent:
                    parent[nb] = cur; queue.append(nb)
        return []          # unreachable (tree is connected)

    # ── cycle = path(u→v) + edge (v,u) ───────────────────────────────────────
    def _cycle_edges(self, u, v):
        """Edges of cycle formed by adding (u,v) to the current tree."""
        path = self._tree_path(u, v)
        if len(path) < 2:
            return []
        edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        edges.append((v, u))   # close the cycle
        return edges

    # ── BFS partition after removing marked edge ──────────────────────────────
    def _partition(self, marked, full_assignment):
        """
        Remove `marked` from adj; BFS from each endpoint → two component sets.
        District labels are taken from `full_assignment` (not assumed from
        edge-endpoint order, which is arbitrary).
        Returns None if the cut does not yield exactly 2 connected components
        covering all nodes (safety check).
        Does NOT modify self.adj permanently.
        """
        u, v = marked
        self.adj[u].discard(v); self.adj[v].discard(u)

        comp = {}
        for root in (u, v):
            dist = full_assignment.get(root, self.d1_id)
            if root in comp:
                continue
            queue = [root]
            while queue:
                cur = queue.pop()
                if cur in comp:
                    continue
                comp[cur] = dist
                queue.extend(nb for nb in self.adj[cur] if nb not in comp)

        self.adj[u].add(v); self.adj[v].add(u)

        # Safety: every node must be assigned
        if len(comp) != len(self.nodes):
            return None
        return comp

    # ── one MEW step ──────────────────────────────────────────────────────────
    def step(self, graph, full_assignment, lam=0.05):
        """
        Attempt one MEW move.
        Returns (new_full_assignment, accepted).
        Modifies self in-place on acceptance.
        """
        if not self.non_tree:
            return dict(full_assignment), False

        # Step 1: pick f
        f_idx = random.randrange(len(self.non_tree))
        f     = tuple(self.non_tree[f_idx])

        # Step 2: cycle
        cycle  = self._cycle_edges(f[0], f[1])
        c_minus_f = [e for e in cycle if frozenset(e) != frozenset(f)]
        if not c_minus_f:
            return dict(full_assignment), False
        C_f_len = len(cycle)

        # Step 3: pick g
        g = random.choice(c_minus_f)

        # Step 4: update tree  T′ = T ∪ {f} \ {g}
        self.adj[f[0]].add(f[1]); self.adj[f[1]].add(f[0])
        self.adj[g[0]].discard(g[1]); self.adj[g[1]].discard(g[0])

        # Update non-tree set: remove f (now in tree), add g (now non-tree)
        self.non_tree[f_idx] = frozenset(g)   # reuse slot
        # M_T_prime == M_T  — |non_tree| is conserved (swap f↔g), so ratio = 1

        # Cycle in T′ formed by re-adding g  (needed for MH ratio)
        cycle_rev = self._cycle_edges(g[0], g[1])
        C_g_len   = max(len(cycle_rev), 2)

        # Step 5: pick new marked edge e′ uniformly from T′
        #   (all tree edges = self.adj entries, but stored once per pair)
        tree_edge_list = [(u, v) for u in self.nodes for v in self.adj[u] if u < v]
        marked_prime   = random.choice(tree_edge_list)

        # ── partition from T′ + marked_prime ─────────────────────────────────
        part = self._partition(marked_prime, full_assignment)
        if part is None:
            # Undo tree update and reject
            self.adj[f[0]].discard(f[1]); self.adj[f[1]].discard(f[0])
            self.adj[g[0]].add(g[1]); self.adj[g[1]].add(g[0])
            self.non_tree[f_idx] = frozenset(f)
            return dict(full_assignment), False
        proposed = dict(full_assignment)
        proposed.update(part)

        # ── MH ratio ─────────────────────────────────────────────────────────
        cut_cur  = n_cut_edges(graph, full_assignment)
        cut_prop = n_cut_edges(graph, proposed)
        # M_T == M_T_prime (non-tree size is conserved), so ratio = 1
        log_q_ratio = (np.log(C_g_len - 1) - np.log(max(C_f_len - 1, 1)))
        log_alpha   = -lam * (cut_prop - cut_cur) + log_q_ratio

        if np.log(random.random() + 1e-300) < log_alpha:
            self.marked = marked_prime
            return proposed, True
        else:
            # Undo tree update
            self.adj[f[0]].discard(f[1]); self.adj[f[1]].discard(f[0])
            self.adj[g[0]].add(g[1]); self.adj[g[1]].add(g[0])
            self.non_tree[f_idx] = frozenset(f)
            return dict(full_assignment), False


# ── module-level MEW state (reinitialised when district pair or sizes change) ──
_mew_state: MEWState = None
_mew_pair            = (None, None)
_mew_sizes           = (None, None)   # (|D1 nodes|, |D2 nodes|) at last init

def mew_proposal(graph, assignment, lam=0.05):
    """
    One MEW step.  Reuses the MEWState when the chosen district pair AND
    their node counts are unchanged (i.e. the last step was accepted for the
    same pair, preserving the spanning tree).  Reinitialises otherwise.
    Returns (new_assignment, accepted).
    """
    global _mew_state, _mew_pair, _mew_sizes

    bp = border_pairs_set(assignment, graph)
    if not bp:
        return dict(assignment), False

    d1_id, d2_id = random.choice(sorted(bp))

    # Count nodes in each district of this pair
    pair_nodes = [n for n, d in assignment.items() if d in (d1_id, d2_id)]
    sz1 = sum(1 for n in pair_nodes if assignment[n] == d1_id)
    sz2 = len(pair_nodes) - sz1

    # Reinitialise if pair changed, sizes changed, or first call
    if ((d1_id, d2_id) != _mew_pair
            or (sz1, sz2) != _mew_sizes
            or _mew_state is None):
        try:
            _mew_state = MEWState(graph, assignment, d1_id, d2_id)
            _mew_pair  = (d1_id, d2_id)
            _mew_sizes = (sz1, sz2)
        except Exception:
            return dict(assignment), False

    return _mew_state.step(graph, assignment, lam=lam)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  RUN ALL CHAINS  (20 steps each)
# ─────────────────────────────────────────────────────────────────────────────

def run_chain(name, step_fn, steps, init_asn):
    """Generic runner. step_fn(asn) → (new_asn, accepted_bool)."""
    asn     = dict(init_asn)
    metrics = []
    times   = []
    accepts = []
    for s in range(steps):
        t0       = time.perf_counter()
        asn, acc = step_fn(asn)
        elapsed  = time.perf_counter() - t0
        m        = record_metrics(graph, asn)
        metrics.append(m)
        times.append(elapsed)
        accepts.append(int(acc))
        print(f"  [{name}] step {s+1:2d}  DS={m['density_score']:.4f}  "
              f"PP={m['pp']:.4f}  acc={acc}  t={elapsed:.3f}s")
    ar = 100 * np.mean(accepts) if accepts else float('nan')
    print(f"  [{name}] accept rate: {ar:.1f}%  "
          f"mean step time: {1000*np.mean(times):.1f}ms\n")
    return metrics, times, accepts, asn


# ── ReCom ─────────────────────────────────────────────────────────────────────
print("\n── ReCom (20 steps) ─────────────────────────────────────────────────────")
_part_recom = make_partition(gc_graph, initial_assignment)

def recom_step(_):
    global _part_recom
    _part_recom = recom(_part_recom, "population", pop_target, 0.10, node_repeats=1)
    return dict(_part_recom.assignment), True   # ReCom always "accepts"

recom_metrics, recom_times, _, recom_final = run_chain("ReCom", recom_step, 20, initial_assignment)


# ── QP-MH ─────────────────────────────────────────────────────────────────────
print("\n── QP-MH  (20 steps) ────────────────────────────────────────────────────")

def qp_mh_step(asn):
    return qp_mh_proposal(graph, asn, alpha=0.5, beta=20.0, epsilon=0.05, lam=0.05)

qp_metrics, qp_times, qp_accepts, qp_final = run_chain("QP-MH", qp_mh_step, 20, initial_assignment)


# ── MEW ───────────────────────────────────────────────────────────────────────
print("\n── MEW  (20 steps) ──────────────────────────────────────────────────────")
_mew_state = None; _mew_pair = (None, None); _mew_sizes = (None, None)   # reset state

def mew_step(asn):
    return mew_proposal(graph, asn, lam=0.05)

mew_metrics, mew_times, mew_accepts, mew_final = run_chain("MEW", mew_step, 20, initial_assignment)


# ── Hybrid  (ReCom + multi-step QP-MH with annealed λ, 10 rounds) ────────────
#
# Improvements over the naive 1-ReCom + 1-QP design:
#
#  1. Multi-step local refinement: after each ReCom global jump, run
#     QP_STEPS_PER_RECOM QP-MH steps to exploit the gradient signal before
#     the next jump.  This lets the QP refine rather than just nudge.
#
#  2. Annealed λ: start with a low λ (broad acceptance, large moves) and
#     increase toward LAM_FINAL over the refinement sub-chain.  The schedule
#     log-linearly interpolates  λ_t = LAM_START * (LAM_FINAL/LAM_START)^{t/T}.
#     This prevents the QP sub-chain from getting stuck at the ReCom output.
#
print("\n── Hybrid (10 rounds: 1 ReCom + 3 QP-MH, annealed λ) ───────────────────")

QP_STEPS_PER_RECOM = 3      # QP-MH steps after each ReCom jump
LAM_START          = 0.01   # broad acceptance right after the jump
LAM_FINAL          = 0.10   # tighter toward end of refinement sub-chain

_part_hyb   = make_partition(gc_graph, initial_assignment)
_hyb_round  = [0]           # mutable counter accessible inside closure

def hybrid_step(_):
    global _part_hyb
    _part_hyb = recom(_part_hyb, "population", pop_target, 0.10, node_repeats=1)
    asn = dict(_part_hyb.assignment)

    # Annealed QP-MH refinement
    any_acc = False
    for t in range(QP_STEPS_PER_RECOM):
        frac = t / max(QP_STEPS_PER_RECOM - 1, 1)
        lam  = LAM_START * (LAM_FINAL / LAM_START) ** frac
        asn, acc = qp_mh_proposal(graph, asn, alpha=1.0, beta=10.0,
                                   epsilon=0.10, lam=lam)
        any_acc = any_acc or acc

    _part_hyb = make_partition(gc_graph, asn)
    return asn, any_acc

hyb_metrics, hyb_times, hyb_accepts, hyb_final = run_chain("Hybrid", hybrid_step, 10, initial_assignment)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  FINAL PARTITION PLOTS
# ─────────────────────────────────────────────────────────────────────────────
print("Plotting final partitions …")
_plots.plot_final_partitions(
    graph,
    assignments=[
        (recom_final, "ReCom (20 steps)"),
        (qp_final,    "QP-MH (20 steps)"),
        (mew_final,   "MEW (20 steps)"),
        (hyb_final,   "Hybrid (10 rounds)"),
    ],
    ns=ns,
    plots_dir=PLOTS_DIR,
)


# ─────────────────────────────────────────────────────────────────────────────
# 8.5  BOUNDARY COMPARISON PLOT  (2 × 3 grid)
# ─────────────────────────────────────────────────────────────────────────────
print("Plotting boundary comparison …")
_plots.plot_boundary_comparison(
    reference_assignment, initial_assignment,
    recom_final, qp_final, mew_final, hyb_final,
    plot_boundary_nodes,
    PLOTS_DIR,
)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  METRIC COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
print("Plotting metric comparison …")
_plots.plot_metric_comparison(recom_metrics, qp_metrics, mew_metrics, hyb_metrics,
                              PLOTS_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# 10.  MIXING-TIME DIAGNOSTICS  (long runs: 200 steps)
# ─────────────────────────────────────────────────────────────────────────────
LONG    = 200
MAX_LAG = 60
print(f"\n── Long runs for mixing diagnostics  ({LONG} steps) ────────────────────")

def long_trace(step_fn, init_asn, steps=LONG):
    asn, trace, wtimes = dict(init_asn), [], []
    for _ in range(steps):
        t0      = time.perf_counter()
        asn, _  = step_fn(asn)
        wtimes.append(time.perf_counter() - t0)
        trace.append(density_score(graph, asn))
    return np.array(trace), np.array(wtimes)

print("  ReCom …")
_part_recom = make_partition(gc_graph, initial_assignment)
recom_trace, recom_wt = long_trace(recom_step, initial_assignment)

print("  QP-MH …")
qp_trace, qp_wt = long_trace(qp_mh_step, initial_assignment)

print("  MEW …")
_mew_state = None; _mew_pair = (None, None); _mew_sizes = (None, None)   # reset for long run
mew_trace, mew_wt = long_trace(mew_step, initial_assignment)

print("  Hybrid …")
_part_hyb = make_partition(gc_graph, initial_assignment)   # reset for long run
hyb_trace, hyb_wt = long_trace(hybrid_step, initial_assignment)


# ─────────────────────────────────────────────────────────────────────────────
# 11.  MIXING-TIME FIGURE  (6-panel)
# ─────────────────────────────────────────────────────────────────────────────
print("\nPlotting mixing-time figure …")
(tau_r, tau_q, tau_m, tau_h,
 ess_r, ess_q, ess_m, ess_h,
 eps_r, eps_q, eps_m, eps_h) = _plots.plot_mixing_time(
    recom_trace, qp_trace, mew_trace, hyb_trace,
    recom_wt, qp_wt, mew_wt, hyb_wt,
    LONG, MAX_LAG, PLOTS_DIR,
)

print(f"\n  τ_int  →  ReCom={tau_r:.2f}  QP-MH={tau_q:.2f}  "
      f"MEW={tau_m:.2f}  Hybrid={tau_h:.2f}")
print(f"  ESS    →  ReCom={ess_r:.1f}   QP-MH={ess_q:.1f}  "
      f"MEW={ess_m:.1f}   Hybrid={ess_h:.1f}")
print(f"  ESS/s  →  ReCom={eps_r:.2f}   QP-MH={eps_q:.2f}  "
      f"MEW={eps_m:.2f}   Hybrid={eps_h:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 12.  SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("SUMMARY  (long runs, 200 steps each)")
print("=" * 78)
hdr = (f"{'Metric':<38} {'ReCom':>9} {'QP-MH':>9} {'MEW':>9} {'Hybrid':>9}")
print(hdr); print("-" * 78)

rows = [
    ("Mean Density Score",  recom_trace.mean(), qp_trace.mean(), mew_trace.mean(), hyb_trace.mean(), "{:>9.4f}"),
    ("Std  Density Score",  recom_trace.std(),  qp_trace.std(),  mew_trace.std(),  hyb_trace.std(),  "{:>9.4f}"),
    ("τ_int (ACF)",         tau_r,              tau_q,           tau_m,            tau_h,            "{:>9.2f}"),
    ("ESS",                 ess_r,              ess_q,           ess_m,            ess_h,            "{:>9.1f}"),
    ("ESS / wall-second",   eps_r,              eps_q,           eps_m,            eps_h,            "{:>9.2f}"),
    ("Mean step time (ms)", 1e3*recom_wt.mean(), 1e3*qp_wt.mean(), 1e3*mew_wt.mean(), 1e3*hyb_wt.mean(), "{:>9.1f}"),
]

for name, r, q, m, h, fmt in rows:
    print(f"  {name:<36} {fmt.format(r)} {fmt.format(q)} "
          f"{fmt.format(m)} {fmt.format(h)}")
print("=" * 78)

print("""
Density Score  DS(x) = Σ_{intra} w(e) / Σ_{all} w(e),   w(e) = exp(-|Δdensity|)
─────────────────────────────────────────────────────────────────────────────
Measures how well district boundaries align with the density kernel.
DS = 1 means all high-weight (same-density) edges are intra-district (perfect).
DS = 0 means all edges cross district boundaries.

Interpretation of ESS/second  (computed on Density Score trace)
─────────────────────────────────────────────────────────────────────────────
ESS/step  answers: "which chain decorrelates fastest?"
ESS/sec   answers: "which chain gives the most independent draws per unit
                    of compute?" — the practical metric for redistricting work.

Expected ordering (hypothesis)
  ESS/step  : Hybrid ≥ ReCom > MEW > QP-MH
              (Hybrid combines ReCom's large jumps with QP's gradient signal)
  ESS/sec   : ReCom > MEW > Hybrid > QP-MH
              (QP solve is expensive; MEW is a cheap local walk)

If ESS/sec(Hybrid) > ESS/sec(MEW):  the gradient signal of QP pays off
                                     even after accounting for QP's cost.
If ESS/sec(Hybrid) < ESS/sec(MEW):  MEW's cheapness wins on this graph size;
                                     Hybrid advantage may emerge at larger n.
─────────────────────────────────────────────────────────────────────────────
""")
print("Done. Output files: initial_partition.png, final_partitions.png,")
print("      boundary_comparison.png, metric_comparison.png, mixing_time_analysis.png")
