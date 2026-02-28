"""
QP Parameter Diagnosis & Tuning
================================

Three root causes of QP performing WORSE than initial partition
---------------------------------------------------------------

1. POPULATION IS UNIFORM → kernel collapses to unit weights
   All nodes have population=50, so
       w(e) = exp(-|pop_u - pop_v|) = exp(0) = 1  for ALL edges.
   L_sym is just the standard normalised graph Laplacian.
   This is actually fine — but it means the only compactness signal
   comes from the graph topology itself (Fiedler vector), not the
   attribute kernel.

2. β >> α → QP barely moves from x0, rounding adds pure noise
   With α=1, β=10 (defaults), the ‖x-x0‖² term dominates.
   The solution x* ≈ x0 + small perturbation.
   Many x*_i land near 0.5. Randomised rounding of 0.5 = coin flip.
   Net effect: random noise injected onto the prior → WORSE cut edges.

3. RANDOMISED ROUNDING IS TOO NOISY when x* is near 0.5
   If x*_i ∈ (0.3, 0.7) for many nodes, each node flips independently.
   Even a perfect QP solution gets destroyed by rounding variance.
   Fix: temperature-controlled rounding  p_i = sigmoid((x*_i - 0.5)/T)
   with T→0 recovering the hard threshold.

Correct regime
--------------
  α >> β   →  Laplacian (spectral bisection) dominates.  x* ≈ Fiedler
               vector of the merged subgraph → compact balanced cut.
  β small   →  QP is free to find a better boundary, not anchored to x0.
  T small   →  Rounding is nearly deterministic → x* signal preserved.
  λ moderate →  MH favours compact states but still accepts improvements.

Grid search over (α, β, T_round, λ) with 20-step chains.
"""

import random, time, warnings
import numpy as np
import scipy.linalg as la
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
import gurobipy as gp
from gurobipy import GRB

from gerrychain import Graph, Partition
from gerrychain.updaters import Tally, cut_edges
from gerrychain.proposals import recom

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Graph setup
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED)

gn=6; k=5; ns=50; NUM_DISTRICTS=k
graph = nx.grid_graph([k*gn, k*gn])
for n in graph.nodes():
    graph.nodes[n]["population"] = 50
    if random.random() < 0.5: graph.nodes[n]["pink"]=1; graph.nodes[n]["purple"]=0
    else: graph.nodes[n]["pink"]=0; graph.nodes[n]["purple"]=1
    if 0 in n or k*gn-1 in n:
        graph.nodes[n]["boundary_node"]=True; graph.nodes[n]["boundary_perim"]=1
    else: graph.nodes[n]["boundary_node"]=False

cddict = {x: int(x[0]/gn) for x in graph.nodes()}
total_pop  = sum(graph.nodes[v]["population"] for v in graph.nodes())
pop_target = total_pop / NUM_DISTRICTS
gc_graph   = Graph.from_networkx(graph)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def build_sym_laplacian(graph, nodes):
    n=len(nodes); idx={v:i for i,v in enumerate(nodes)}
    sub=graph.subgraph(nodes); W=np.zeros((n,n))
    for u,v in sub.edges():
        w=np.exp(-abs(graph.nodes[u].get("population",0)-graph.nodes[v].get("population",0)))
        i,j=idx[u],idx[v]; W[i,j]=w; W[j,i]=w
    d=W.sum(axis=1); d_isqrt=np.where(d>0,1/np.sqrt(d),0)
    return np.diag(d_isqrt)@(np.diag(d)-W)@np.diag(d_isqrt), idx

def n_cut_edges(graph, asn):
    return sum(1 for u,v in graph.edges() if asn[u]!=asn[v])

def polsby_popper(graph, asn):
    scores=[]
    for d in set(asn.values()):
        nd=[v for v,dist in asn.items() if dist==d]
        area=len(nd); internal=graph.subgraph(nd).number_of_edges()
        perim=4*area-2*internal
        if perim>0: scores.append(4*np.pi*area/perim**2)
    return float(np.mean(scores)) if scores else 0.0

def border_pairs_set(asn, graph):
    bp=set()
    for u,v in graph.edges():
        d1,d2=asn[u],asn[v]
        if d1!=d2: bp.add((min(d1,d2),max(d1,d2)))
    return bp

def make_partition(gc_graph, asn):
    return Partition(gc_graph, assignment=asn,
                     updaters={"population": Tally("population"),
                               "cut_edges": cut_edges})


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: ANALYTICAL DIAGNOSIS — what does the Fiedler vector look like?
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 68)
print("PART 1: ANALYTICAL DIAGNOSIS")
print("=" * 68)

asn0 = dict(cddict)
bp0  = border_pairs_set(asn0, graph)
d1_id, d2_id = sorted(bp0)[0]
nodes = [n for n,d in asn0.items() if d in (d1_id, d2_id)]
n_sub = len(nodes)

L_sym, idx = build_sym_laplacian(graph, nodes)
eigs, evecs = la.eigh(L_sym)
fiedler_vec = evecs[:, 1]   # second eigenvector = Fiedler

x0 = np.array([1.0 if asn0[v]==d1_id else 0.0 for v in nodes])
pop = np.array([graph.nodes[v]["population"] for v in nodes])
p_bar = pop.sum() / 2.0

print(f"\nSubgraph: {n_sub} nodes  ({d1_id} ∪ {d2_id})")
print(f"L_sym eigenvalues: λ0={eigs[0]:.4f}  λ1(Fiedler)={eigs[1]:.4f}  λ_max={eigs[-1]:.4f}")
print(f"\nFiedler vector statistics:")
print(f"  range:  [{fiedler_vec.min():.3f}, {fiedler_vec.max():.3f}]")
print(f"  bimodal (fraction far from 0): {np.mean(np.abs(fiedler_vec)>0.05):.2%}")

# How does x* look for different α/β?
print(f"\n{'α/β ratio':>10} | {'x* near 0.5 (bad)':>18} | {'x* bimodal':>12} | "
      f"{'cut(hard)':>10} | {'Δcut':>6}")
print("-" * 68)

def solve_qp_direct(L_sym, x0, pop, p_bar, alpha, beta, epsilon=0.10):
    n = len(x0)
    try:
        m=gp.Model(); m.setParam("OutputFlag",0)
        x=m.addVars(n,lb=0,ub=1)
        obj=gp.QuadExpr()
        for i in range(n):
            for j in range(i,n):
                val=L_sym[i,j]
                if abs(val)>1e-12:
                    obj+=(alpha*val*x[i]*x[i] if i==j else 2*alpha*val*x[i]*x[j])
        for i in range(n): obj+=beta*(x[i]*x[i]-2*x0[i]*x[i])
        m.setObjective(obj,GRB.MINIMIZE)
        pe=gp.LinExpr(pop.tolist(),[x[i] for i in range(n)])
        m.addConstr(pe>=(1-epsilon)*p_bar); m.addConstr(pe<=(1+epsilon)*p_bar)
        m.optimize()
        if m.Status in (GRB.OPTIMAL,GRB.SUBOPTIMAL):
            return np.array([x[i].X for i in range(n)])
    except: pass
    return None

for alpha, beta in [(1,10),(2,5),(5,2),(10,1),(20,1),(50,1),(100,1)]:
    xs = solve_qp_direct(L_sym, x0, pop, p_bar, alpha, beta)
    if xs is None: continue
    near05     = np.mean((xs>0.3)&(xs<0.7))
    bimodal    = np.mean((xs<0.2)|(xs>0.8))
    hard_asn   = dict(asn0)
    for i,v in enumerate(nodes): hard_asn[v]=d1_id if xs[i]>=0.5 else d2_id
    cut_h      = n_cut_edges(graph, hard_asn)
    delta      = cut_h - n_cut_edges(graph, asn0)
    flag       = "★ BETTER" if delta<0 else ("same" if delta==0 else "worse")
    print(f"  α/β={alpha/beta:>6.1f} (α={alpha:>4},β={beta:>3}) | "
          f"{near05:>18.2%} | {bimodal:>12.2%} | {cut_h:>10} | {delta:>+6}  {flag}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: ROUNDING STRATEGY COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 68}")
print("PART 2: ROUNDING STRATEGY (α=20, β=1 fixed)")
print("=" * 68)

xs_good = solve_qp_direct(L_sym, x0, pop, p_bar, alpha=20, beta=1)
orig_cut = n_cut_edges(graph, asn0)

if xs_good is not None:
    print(f"\nx* bimodal fraction: {np.mean((xs_good<0.2)|(xs_good>0.8)):.2%}")
    print(f"x* histogram bins:  ", end="")
    hist,_ = np.histogram(xs_good, bins=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
    for h in hist: print(f"{h:3}", end=" ")
    print()

    print(f"\n{'Strategy':>30} | {'mean cut over 50 trials':>24} | {'std':>6} | {'Δ from orig':>12}")
    print("-" * 80)
    N_TRIALS = 50

    # 1. Hard threshold 0.5
    hard_asn = dict(asn0)
    for i,v in enumerate(nodes): hard_asn[v]=d1_id if xs_good[i]>=0.5 else d2_id
    cut_hard = n_cut_edges(graph, hard_asn)
    print(f"{'Hard threshold (t=0.5)':>30} | {cut_hard:>24.1f} | {'0.0':>6} | {cut_hard-orig_cut:>+12}")

    # 2. Randomised rounding (Bernoulli)  — various temperatures
    for T in [1.0, 0.5, 0.2, 0.1, 0.05]:
        cuts = []
        for _ in range(N_TRIALS):
            probs = 1/(1+np.exp(-(xs_good-0.5)/T))   # sigmoid temperature
            bits  = (np.random.rand(n_sub) < probs)
            trial_asn = dict(asn0)
            for i,v in enumerate(nodes): trial_asn[v]=d1_id if bits[i] else d2_id
            sub1=graph.subgraph([v for v,d in trial_asn.items() if d==d1_id])
            sub2=graph.subgraph([v for v,d in trial_asn.items() if d==d2_id])
            if nx.is_connected(sub1) and nx.is_connected(sub2):
                cuts.append(n_cut_edges(graph,trial_asn))
        if cuts:
            mc,sc=np.mean(cuts),np.std(cuts)
            print(f"{'Sigmoid rounding T='+str(T):>30} | {mc:>24.1f} | {sc:>6.2f} | {mc-orig_cut:>+12.1f}")

    # 3. Randomised rounding (raw Bernoulli)
    cuts = []
    for _ in range(N_TRIALS):
        bits = (np.random.rand(n_sub) < np.clip(xs_good,0.01,0.99))
        trial_asn = dict(asn0)
        for i,v in enumerate(nodes): trial_asn[v]=d1_id if bits[i] else d2_id
        sub1=graph.subgraph([v for v,d in trial_asn.items() if d==d1_id])
        sub2=graph.subgraph([v for v,d in trial_asn.items() if d==d2_id])
        if nx.is_connected(sub1) and nx.is_connected(sub2):
            cuts.append(n_cut_edges(graph,trial_asn))
    if cuts:
        mc,sc=np.mean(cuts),np.std(cuts)
        print(f"{'Raw Bernoulli(x*)':>30} | {mc:>24.1f} | {sc:>6.2f} | {mc-orig_cut:>+12.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 3: GRID SEARCH  (α, β, T_round, λ) — 20-step chains
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 68}")
print("PART 3: GRID SEARCH — 20-step QP-MH chains")
print("=" * 68)

def sigmoid_round(x_star, nodes, d1_id, d2_id, graph, assignment, T=0.1, max_tries=15):
    n = len(nodes); eps=1e-9
    probs = 1/(1+np.exp(-(x_star-0.5)/T))
    probs = np.clip(probs, eps, 1-eps)
    for _ in range(max_tries):
        bits = (np.random.rand(n) < probs)
        new_asn = dict(assignment)
        for i,v in enumerate(nodes): new_asn[v]=d1_id if bits[i] else d2_id
        sub1=graph.subgraph([v for v,d in new_asn.items() if d==d1_id])
        sub2=graph.subgraph([v for v,d in new_asn.items() if d==d2_id])
        if nx.is_connected(sub1) and nx.is_connected(sub2):
            log_q = np.sum(bits*np.log(probs)+(1-bits)*np.log(1-probs))
            return new_asn, log_q, bits, probs
    return None, None, None, None

def run_qp_mh_chain(graph, init_asn, alpha, beta, epsilon, lam, T_round, steps=20):
    """Run QP-MH chain with given parameters. Returns cut-edge trace."""
    asn = dict(init_asn)
    trace = [n_cut_edges(graph, asn)]
    accepts = 0
    for _ in range(steps):
        bp = border_pairs_set(asn, graph)
        if not bp: trace.append(trace[-1]); continue
        d1_id,d2_id = random.choice(sorted(bp))
        nodes = [n for n,d in asn.items() if d in (d1_id,d2_id)]
        n_nodes = len(nodes)
        L_sym_s,_ = build_sym_laplacian(graph, nodes)
        x0_s = np.array([1.0 if asn[v]==d1_id else 0.0 for v in nodes])
        pop_s= np.array([graph.nodes[v]["population"] for v in nodes])
        p_bar_s=pop_s.sum()/2.0
        xs = solve_qp_direct(L_sym_s, x0_s, pop_s, p_bar_s, alpha, beta, epsilon)
        if xs is None: trace.append(trace[-1]); continue

        # Forward rounding
        prop,lq_fwd,bits,probs_fwd = sigmoid_round(xs,nodes,d1_id,d2_id,graph,asn,T_round)
        if prop is None: trace.append(trace[-1]); continue

        # Reverse QP
        xs_rev = solve_qp_direct(L_sym_s,
                                  np.array([1.0 if prop[v]==d1_id else 0.0 for v in nodes]),
                                  pop_s, p_bar_s, alpha, beta, epsilon)
        if xs_rev is None: trace.append(trace[-1]); continue
        probs_rev = np.clip(1/(1+np.exp(-(xs_rev-0.5)/T_round)), 1e-9, 1-1e-9)
        orig_bits = np.array([1.0 if asn[v]==d1_id else 0.0 for v in nodes])
        lq_rev = np.sum(orig_bits*np.log(probs_rev)+(1-orig_bits)*np.log(1-probs_rev))

        # MH
        n_bp_cur  = len(bp)
        n_bp_prop = max(len(border_pairs_set(prop,graph)),1)
        cut_cur   = trace[-1]
        cut_prop  = n_cut_edges(graph,prop)
        log_alpha = (-lam*(cut_prop-cut_cur) + (lq_rev-lq_fwd)
                     + np.log(n_bp_cur)-np.log(n_bp_prop))
        if np.log(random.random()+1e-300) < log_alpha:
            asn = prop; accepts += 1
        trace.append(n_cut_edges(graph,asn))
    return trace, accepts/steps

# Parameter grid
alphas   = [5, 10, 20, 50]
betas    = [0.5, 1, 2]
T_rounds = [0.05, 0.1, 0.2]
lams     = [0.02, 0.05, 0.1]
epsilon  = 0.10

results = []
N_REPS  = 3    # repeat each config to reduce noise

print(f"\n{'α':>5} {'β':>5} {'T':>6} {'λ':>6} | "
      f"{'mean_cut':>10} {'final_cut':>10} {'accept%':>9} | note")
print("-" * 70)

best_score = 1e9; best_params = None

for alpha,beta,T_round,lam in itertools.product(alphas,betas,T_rounds,lams):
    all_traces=[]; all_accepts=[]
    for rep in range(N_REPS):
        random.seed(SEED+rep); np.random.seed(SEED+rep)
        tr,ar=run_qp_mh_chain(graph,cddict,alpha,beta,epsilon,lam,T_round,steps=20)
        all_traces.append(tr); all_accepts.append(ar)
    mean_cut  = np.mean([t[-1] for t in all_traces])
    mean_full = np.mean([np.mean(t) for t in all_traces])
    mean_acc  = np.mean(all_accepts)*100
    score     = mean_cut   # lower final cut = better compactness
    flag = ""
    if score < best_score:
        best_score=score; best_params=(alpha,beta,T_round,lam)
        flag = "★ BEST"
    print(f"  {alpha:>4} {beta:>4} {T_round:>6} {lam:>6} | "
          f"{mean_full:>10.1f} {mean_cut:>10.1f} {mean_acc:>8.1f}% | {flag}")
    results.append({"alpha":alpha,"beta":beta,"T":T_round,"lam":lam,
                    "mean_cut":mean_full,"final_cut":mean_cut,"accept":mean_acc,
                    "traces":all_traces})

print(f"\n★ Best params: α={best_params[0]}, β={best_params[1]}, "
      f"T={best_params[2]}, λ={best_params[3]}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 4: COMPARE BEST QP-MH vs ReCom vs Hybrid (20 steps)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 68}")
print("PART 4: BEST QP-MH vs ReCom vs Hybrid (20 steps, 5 replications)")
print("=" * 68)

a_best,b_best,T_best,l_best = best_params
N_REPS2 = 5

recom_traces_all=[]; qp_traces_all=[]; hyb_traces_all=[]

for rep in range(N_REPS2):
    random.seed(SEED+rep); np.random.seed(SEED+rep)

    # ReCom
    part = make_partition(gc_graph, cddict)
    tr=[n_cut_edges(graph,cddict)]
    for _ in range(20):
        part=recom(part,"population",pop_target,0.10,node_repeats=1)
        tr.append(n_cut_edges(graph,dict(part.assignment)))
    recom_traces_all.append(tr)

    # Best QP-MH
    random.seed(SEED+rep); np.random.seed(SEED+rep)
    tr,_=run_qp_mh_chain(graph,cddict,a_best,b_best,epsilon,l_best,T_best,steps=20)
    qp_traces_all.append(tr)

    # Hybrid
    random.seed(SEED+rep); np.random.seed(SEED+rep)
    asn_h=dict(cddict); part_h=make_partition(gc_graph,cddict)
    tr_h=[n_cut_edges(graph,cddict)]
    for _ in range(20):
        part_h=recom(part_h,"population",pop_target,0.10,node_repeats=1)
        asn_h=dict(part_h.assignment)
        xs_h=None
        bp_h=border_pairs_set(asn_h,graph)
        if bp_h:
            d1h,d2h=random.choice(sorted(bp_h))
            nds_h=[n for n,d in asn_h.items() if d in (d1h,d2h)]
            L_h,_=build_sym_laplacian(graph,nds_h)
            x0_h=np.array([1.0 if asn_h[v]==d1h else 0.0 for v in nds_h])
            pop_h=np.array([graph.nodes[v]["population"] for v in nds_h])
            xs_h=solve_qp_direct(L_h,x0_h,pop_h,pop_h.sum()/2,a_best,b_best,epsilon)
        if xs_h is not None:
            prop_h,lqf,_,pf=sigmoid_round(xs_h,nds_h,d1h,d2h,graph,asn_h,T_best)
            if prop_h is not None:
                # quick MH check
                xs_rev_h=solve_qp_direct(L_h,
                    np.array([1.0 if prop_h[v]==d1h else 0.0 for v in nds_h]),
                    pop_h,pop_h.sum()/2,a_best,b_best,epsilon)
                if xs_rev_h is not None:
                    pr=np.clip(1/(1+np.exp(-(xs_rev_h-0.5)/T_best)),1e-9,1-1e-9)
                    ob=np.array([1.0 if asn_h[v]==d1h else 0.0 for v in nds_h])
                    lqr=np.sum(ob*np.log(pr)+(1-ob)*np.log(1-pr))
                    la_mh=(-l_best*(n_cut_edges(graph,prop_h)-n_cut_edges(graph,asn_h))
                           +(lqr-lqf))
                    if np.log(random.random()+1e-300)<la_mh:
                        asn_h=prop_h; part_h=make_partition(gc_graph,asn_h)
        tr_h.append(n_cut_edges(graph,asn_h))
    hyb_traces_all.append(tr_h)

def mean_trace(traces):
    return np.mean([t for t in traces], axis=0)

mr=mean_trace(recom_traces_all)
mq=mean_trace(qp_traces_all)
mh=mean_trace(hyb_traces_all)
steps=np.arange(21)

print(f"\n{'':>30} {'init':>8} {'step 10':>8} {'step 20':>8} {'Δ':>8}")
for label,tr_all in [("ReCom",recom_traces_all),
                     (f"QP-MH (α={a_best},β={b_best},T={T_best})",qp_traces_all),
                     ("Hybrid",hyb_traces_all)]:
    mt=mean_trace(tr_all)
    print(f"  {label:<38} {mt[0]:>8.1f} {mt[10]:>8.1f} {mt[20]:>8.1f} "
          f"{mt[20]-mt[0]:>+8.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 5: PLOTS
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20,14))
gs  = gridspec.GridSpec(2,2,figure=fig,hspace=0.35,wspace=0.30)

# ── Panel A: x* histograms for bad vs good α/β ───────────────────────────────
ax_hist = fig.add_subplot(gs[0,0])
xs_bad  = solve_qp_direct(L_sym,x0,pop,p_bar,alpha=1,beta=10)
xs_good2= solve_qp_direct(L_sym,x0,pop,p_bar,alpha=a_best,beta=b_best)
if xs_bad is not None:
    ax_hist.hist(xs_bad, bins=20, alpha=0.6, color="tomato",
                 label=f"α=1, β=10 (default — bad)\n"
                       f"{np.mean((xs_bad>0.3)&(xs_bad<0.7)):.0%} near 0.5")
if xs_good2 is not None:
    ax_hist.hist(xs_good2, bins=20, alpha=0.6, color="steelblue",
                 label=f"α={a_best}, β={b_best} (tuned)\n"
                       f"{np.mean((xs_good2>0.3)&(xs_good2<0.7)):.0%} near 0.5")
ax_hist.axvline(0.5, color="k", ls="--", lw=1)
ax_hist.set_xlabel("x* value", fontsize=12)
ax_hist.set_ylabel("Count", fontsize=12)
ax_hist.set_title("A.  x* Distribution\n"
                  "Values near 0.5 → rounding noise → worse partitions", fontsize=11)
ax_hist.legend(fontsize=9)
ax_hist.grid(True, alpha=0.3)

# ── Panel B: rounding temperature effect ─────────────────────────────────────
ax_round = fig.add_subplot(gs[0,1])
if xs_good2 is not None:
    x_range = np.linspace(0,1,200)
    for T,col in [(1.0,"#d62728"),(0.2,"#ff7f0e"),(0.1,"#2ca02c"),(0.05,"#1f77b4")]:
        p_sigmoid = 1/(1+np.exp(-(x_range-0.5)/T))
        ax_round.plot(x_range, p_sigmoid, color=col, lw=2, label=f"T={T}")
ax_round.axvline(0.5, color="k", ls="--", lw=0.8)
ax_round.axhline(0.5, color="k", ls="--", lw=0.8)
ax_round.set_xlabel("x* value", fontsize=12)
ax_round.set_ylabel("P(assign to D1)", fontsize=12)
ax_round.set_title("B.  Sigmoid Rounding Temperature\n"
                   "Small T → near-deterministic (preserves QP signal)", fontsize=11)
ax_round.legend(fontsize=10)
ax_round.grid(True, alpha=0.3)

# ── Panel C: cut-edge traces ──────────────────────────────────────────────────
ax_trace = fig.add_subplot(gs[1,0])
COLS = {"ReCom":"tomato","QP-MH":"steelblue","Hybrid":"mediumseagreen"}
for label,mt,col in [("ReCom",mr,"tomato"),
                     (f"QP-MH (tuned)",mq,"steelblue"),
                     ("Hybrid",mh,"mediumseagreen")]:
    ax_trace.plot(steps, mt, color=col, lw=2.5, marker="o", ms=4, label=label)
ax_trace.set_xlabel("Step"); ax_trace.set_ylabel("Cut Edges (mean over 5 reps)")
ax_trace.set_title("C.  Cut-Edge Trace  (lower = more compact)\n"
                   "Tuned QP-MH should now decrease cut edges", fontsize=11)
ax_trace.legend(); ax_trace.grid(True,alpha=0.3)

# ── Panel D: heatmap of final cut edges over α-β grid ────────────────────────
ax_heat = fig.add_subplot(gs[1,1])
# extract best T and lam, vary α and β
heat_alphas=[5,10,20,50]; heat_betas=[0.5,1,2]
heat_data=np.zeros((len(heat_betas),len(heat_alphas)))
for i,hb in enumerate(heat_betas):
    for j,ha in enumerate(heat_alphas):
        scores=[]
        for rep in range(3):
            random.seed(SEED+rep); np.random.seed(SEED+rep)
            tr,_=run_qp_mh_chain(graph,cddict,ha,hb,epsilon,l_best,T_best,steps=20)
            scores.append(tr[-1])
        heat_data[i,j]=np.mean(scores)

im=ax_heat.imshow(heat_data,cmap="RdYlGn_r",aspect="auto")
ax_heat.set_xticks(range(len(heat_alphas))); ax_heat.set_xticklabels(heat_alphas)
ax_heat.set_yticks(range(len(heat_betas)));  ax_heat.set_yticklabels(heat_betas)
ax_heat.set_xlabel("α  (Laplacian weight)", fontsize=12)
ax_heat.set_ylabel("β  (prior weight)", fontsize=12)
ax_heat.set_title("D.  Final Cut Edges Heatmap  (α, β)\n"
                  "Green = fewer cut edges = more compact", fontsize=11)
plt.colorbar(im, ax=ax_heat, shrink=0.8)
for i in range(len(heat_betas)):
    for j in range(len(heat_alphas)):
        ax_heat.text(j,i,f"{heat_data[i,j]:.0f}",ha="center",va="center",
                     fontsize=10, color="white" if heat_data[i,j]>heat_data.mean() else "black")

plt.suptitle("QP Parameter Tuning Diagnosis", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("qp_parameter_tuning.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: qp_parameter_tuning.png")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"""
{'=' * 68}
PARAMETER TUNING SUMMARY
{'=' * 68}

Root causes of QP performing worse with default parameters
──────────────────────────────────────────────────────────
① β=10 >> α=1  →  prior term dominates, x* ≈ x0 + noise.
  Many x*_i land near 0.5.  Randomised rounding of 0.5 = coin flip.
  Fix: α >> β  (spectral bisection regime).

② Raw Bernoulli rounding destroys x* signal.
  Fix: sigmoid rounding with low temperature T (0.05–0.1).
  At T=0.05, p(x*=0.7) = sigmoid(0.2/0.05) = 0.98  (nearly deterministic).

③ λ (MH temperature) too low → accepts bad moves; too high → rejects all.
  Fix: tune λ so that accept rate is 20–50%.

Recommended parameters
──────────────────────
  α     = {a_best}    (Laplacian weight — let spectral bisection dominate)
  β     = {b_best}    (prior weight — weak anchor to x0)
  T     = {T_best}   (rounding temperature — near-deterministic)
  λ     = {l_best}   (MH compactness bias)
  ε     = 0.10   (population tolerance — unchanged)

Intuition for α >> β
──────────────────────────────────────────────────────────────────────────────
With α >> β, the QP solution approaches the Fiedler vector of the merged
subgraph Laplacian.  The Fiedler vector is the spectral bisection solution —
it minimises the normalised cut, which directly measures compactness.
Hard-thresholding or low-T sigmoid rounding of the Fiedler vector gives
the Cheeger-optimal compact split of the merged district pair.
This is exactly what we want QP to do: find the most compact re-split
of two merged districts, which ReCom cannot do by random spanning-tree cuts.
{'=' * 68}
""")
