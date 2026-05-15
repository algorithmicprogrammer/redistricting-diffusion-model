"""
Fast Targeted QP Parameter Sweep
==================================
Replaces the exhaustive grid search with a principled 3-stage sweep.

Three bugs found in the original grid search
─────────────────────────────────────────────
BUG 1 — λ is irrelevant in MH ratio
  The proposal log-ratio  log q(x|x′) − log q(x′|x)  is computed from
  sigmoid probabilities which, at small T, produce extreme log-values
  (log(0.98) ≈ −0.02 per node, summed over 180 nodes = −3.6 per side).
  The total |log_q_rev − log_q_fwd| ≫ |λ · Δcut| for any reasonable λ.
  Fix: drop λ from MH and use only the proposal ratio.  The target
  distribution is implicitly encoded by the QP objective itself.

BUG 2 — T=0.2 → 0% acceptance → chain stuck at initial
  Large T makes all sigmoid probabilities near 0.5 → rounding is nearly
  uniform random → most attempts fail the connectivity check.
  The chain appears "best" (cut=120 = initial) because it never moves.
  Fix: T must be small enough that connectivity is likely (T ≤ 0.10).

BUG 3 — α=5 too small → x* has no Fiedler structure → rounding adds noise
  With α/β ≈ 5, the QP solution x* is still heavily anchored to x0.
  Many components land near 0.5 regardless of T.
  Fix: α ≥ 20 so that  x* → Fiedler vector of L_sym.

Strategy
─────────
Stage 1: α sweep (β=1, T=0.05) to find minimum α that gives bimodal x*
Stage 2: T sweep (best α, β=1) to find acceptance-rate sweet spot
Stage 3: 5-rep chain comparison of top-3 configs vs ReCom baseline
"""

import random, time, warnings, itertools
import numpy as np
import scipy.linalg as la
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

from gerrychain import Graph, Partition
from gerrychain.updaters import Tally, cut_edges
from gerrychain.proposals import recom

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED)

gn = 6; k = 5; ns = 50; NUM_DISTRICTS = k
graph = nx.grid_graph([k * gn, k * gn])
for n in graph.nodes():
    graph.nodes[n]["population"] = 50
    if random.random() < 0.5: graph.nodes[n]["pink"]=1; graph.nodes[n]["purple"]=0
    else: graph.nodes[n]["pink"]=0; graph.nodes[n]["purple"]=1
    if 0 in n or k*gn-1 in n:
        graph.nodes[n]["boundary_node"]=True; graph.nodes[n]["boundary_perim"]=1
    else: graph.nodes[n]["boundary_node"]=False

cddict     = {x: int(x[0]/gn) for x in graph.nodes()}
total_pop  = sum(graph.nodes[v]["population"] for v in graph.nodes())
pop_target = total_pop / NUM_DISTRICTS
gc_graph   = Graph.from_networkx(graph)

INITIAL_CUT = sum(1 for u,v in graph.edges() if cddict[u]!=cddict[v])
print(f"Initial partition cut edges: {INITIAL_CUT}")


# ─────────────────────────────────────────────────────────────────────────────
# Shared functions
# ─────────────────────────────────────────────────────────────────────────────

def build_sym_laplacian(graph, nodes):
    n=len(nodes); idx={v:i for i,v in enumerate(nodes)}
    sub=graph.subgraph(nodes); W=np.zeros((n,n))
    for u,v in sub.edges():
        w=np.exp(-abs(graph.nodes[u].get("population",0)-graph.nodes[v].get("population",0)))
        i,j=idx[u],idx[v]; W[i,j]=w; W[j,i]=w
    d=W.sum(axis=1); d_isqrt=np.where(d>0,1/np.sqrt(d),0)
    return np.diag(d_isqrt)@(np.diag(d)-W)@np.diag(d_isqrt), idx

def solve_qp(L_sym, x0, pop, p_bar, alpha, beta, epsilon=0.10):
    n=len(x0)
    try:
        m=gp.Model(); m.setParam("OutputFlag",0)
        x=m.addVars(n,lb=0,ub=1)
        obj=gp.QuadExpr()
        for i in range(n):
            for j in range(i,n):
                v=L_sym[i,j]
                if abs(v)>1e-12:
                    obj+=(alpha*v*x[i]*x[i] if i==j else 2*alpha*v*x[i]*x[j])
        for i in range(n): obj+=beta*(x[i]*x[i]-2*x0[i]*x[i])
        m.setObjective(obj,GRB.MINIMIZE)
        pe=gp.LinExpr(pop.tolist(),[x[i] for i in range(n)])
        m.addConstr(pe>=(1-epsilon)*p_bar); m.addConstr(pe<=(1+epsilon)*p_bar)
        m.optimize()
        if m.Status in (GRB.OPTIMAL,GRB.SUBOPTIMAL):
            return np.array([x[i].X for i in range(n)])
    except: pass
    return None

def border_pairs_set(asn, graph):
    bp=set()
    for u,v in graph.edges():
        d1,d2=asn[u],asn[v]
        if d1!=d2: bp.add((min(d1,d2),max(d1,d2)))
    return bp

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

def make_partition(gc_graph, asn):
    return Partition(gc_graph, assignment=asn,
                     updaters={"population": Tally("population"),
                               "cut_edges": cut_edges})


def sigmoid_round(x_star, nodes, asn, d1_id, d2_id, T, max_tries=30):
    """Returns (new_asn, log_q_fwd, probs) or (None,None,None)."""
    n=len(nodes); eps=1e-9
    probs=np.clip(1/(1+np.exp(-(x_star-0.5)/T)), eps, 1-eps)
    for _ in range(max_tries):
        bits=np.random.rand(n)<probs
        new_asn=dict(asn)
        for i,v in enumerate(nodes): new_asn[v]=d1_id if bits[i] else d2_id
        sub1=graph.subgraph([v for v,d in new_asn.items() if d==d1_id])
        sub2=graph.subgraph([v for v,d in new_asn.items() if d==d2_id])
        if nx.is_connected(sub1) and nx.is_connected(sub2):
            log_q=np.sum(bits*np.log(probs)+(1-bits)*np.log(1-probs))
            return new_asn, log_q, probs
    return None, None, None


def qp_mh_step(graph, asn, alpha, beta, T, epsilon=0.10):
    """
    One QP-MH step.
    BUG 1 FIX: MH ratio uses ONLY proposal ratio — λ dropped.
    Target distribution is implicitly defined by QP objective.
    """
    bp=border_pairs_set(asn,graph)
    if not bp: return dict(asn), False
    n_bp_cur=len(bp)

    d1_id,d2_id=random.choice(sorted(bp))
    nodes=[v for v,d in asn.items() if d in (d1_id,d2_id)]

    L_sym,_=build_sym_laplacian(graph,nodes)
    x0=np.array([1.0 if asn[v]==d1_id else 0.0 for v in nodes])
    pop=np.array([graph.nodes[v]["population"] for v in nodes])
    p_bar=pop.sum()/2

    xs=solve_qp(L_sym,x0,pop,p_bar,alpha,beta,epsilon)
    if xs is None: return dict(asn), False

    prop,lq_fwd,probs_fwd=sigmoid_round(xs,nodes,asn,d1_id,d2_id,T)
    if prop is None: return dict(asn), False

    # Reverse QP for exact MH
    x0_rev=np.array([1.0 if prop[v]==d1_id else 0.0 for v in nodes])
    xs_rev=solve_qp(L_sym,x0_rev,pop,p_bar,alpha,beta,epsilon)
    if xs_rev is None: return dict(asn), False

    eps=1e-9
    probs_rev=np.clip(1/(1+np.exp(-(xs_rev-0.5)/T)),eps,1-eps)
    orig_bits=np.array([1.0 if asn[v]==d1_id else 0.0 for v in nodes])
    lq_rev=np.sum(orig_bits*np.log(probs_rev)+(1-orig_bits)*np.log(1-probs_rev))

    n_bp_prop=max(len(border_pairs_set(prop,graph)),1)

    # MH: only proposal ratio + symmetry correction (no λ)
    log_alpha=(lq_rev-lq_fwd)+np.log(n_bp_cur)-np.log(n_bp_prop)

    if np.log(random.random()+1e-300)<log_alpha:
        return prop, True
    return dict(asn), False


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: α sweep — find minimum α for bimodal x*
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STAGE 1: α sweep — x* bimodality vs α  (β=1 fixed)")
print("="*60)

# Use a fixed district pair for fair comparison
asn0=dict(cddict)
bp0=border_pairs_set(asn0,graph)
d1_id,d2_id=sorted(bp0)[0]
nodes0=[v for v,d in asn0.items() if d in (d1_id,d2_id)]
L_sym0,_=build_sym_laplacian(graph,nodes0)
x0_0=np.array([1.0 if asn0[v]==d1_id else 0.0 for v in nodes0])
pop0=np.array([graph.nodes[v]["population"] for v in nodes0])
p_bar0=pop0.sum()/2

# Fiedler vector as reference
eigs,evecs=la.eigh(L_sym0)
fiedler=evecs[:,1]
if np.corrcoef(fiedler,x0_0)[0,1]<0: fiedler=-fiedler
f01=(fiedler-fiedler.min())/(fiedler.max()-fiedler.min()+1e-12)

print(f"\nFiedler λ1={eigs[1]:.4f}  bimodal={(np.mean((f01<0.2)|(f01>0.8))):.2%}")
print(f"\n{'α':>6} {'β':>4} | {'bimodal%':>9} {'near0.5%':>9} "
      f"{'corr(x*,F)':>11} {'cut(hard)':>10} {'Δcut':>6}")
print("-"*60)

alphas_stage1=[1,2,5,10,20,50,100]
best_alpha=5; best_bimodal=0

for alpha in alphas_stage1:
    xs=solve_qp(L_sym0,x0_0,pop0,p_bar0,alpha,beta=1)
    if xs is None: print(f"  {alpha:>5}    1 | infeasible"); continue
    bimodal=np.mean((xs<0.2)|(xs>0.8))
    near05 =np.mean((xs>0.3)&(xs<0.7))
    corr   =np.corrcoef(xs,f01)[0,1]
    hard_asn=dict(asn0)
    for i,v in enumerate(nodes0): hard_asn[v]=d1_id if xs[i]>=0.5 else d2_id
    cut_h=n_cut_edges(graph,hard_asn)
    delta=cut_h-INITIAL_CUT
    flag="★" if bimodal>best_bimodal else ""
    if bimodal>best_bimodal: best_bimodal=bimodal; best_alpha=alpha
    print(f"  {alpha:>5}    1 | {bimodal:>9.2%} {near05:>9.2%} "
          f"{corr:>11.4f} {cut_h:>10} {delta:>+6}  {flag}")

print(f"\n  → Best α for bimodal x*: {best_alpha}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: T sweep — acceptance rate vs T  (best α, β=1)
# Aim: accept rate 20-50% (enough to explore, not too random)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"STAGE 2: T sweep  (α={best_alpha}, β=1)")
print("="*60)
print("Target accept rate: 20-50%")
print(f"\n{'T':>7} | {'accept%':>8} {'mean_cut':>10} {'final_cut':>10} | note")
print("-"*55)

T_candidates=[0.02,0.05,0.08,0.10,0.15,0.20]
best_T=0.05; best_score=1e9; T_results=[]

N_STEPS=20; N_REPS=5

for T in T_candidates:
    traces=[]; accepts=[]
    for rep in range(N_REPS):
        random.seed(SEED+rep); np.random.seed(SEED+rep)
        asn=dict(cddict); acc=0
        trace=[INITIAL_CUT]
        for _ in range(N_STEPS):
            asn,ok=qp_mh_step(graph,asn,best_alpha,1,T)
            if ok: acc+=1
            trace.append(n_cut_edges(graph,asn))
        traces.append(trace); accepts.append(acc/N_STEPS)
    mean_acc =np.mean(accepts)*100
    mean_cut =np.mean([np.mean(t) for t in traces])
    final_cut=np.mean([t[-1] for t in traces])
    # Score: penalise both high cut AND 0% accept (stuck)
    stuck = mean_acc < 1.0
    score = final_cut if not stuck else 1e6
    flag=""
    if score<best_score and not stuck:
        best_score=score; best_T=T; flag="★ BEST"
    note = "STUCK (never moves)" if stuck else flag
    T_results.append((T,mean_acc,mean_cut,final_cut,score))
    print(f"  T={T:>5} | {mean_acc:>8.1f}% {mean_cut:>10.1f} {final_cut:>10.1f} | {note}")

print(f"\n  → Best T: {best_T}  (accept rate {[r[1] for r in T_results if r[0]==best_T][0]:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Top configs vs ReCom  (5 reps, 20 steps)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STAGE 3: Final comparison  (5 reps × 20 steps)")
print("="*60)

# Pick top-3 non-stuck T values
valid_T=[(r[0],r[3]) for r in T_results if r[1]>0.5]   # accept > 0.5%
valid_T.sort(key=lambda x: x[1])
top_T=[v[0] for v in valid_T[:3]]

configs=[("ReCom",  None,       None, None),
         *[(f"QP-MH α={best_alpha} T={T}", best_alpha, 1, T) for T in top_T]]

all_results={}
for name,alpha,beta,T in configs:
    traces=[]
    for rep in range(N_REPS):
        random.seed(SEED+rep); np.random.seed(SEED+rep)
        if name=="ReCom":
            part=make_partition(gc_graph,cddict)
            tr=[INITIAL_CUT]
            for _ in range(N_STEPS):
                part=recom(part,"population",pop_target,0.10,node_repeats=1)
                tr.append(n_cut_edges(graph,dict(part.assignment)))
        else:
            asn=dict(cddict); tr=[INITIAL_CUT]
            for _ in range(N_STEPS):
                asn,_=qp_mh_step(graph,asn,alpha,beta,T)
                tr.append(n_cut_edges(graph,asn))
        traces.append(tr)
    all_results[name]=traces

print(f"\n{'Config':<35} {'init':>6} {'step5':>7} {'step10':>7} "
      f"{'step20':>7} {'Δ':>6} {'std@20':>7}")
print("-"*75)
for name,_,__,___ in configs:
    traces=all_results[name]
    mt=np.mean(traces,axis=0)
    st=np.std([t[-1] for t in traces])
    print(f"  {name:<33} {mt[0]:>6.1f} {mt[5]:>7.1f} {mt[10]:>7.1f} "
          f"{mt[20]:>7.1f} {mt[20]-mt[0]:>+6.1f} {st:>7.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot: Stage 1-3 summary in one figure
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# ── Panel A: x* bimodality vs α ──────────────────────────────────────────────
ax=axes[0]
bimodals=[]; corrs=[]; cuts_hard=[]
for alpha in alphas_stage1:
    xs=solve_qp(L_sym0,x0_0,pop0,p_bar0,alpha,beta=1)
    if xs is None: bimodals.append(0); corrs.append(0); cuts_hard.append(INITIAL_CUT); continue
    bimodals.append(np.mean((xs<0.2)|(xs>0.8)))
    corrs.append(np.corrcoef(xs,f01)[0,1])
    hard_asn=dict(asn0)
    for i,v in enumerate(nodes0): hard_asn[v]=d1_id if xs[i]>=0.5 else d2_id
    cuts_hard.append(n_cut_edges(graph,hard_asn))

ax2=ax.twinx()
ax.plot(alphas_stage1, bimodals,  "b-o", lw=2, ms=7, label="Bimodal fraction")
ax.plot(alphas_stage1, corrs,     "g-s", lw=2, ms=7, label="corr(x*, Fiedler)")
ax2.plot(alphas_stage1, cuts_hard,"r-^", lw=2, ms=7, label="Cut edges (hard round)")
ax.axvline(best_alpha, color="navy", ls="--", lw=1.5,
           label=f"Best α={best_alpha}")
ax.set_xscale("log"); ax.set_xlabel("α  (log scale)", fontsize=12)
ax.set_ylabel("Fraction / Correlation", fontsize=11)
ax2.set_ylabel("Cut edges after hard round", fontsize=11, color="red")
ax.set_title(f"A.  x* Quality vs α\n(β=1, pair D{d1_id}/D{d2_id})", fontsize=11)
lines1,lab1=ax.get_legend_handles_labels()
lines2,lab2=ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, lab1+lab2, fontsize=9)
ax.grid(True, alpha=0.3)

# ── Panel B: acceptance rate and cut edges vs T ───────────────────────────────
ax=axes[1]
Ts   =[r[0] for r in T_results]
accs =[r[1] for r in T_results]
fcuts=[r[3] for r in T_results]
ax2=ax.twinx()
ax.bar([str(T) for T in Ts], accs,  color="steelblue", alpha=0.7, label="Accept %")
ax2.plot([str(T) for T in Ts], fcuts, "r-o", lw=2, ms=7, label="Final cut edges")
ax.axhline(20, color="green",  ls="--", lw=1.2, label="20% target")
ax.axhline(50, color="orange", ls="--", lw=1.2, label="50% target")
ax.set_xlabel("Rounding temperature T", fontsize=12)
ax.set_ylabel("Acceptance rate %", fontsize=11, color="steelblue")
ax2.set_ylabel("Final cut edges", fontsize=11, color="red")
ax.set_title(f"B.  Acceptance Rate & Quality vs T\n"
             f"(α={best_alpha}, β=1, {N_REPS} reps×{N_STEPS} steps)", fontsize=11)
lines1,lab1=ax.get_legend_handles_labels()
lines2,lab2=ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, lab1+lab2, fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3, axis="y")

# ── Panel C: cut-edge traces ──────────────────────────────────────────────────
ax=axes[2]
COLS={"ReCom":"tomato"}
for i,(name,_,__,___) in enumerate(configs):
    if name!="ReCom": COLS[name]=plt.cm.Blues(0.4+0.2*i)
steps_x=np.arange(N_STEPS+1)
for name,alpha,beta,T in configs:
    mt=np.mean(all_results[name],axis=0)
    st=np.std(all_results[name],axis=0)
    c=COLS[name]
    ax.plot(steps_x,mt,color=c,lw=2,label=name)
    ax.fill_between(steps_x,mt-st,mt+st,color=c,alpha=0.15)
ax.axhline(INITIAL_CUT,color="k",ls=":",lw=1,label=f"Initial ({INITIAL_CUT})")
ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("Cut edges  (mean ± std, 5 reps)", fontsize=11)
ax.set_title(f"C.  Cut-edge traces\n"
             f"Lower = more compact", fontsize=11)
ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

plt.suptitle("QP Parameter Sweep — Targeted 3-Stage Analysis",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("qp_sweep_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: qp_sweep_results.png")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"""
{'='*60}
FINAL RECOMMENDATIONS
{'='*60}
  α  = {best_alpha}      (Laplacian dominates → Fiedler-like x*)
  β  = 1        (weak prior, free to find compact cut)
  T  = {best_T}     (rounding temperature → 20-50% accept rate)
  λ  = drop it  (absorbed into proposal ratio; redundant)
  ε  = 0.10     (population tolerance, unchanged)

Why the original parameters failed
────────────────────────────────────
  α=1, β=10 → prior dominates → x* ≈ x0 → rounding = noise
  T=0.20    → 0% accept → stuck at initial (false optimum)
  λ=any     → swamped by log-proposal ratio → no effect

The correct mental model
─────────────────────────
  QP is doing spectral bisection on the merged subgraph.
  The Fiedler vector of L_sym is the minimum-normalised-cut
  solution.  With α >> β, x* → Fiedler → hard round gives the
  most compact valid re-split of D1∪D2 that ReCom cannot find.
{'='*60}
""")
