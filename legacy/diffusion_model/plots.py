"""
plots.py — All matplotlib figures for the redistricting chain comparison.

Public API
----------
plot_initial_partition(graph, assignment, ns, plots_dir)
plot_final_partitions(graph, assignments, ns, plots_dir)
plot_boundary_comparison(reference_assignment, initial_assignment,
                         recom_final, qp_final, mew_final, hyb_final,
                         plot_boundary_nodes, plots_dir)
plot_metric_comparison(recom_metrics, qp_metrics, mew_metrics, hyb_metrics,
                       plots_dir)
plot_mixing_time(recom_trace, qp_trace, mew_trace, hyb_trace,
                 recom_wt, qp_wt, mew_wt, hyb_wt,
                 LONG, MAX_LAG, plots_dir)

Analysis helpers (used by plot_mixing_time)
-------------------------------------------
acf(trace, max_lag)
tau_and_ess(trace, max_lag)
ess_per_second(ess_val, wall_times)
running_mean(trace)
tv_proxy(trace, window)
"""

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────────────────────────────────────
# Analysis helpers
# ─────────────────────────────────────────────────────────────────────────────

def acf(trace, max_lag):
    n, mu, var = len(trace), trace.mean(), trace.var()
    if var < 1e-12:
        return np.zeros(max_lag)
    out = [1.0]
    for lag in range(1, max_lag):
        cov = np.mean((trace[:n - lag] - mu) * (trace[lag:] - mu))
        out.append(cov / var)
    return np.array(out)


def tau_and_ess(trace, max_lag):
    r   = acf(trace, max_lag)
    tau = 1.0
    for k in range(1, max_lag):
        if r[k] <= 0:
            break
        tau += 2 * r[k]
    return tau, len(trace) / tau


def ess_per_second(ess_val, wall_times):
    return ess_val / wall_times.sum()


def running_mean(t):
    return np.cumsum(t) / np.arange(1, len(t) + 1)


def tv_proxy(trace, window=20):
    rm = running_mean(trace)
    return np.array([rm[max(0, t - window):t].std() for t in range(window, len(rm))])


# ─────────────────────────────────────────────────────────────────────────────
# Figure: initial partition
# ─────────────────────────────────────────────────────────────────────────────

def plot_initial_partition(graph, assignment, ns, plots_dir):
    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw(
        graph,
        pos={x: x for x in graph.nodes()},
        node_color=[assignment[x] for x in graph.nodes()],
        node_size=ns, node_shape="s", cmap="tab20", ax=ax,
    )
    ax.set_title("Initial Partition", fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_dir / "initial_partition.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: initial_partition.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure: final partitions (1 × 4)
# ─────────────────────────────────────────────────────────────────────────────

def plot_final_partitions(graph, assignments, ns, plots_dir):
    """
    assignments : list of (asn_dict, title_str)
    """
    fig, axes = plt.subplots(1, len(assignments), figsize=(6 * len(assignments), 6))
    pos = {x: x for x in graph.nodes()}
    for ax, (asn, title) in zip(axes, assignments):
        nx.draw(
            graph, pos=pos,
            node_color=[asn[x] for x in graph.nodes()],
            node_size=ns, node_shape="s", cmap="tab20", ax=ax,
        )
        ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(plots_dir / "final_partitions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: final_partitions.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure: boundary comparison (2 × 3)
# ─────────────────────────────────────────────────────────────────────────────

def plot_boundary_comparison(
    reference_assignment, initial_assignment,
    recom_final, qp_final, mew_final, hyb_final,
    plot_boundary_nodes_fn, plots_dir,
):
    fig, axes = plt.subplots(2, 3, figsize=(21, 14))

    plot_boundary_nodes_fn(reference_assignment, axes[0, 0],
                           "Reference Partition\n(density assignment only)")
    plot_boundary_nodes_fn(initial_assignment,   axes[0, 1],
                           "Random Initial Partition\n(chain starting state)")
    plot_boundary_nodes_fn(recom_final,          axes[0, 2],
                           "ReCom Final — Boundary Nodes")
    plot_boundary_nodes_fn(qp_final,             axes[1, 0],
                           "QP-MH Final — Boundary Nodes")
    plot_boundary_nodes_fn(mew_final,            axes[1, 1],
                           "MEW Final — Boundary Nodes")
    plot_boundary_nodes_fn(hyb_final,            axes[1, 2],
                           "Hybrid Final — Boundary Nodes")

    plt.tight_layout()
    plt.savefig(plots_dir / "boundary_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: boundary_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure: metric comparison (1 × 3)
# ─────────────────────────────────────────────────────────────────────────────

_DS_FORMULA = (
    r"$DS(x)=\dfrac{\sum_{e\,\mathrm{intra}}w(e)}{\sum_{e\,\mathrm{all}}w(e)}$"
    "\n"
    r"$w(e)=\exp(-|\Delta\mathrm{density}|)$"
)


def _ext(ml, key):
    return [m[key] for m in ml]


def plot_metric_comparison(recom_metrics, qp_metrics, mew_metrics, hyb_metrics,
                           plots_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, key, label, hint in [
        (axes[0], "density_score", "Density Score",        "higher → density aligned"),
        (axes[1], "pp",            "Polsby-Popper",         "higher → compact"),
        (axes[2], "pop_dev",       "Population Deviation",  "lower → balanced"),
    ]:
        ax.plot(range(1, 21), _ext(recom_metrics, key), "r-s", ms=4, label="ReCom")
        ax.plot(range(1, 21), _ext(qp_metrics,    key), "b-o", ms=4, label="QP-MH")
        ax.plot(range(1, 21), _ext(mew_metrics,   key), "m-D", ms=4, label="MEW")
        ax.plot(range(1, 11), _ext(hyb_metrics,   key), "g-^", ms=4, label="Hybrid")
        ax.set_xlabel("Step / Round")
        ax.set_ylabel(label)
        ax.set_title(f"{label}  ({hint})")
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[0].text(
        0.97, 0.05, _DS_FORMULA,
        transform=axes[0].transAxes,
        ha="right", va="bottom", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray", alpha=0.85),
    )

    plt.tight_layout()
    plt.savefig(plots_dir / "metric_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: metric_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure: mixing-time diagnostics (6-panel)
# ─────────────────────────────────────────────────────────────────────────────

COLS = {
    "ReCom":  "tomato",
    "QP-MH":  "steelblue",
    "MEW":    "mediumpurple",
    "Hybrid": "mediumseagreen",
}


def plot_mixing_time(
    recom_trace, qp_trace, mew_trace, hyb_trace,
    recom_wt, qp_wt, mew_wt, hyb_wt,
    LONG, MAX_LAG, plots_dir,
):
    tau_r, ess_r = tau_and_ess(recom_trace, MAX_LAG)
    tau_q, ess_q = tau_and_ess(qp_trace,   MAX_LAG)
    tau_m, ess_m = tau_and_ess(mew_trace,  MAX_LAG)
    tau_h, ess_h = tau_and_ess(hyb_trace,  MAX_LAG)

    eps_r = ess_per_second(ess_r, recom_wt)
    eps_q = ess_per_second(ess_q, qp_wt)
    eps_m = ess_per_second(ess_m, mew_wt)
    eps_h = ess_per_second(ess_h, hyb_wt)

    steps = np.arange(1, LONG + 1)
    lags  = np.arange(MAX_LAG)

    fig = plt.figure(figsize=(22, 16))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.40, wspace=0.30)

    ax_trace = fig.add_subplot(gs[0, :])
    ax_acf   = fig.add_subplot(gs[1, 0])
    ax_ess   = fig.add_subplot(gs[1, 1])
    ax_rm    = fig.add_subplot(gs[2, 0])
    ax_tv    = fig.add_subplot(gs[2, 1])

    # ── Trace ─────────────────────────────────────────────────────────────────
    for trace, label in [(recom_trace, "ReCom"), (qp_trace, "QP-MH"),
                         (mew_trace, "MEW"),   (hyb_trace, "Hybrid")]:
        c = COLS[label]
        tau_v, _ = tau_and_ess(trace, MAX_LAG)
        ax_trace.plot(steps, trace, color=c, alpha=0.40, lw=0.8)
        ax_trace.plot(steps, running_mean(trace), color=c, lw=2.2,
                      label=f"{label} (τ={tau_v:.1f})", ls="--")

    ax_trace.set_xlabel("Step")
    ax_trace.set_ylabel("Density Score")
    ax_trace.set_title(
        "Trace & Running Mean of Density Score\n"
        r"$DS(x)=\Sigma_{\rm intra}w(e)\,/\,\Sigma_{\rm all}w(e)$,"
        "  $w(e)=\\exp(-|\\Delta\\mathrm{density}|)$"
        "\n(dashed = running mean;  higher & stable = better alignment)"
    )
    ax_trace.legend(ncol=2)
    ax_trace.grid(True, alpha=0.3)

    # ── ACF ───────────────────────────────────────────────────────────────────
    for trace, label in [(recom_trace, "ReCom"), (qp_trace, "QP-MH"),
                         (mew_trace, "MEW"),   (hyb_trace, "Hybrid")]:
        _, ess_v = tau_and_ess(trace, MAX_LAG)
        ax_acf.plot(lags, acf(trace, MAX_LAG), color=COLS[label], lw=1.5,
                    label=f"{label}  ESS={ess_v:.0f}")

    ax_acf.axhline(0, color="k", lw=0.8, ls="--")
    ax_acf.fill_between(lags, -1.96 / np.sqrt(LONG), 1.96 / np.sqrt(LONG),
                        color="grey", alpha=0.15, label="95% CI (white noise)")
    ax_acf.set_xlabel("Lag")
    ax_acf.set_ylabel("Autocorrelation")
    ax_acf.set_title("ACF of Density Score\n(faster decay to zero = better mixing)")
    ax_acf.legend()
    ax_acf.grid(True, alpha=0.3)

    # ── ESS & ESS/second bars ──────────────────────────────────────────────────
    methods  = ["ReCom", "QP-MH", "MEW", "Hybrid"]
    ess_vals = [ess_r,   ess_q,   ess_m, ess_h]
    eps_vals = [eps_r,   eps_q,   eps_m, eps_h]
    bar_cols = [COLS[m] for m in methods]

    x      = np.arange(len(methods))
    width  = 0.35
    bars1  = ax_ess.bar(x - width / 2, ess_vals, width, color=bar_cols,
                        alpha=0.85, edgecolor="k", lw=0.8, label="ESS")
    ax_ess.bar(x + width / 2, eps_vals, width, color=bar_cols,
               alpha=0.45, edgecolor="k", lw=0.8, hatch="//",
               label="ESS / wall-second")
    ax_ess.set_xticks(x)
    ax_ess.set_xticklabels(methods)
    ax_ess.set_ylabel("Value")
    ax_ess.legend()
    ax_ess.set_title(
        f"ESS and ESS/second over {LONG} steps  (Density Score)\n"
        "(higher = better mixing / more efficient)"
    )
    for bar, val in zip(bars1, ess_vals):
        ax_ess.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.0f}", ha="center", fontsize=9, fontweight="bold")
    ax_ess.grid(True, alpha=0.3, axis="y")

    # ── Running mean ───────────────────────────────────────────────────────────
    for trace, label in [(recom_trace, "ReCom"), (qp_trace, "QP-MH"),
                         (mew_trace, "MEW"),   (hyb_trace, "Hybrid")]:
        ax_rm.plot(steps, running_mean(trace), color=COLS[label], lw=1.8, label=label)
    ax_rm.set_xlabel("Step")
    ax_rm.set_ylabel("Running Mean  Density Score")
    ax_rm.set_title("Running Mean — convergence speed\n(higher = better density alignment)")
    ax_rm.legend()
    ax_rm.grid(True, alpha=0.3)

    # ── TV-distance proxy (log scale) ──────────────────────────────────────────
    tv_steps = np.arange(20, LONG) + 1
    for trace, label in [(recom_trace, "ReCom"), (qp_trace, "QP-MH"),
                         (mew_trace, "MEW"),   (hyb_trace, "Hybrid")]:
        tv = tv_proxy(trace)
        tv = np.where(tv > 0, tv, 1e-10)
        ax_tv.semilogy(tv_steps, tv, color=COLS[label], lw=1.5, label=label)
    ax_tv.set_xlabel("Step")
    ax_tv.set_ylabel("Running-Mean Fluctuation  (log)")
    ax_tv.set_title("TV-Distance Proxy (Density Score)\nfaster decay = faster convergence to π")
    ax_tv.legend()
    ax_tv.grid(True, alpha=0.3)

    # ── Formula footer ─────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.01,
        r"Density Score:  $DS(x)=\dfrac{\sum_{e\;\mathrm{intra\text{-}district}}w(e)}"
        r"{\sum_{e\;\mathrm{all\;edges}}w(e)}$,  "
        r"$w(e)=\exp\!\left(-\,|\Delta\mathrm{density}|\right)$  "
        r"$\in[0,1]$  —  higher is better",
        ha="center", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="gray", alpha=0.9),
    )

    plt.savefig(plots_dir / "mixing_time_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: mixing_time_analysis.png")

    # Return stats for the summary table / console
    return (tau_r, tau_q, tau_m, tau_h,
            ess_r, ess_q, ess_m, ess_h,
            eps_r, eps_q, eps_m, eps_h)
