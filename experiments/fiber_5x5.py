"""Generate a 5x5 grid fiber visualization.

Run from the repository root:
    python experiments/fiber_5x5.py --mode sample --samples 36
    python experiments/fiber_5x5.py --mode full

Outputs are written to outputs/fiber_5x5/ by default.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from redistricting_diffusion_model.fiber import enumerate_full_fiber, sample_fiber
from redistricting_diffusion_model.grid import GridConfig, make_grid, vertical_stripes
from redistricting_diffusion_model.metrics import record_metrics
from redistricting_diffusion_model.visualization import (
    draw_boundary_nodes,
    draw_partition,
    plot_fiber_gallery,
    plot_fiber_gallery_pages,
    plot_metric_projection,
)


def write_metrics(graph, plans, path: Path) -> list[dict]:
    rows = []
    for i, asn in enumerate(plans, start=1):
        row = {"plan": i}
        row.update(record_metrics(graph, asn))
        rows.append(row)

    if not rows:
        return rows

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sample", "full"], default="full")
    parser.add_argument("--samples", type=int, default=36, help="Number of plans for sample mode.")
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--thinning", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-plans", type=int, default=None, help="Optional cap for debugging full enumeration.")
    parser.add_argument("--gallery-cols", type=int, default=6)
    parser.add_argument("--gallery-rows", type=int, default=6)
    parser.add_argument("--max-gallery-pages", default="all", help="For full mode: number of gallery PNG pages to write, or all.")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "fiber_5x5")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    graph = make_grid(GridConfig(width=5, height=5, num_districts=5, population=1, seed=args.seed))
    initial = vertical_stripes(graph, num_districts=5)

    fig, ax = plt.subplots(figsize=(5, 5))
    draw_partition(graph, initial, ax, "Initial 5x5 vertical-stripe plan")
    fig.savefig(args.out / "initial_partition.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    draw_boundary_nodes(graph, initial, ax, "Boundary nodes of initial plan")
    fig.savefig(args.out / "boundary_nodes.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    if args.mode == "full":
        plans = enumerate_full_fiber(
            graph,
            district_size=5,
            num_districts=5,
            max_plans=args.max_plans,
        )
        per_page = args.gallery_cols * args.gallery_rows
        if args.max_gallery_pages.lower() == "all":
            gallery_plans = plans
        else:
            gallery_plans = plans[: per_page * int(args.max_gallery_pages)]
        gallery_paths = plot_fiber_gallery_pages(
            graph,
            gallery_plans,
            args.out,
            prefix="full_fiber_gallery",
            ncols=args.gallery_cols,
            nrows=args.gallery_rows,
        )
        projection_path = args.out / "full_fiber_projection.png"
        metrics_path = args.out / "full_fiber_metrics.csv"
    else:
        plans = sample_fiber(
            graph,
            initial,
            district_size=5,
            n_samples=args.samples,
            burn_in=args.burn_in,
            thinning=args.thinning,
            seed=args.seed,
        )
        plot_fiber_gallery(graph, plans, args.out / "fiber_gallery.png", ncols=args.gallery_cols)
        gallery_paths = [args.out / "fiber_gallery.png"]
        projection_path = args.out / "fiber_projection.png"
        metrics_path = args.out / "metrics.csv"

    plot_metric_projection(graph, plans, projection_path)
    rows = write_metrics(graph, plans, metrics_path)

    summary = {
        "grid": "5x5",
        "mode": args.mode,
        "fiber_definition": "five connected districts, exactly five unit-population nodes per district",
        "n_plans": len(plans),
        "labeling_convention": "canonical unlabeled plans; district-label permutations are not duplicated",
        "initial_metrics": record_metrics(graph, initial),
        "metrics_csv": str(metrics_path),
        "projection_png": str(projection_path),
        "gallery_files": [str(p) for p in gallery_paths],
        "gallery_note": "In full mode, metric files include all plans; gallery pages may be capped by --max-gallery-pages.",
    }
    if rows:
        summary["best_by_cut_edges"] = min(rows, key=lambda r: r["cut_edges"])
        summary["best_by_polsby_popper"] = max(rows, key=lambda r: r["polsby_popper"])

    summary_path = args.out / ("full_fiber_summary.json" if args.mode == "full" else "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote outputs to {args.out}")


if __name__ == "__main__":
    main()
