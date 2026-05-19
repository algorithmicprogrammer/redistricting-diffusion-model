
<br />
<div align="center">

  <h3 align="center">A Diffusion Model for Political Redistricting</h3>

  <p align="center">
    Laplacian diffusion model 
    <br />
    <br />
    <br />
    <a href="https://github.com/algorithmicprogrammer/redistricting-diffusion-model/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/algorithmicprogrammer/redistricting-diffusion-model/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
     <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
<li>
      <a href="#running-the-5x5-fiber-visualization">Running the 5x5 Fiber Visualization</a>
      <ul>
        <li><a href="#sample-mode">Sample Mode</a></li>
        <li><a href="#full-enumeration">Full Enumeration Mode</a></li> 
        <li><a href="#output-directory">Output Directory</a></li>
        <li><a href="#parameters">Parameters</a></li>
      </ul>
    </li>
    <li><a href="#project-organization">Project Organization</a></li>
  </ol>
</details>

## Getting Started
### Prerequisites
1. Install git (Debian/Ubuntu).
```
sudo apt install git
```

### Installation
1. Clone the repository.
```
git clone https://github.com/algorithmicprogrammer/redistricting-diffusion-model.git
```

2. Navigate to the cloned repository. 
```
cd redistricting-diffusion-model
```

3. Create a Python virtual environment (MacOS/Linux):
```
python3 -m venv venv
```

4. Activate virtual environment (MacOS/Linux).
```
source venv/bin/activate
```

5. Install requirements.
```
pip install -r requirements.txt
```

## Running the 5x5 Fiber Visualization

This script generates visualizations and metrics for the fiber of connected district plans on a `5x5` grid. Each valid plan has:

- 5 connected districts
- 5 unit-population nodes per district
- Canonical unlabeled plans, so district-label permutations are not duplicated

Run the script from the repository root.

### Sample Mode

Sample mode draws a specified number of plans from the fiber using the sampler.

```bash
python experiments/fiber_5x5.py --mode sample --samples 36
```

### Full Enumeration Mode
Full mode enumerates the full fiber of valid plans.
```
python experiments/fiber_5x5.py --mode full
```

### Output Directory
By default, outputs are written to:
```
outputs/fiber_5x5/
```

You can change the output directory with:
```
python experiments/fiber_5x5.py --mode sample --out outputs/my_fiber_run
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---:|---|
| `--mode` | string | `full` | Which run mode to use. Options are `sample` or `full`. |
| `--samples` | int | `36` | Number of plans to generate in sample mode. Ignored in full mode. |
| `--burn-in` | int | `200` | Number of initial sampler steps to discard before collecting samples. Used only in sample mode. |
| `--thinning` | int | `20` | Number of sampler steps between saved plans. Used only in sample mode. |
| `--seed` | int | `42` | Random seed used for grid construction and sampling. |
| `--max-plans` | int or `None` | `None` | Optional cap on the number of plans to enumerate in full mode. Useful for debugging. |
| `--gallery-cols` | int | `6` | Number of columns in the gallery image. |
| `--gallery-rows` | int | `6` | Number of rows per gallery page in full mode. |
| `--max-gallery-pages` | int or `all` | `all` | In full mode, controls how many gallery PNG pages to write. Use `all` to write every page. |
| `--out` | path | `outputs/fiber_5x5` | Directory where output files are saved. |

## Project Organization

This repository is organized around a small redistricting diffusion model package, an executable experiment script, generated outputs, and supporting data.

```text
redistricting-diffusion-model/
├── README.md
├── LICENSE
├── Makefile
├── pyproject.toml
├── requirements.txt
│
├── redistricting_diffusion_model/
│   ├── __init__.py
│   ├── config.py
│   ├── grid.py
│   ├── fiber.py
│   ├── metrics.py
│   ├── laplacian.py
│   ├── qp.py
│   └── visualization.py
│
├── experiments/
│   └── fiber_5x5.py
│
├── data/
│   ├── raw/
│   │   └── IA_counties/
│   ├── interim/
│   ├── processed/
│   └── external/
│
├── outputs/
│   └── fiber_5x5/
│       ├── initial_partition.png
│       ├── boundary_nodes.png
│       ├── fiber_gallery.png
│       ├── fiber_projection.png
│       ├── metrics.csv
│       ├── summary.json
│       ├── full_fiber_gallery_*.png
│       ├── full_fiber_metrics.csv
│       ├── full_fiber_projection.png
│       └── full_fiber_summary.json
│
├── reports/
│   └── figures/
│
├── notebooks/
│
├── tests/
│   └── test_data.py
│
└── legacy/
    ├── diffusion_model/
    ├── plots/
    ├── qp_boundary_visualisation.py
    ├── qp_fast_sweep.py
    ├── qp_parameter_tuning.py
    └── weight.py
```
---
Made with ♥ by <a href="https://github.com/kirtisoglu">@kirtisoglu</a> &  <a href="https://github.com/algorithmicprogrammer">@algorithmicprogrammer</a>
